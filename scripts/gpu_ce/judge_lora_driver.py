"""CL-090: supervised hard-pair judge. Qwen3-4B (Apache-2.0, 4B) + LoRA as a single-logit MATCH classifier.

Trained on labeled hard pairs of train folds (config train_globs), scores held pairs (score_globs). Records never leave the
job; no external lookups. Channels: code (this file + config.json), data (parquet files: q, t, label?, q_name, q_addr,
t_name, t_addr). Outputs: parquet (q, t, logit) per scored file uploaded to s3_out, plus the LoRA adapter. Fold4 CLOSED.
"""
import glob, json, math, os, subprocess, sys, time
from pathlib import Path

T0 = time.time()


def log(**k):
    print(json.dumps({"t": round(time.time() - T0, 1), **k}), flush=True)


def main():
    root = Path("/opt/ml/input/data"); cfg = json.loads((root / "code" / os.environ.get("JUDGE_CONFIG", "config.json")).read_text()); log(event="config", **cfg)
    try:
        import peft  # noqa
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "peft==0.13.2"])
    import numpy as np, pandas as pd, torch, boto3
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    from peft import LoraConfig, get_peft_model
    torch.manual_seed(cfg.get("seed", 0)); np.random.seed(cfg.get("seed", 0))
    s3 = boto3.client("s3"); b, k = cfg["s3_out"].replace("s3://", "").split("/", 1)
    def put(p, name): s3.upload_file(str(p), b, k.rstrip("/") + "/" + name); log(event="uploaded", name=name)
    tok = AutoTokenizer.from_pretrained(cfg["model_id"], revision=cfg.get("revision")); tok.padding_side = "right"
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    model = AutoModelForSequenceClassification.from_pretrained(cfg["model_id"], revision=cfg.get("revision"), num_labels=1, torch_dtype=torch.bfloat16)
    model.config.pad_token_id = tok.pad_token_id
    lcfg = LoraConfig(r=cfg.get("lora_r", 16), lora_alpha=cfg.get("lora_alpha", 32), lora_dropout=0.05, task_type="SEQ_CLS",
                      target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
    if cfg.get("adapter_channel"):  # score-only mode: load a trained LoRA adapter (model.tar.gz from a previous job)
        import tarfile
        from peft import PeftModel
        tars = sorted(glob.glob(str(root / cfg["adapter_channel"] / "*.tar.gz")))
        with tarfile.open(tars[0]) as tf: tf.extractall("/tmp/adapter_pkg")
        model = PeftModel.from_pretrained(model, "/tmp/adapter_pkg/adapter").cuda()
    else:
        model = get_peft_model(model, lcfg).cuda(); model.print_trainable_parameters()
    if cfg.get("grad_ckpt", True):
        model.gradient_checkpointing_enable(); model.enable_input_require_grads()
    def text(df):
        src = df["t"].str[:2].str.lower()
        return ("record A: " + df["q_name"].fillna("") + " | " + df["q_addr"].fillna("") + "\nrecord B (" + src + "): " + df["t_name"].fillna("") + " | " + df["t_addr"].fillna("")).tolist()
    def enc(texts):
        return tok(texts, truncation=True, max_length=cfg.get("max_len", 160), padding=False)["input_ids"]
    def batches(ids, bs, order):
        for i in range(0, len(order), bs):
            idx = order[i:i + bs]; L = max(len(ids[j]) for j in idx)
            x = np.full((len(idx), L), tok.pad_token_id, np.int64); m = np.zeros((len(idx), L), np.int64)
            for r, j in enumerate(idx): x[r, :len(ids[j])] = ids[j]; m[r, :len(ids[j])] = 1
            yield idx, torch.from_numpy(x).cuda(), torch.from_numpy(m).cuda()
    # ---- train
    tr = pd.concat([pd.DataFrame(columns=["label"])]) if cfg.get("adapter_channel") else pd.concat([pd.read_parquet(p) for g in cfg["train_globs"] for p in sorted(glob.glob(str(root / "data" / g)))], ignore_index=True)
    if cfg.get("train_country"): tr = tr[tr["country"] == cfg["train_country"]]  # held-country transfer test
    if cfg.get("max_train"): tr = tr.sample(n=min(cfg["max_train"], len(tr)), random_state=0)
    if cfg.get("adapter_channel"): cfg["epochs"] = 0; tr = tr.iloc[:0]
    ids = enc(text(tr)) if len(tr) else []; y = torch.tensor(tr["label"].values.astype("float32") if len(tr) else [], dtype=torch.float32)
    log(event="train_data", rows=len(tr), pos=float(tr["label"].mean()) if len(tr) else None)
    tp_ = [p for p in model.parameters() if p.requires_grad] or [torch.nn.Parameter(torch.zeros(1, device="cuda"))]  # score-only mode has no trainable params
    opt = torch.optim.AdamW(tp_, lr=cfg.get("lr", 1e-4), weight_decay=0.0)
    bs, acc = cfg.get("batch", 16), cfg.get("accum", 2); epochs = cfg.get("epochs", 1)
    steps = math.ceil(len(ids) / bs) * epochs // acc; warm = max(1, int(0.03 * steps))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / warm) * max(0.05, 1 - s / max(steps, 1)))
    lossf = torch.nn.BCEWithLogitsLoss(); model.train(); step = 0; it = 0
    for ep in range(epochs):
        order = np.random.permutation(len(ids)); order = order[np.argsort([len(ids[j]) // 16 for j in order], kind="stable")]  # bucket by length
        chunks = [order[i:i + bs] for i in range(0, len(order), bs)]; np.random.shuffle(chunks)
        for idx in chunks:
            L = max(len(ids[j]) for j in idx)
            x = np.full((len(idx), L), tok.pad_token_id, np.int64); m = np.zeros((len(idx), L), np.int64)
            for r, j in enumerate(idx): x[r, :len(ids[j])] = ids[j]; m[r, :len(ids[j])] = 1
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = model(input_ids=torch.from_numpy(x).cuda(), attention_mask=torch.from_numpy(m).cuda()).logits[:, 0].float()
            loss = lossf(out, y[idx].cuda()) / acc; loss.backward(); it += 1
            if it % acc == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sched.step(); opt.zero_grad(set_to_none=True); step += 1
                if step % 50 == 0: log(event="train", step=step, total=steps, loss=float(loss.item() * acc))
            if cfg.get("max_minutes") and time.time() - T0 > 60 * cfg["max_minutes"]: log(event="time_cap"); break
    ad = Path("/opt/ml/model/adapter")
    if not cfg.get("adapter_channel"): model.save_pretrained(ad)
    # ---- score
    model.eval()
    for g in cfg["score_globs"]:
        for p in sorted(glob.glob(str(root / "data" / g))):
            df = pd.read_parquet(p); sids = enc(text(df)); order = np.argsort([len(s) for s in sids], kind="stable"); outv = np.zeros(len(sids), np.float32)
            with torch.inference_mode():
                for idx, x, m in batches(sids, cfg.get("infer_batch", 64), order):
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        outv[idx] = model(input_ids=x, attention_mask=m).logits[:, 0].float().cpu().numpy()
            name = "judge-" + Path(p).stem + ".parquet"; op = Path("/tmp") / name
            pd.DataFrame({"q": df["q"].values, "t": df["t"].values, "logit": outv}).to_parquet(op, index=False); put(op, name)
            log(event="scored", file=name, rows=len(df))
    log(event="complete")


if __name__ == "__main__":
    main()
