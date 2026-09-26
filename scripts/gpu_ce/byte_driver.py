"""CL-033 Track B: byte-level cross-encoder trained from scratch (no pretrained vocabulary), SageMaker 1 GPU.

Input: UTF-8 bytes of '<q name> | <q address>' [SEP] '<s2|s3> <t name> | <t address>', truncated to max_len bytes.
Model: byte embedding + learned positions + segment embedding, pre-LN Transformer encoder, CLS -> logit (BCE).
Training pairs: stage-1 band (base >= band) or positive pairs of config.train_folds, restricted to config.train_countries,
owner-safe (targets owned by held-fold S1 removed). Scores all held-fold band pairs (both countries) after each epoch.
Channels: code, train (raw train TSVs + ground truth), top12. Fold4 absent.
"""
import glob, json, math, random, time
from pathlib import Path
import numpy as np, pandas as pd, torch

T0 = time.time(); ROOT = Path("/opt/ml/input/data")


def log(**kw): print(json.dumps({"t": round(time.time() - T0, 1), **kw}), flush=True)


class ByteCE(torch.nn.Module):
    def __init__(self, L, d, layers, heads):
        super().__init__()
        self.emb = torch.nn.Embedding(260, d); self.pos = torch.nn.Embedding(L, d); self.seg = torch.nn.Embedding(2, d)
        layer = torch.nn.TransformerEncoderLayer(d, heads, 4 * d, dropout=0.1, batch_first=True, norm_first=True, activation="gelu")
        self.enc = torch.nn.TransformerEncoder(layer, layers); self.norm = torch.nn.LayerNorm(d); self.head = torch.nn.Linear(d, 1)

    def forward(self, x, s, pad):
        p = torch.arange(x.shape[1], device=x.device)[None]
        h = self.enc(self.emb(x) + self.pos(p) + self.seg(s), src_key_padding_mask=pad)
        return self.head(self.norm(h[:, 0])).squeeze(-1)


def encode(qs, ts, L):
    # 256 CLS, 257 SEP, 258 PAD
    X = np.full((len(qs), L), 258, np.int64); S = np.zeros((len(qs), L), np.int64)
    half = (L - 2) // 2
    for i, (q, t) in enumerate(zip(qs, ts)):
        qb = list(q.encode("utf-8"))[:half]; tb = list(t.encode("utf-8"))[: L - 2 - len(qb)]
        seq = [256] + qb + [257] + tb; X[i, :len(seq)] = seq; S[i, len(qb) + 2:len(seq)] = 1
    return X, S


def main():
    cfg = json.loads((ROOT / "code/config.json").read_text()); log(event="config", **cfg)
    random.seed(cfg["seed"]); np.random.seed(cfg["seed"]); torch.manual_seed(cfg["seed"]); dev = torch.device("cuda")
    tx = {}
    for i in (1, 2, 3):
        df = pd.read_csv(ROOT / f"train/train_source{i}.tsv", sep="\t", quoting=3, dtype=str, keep_default_na=False, na_filter=False,
                         usecols=["entity_id", "business_name", "business_address"] + (["country"] if i == 1 else []))
        tx.update(zip(df["entity_id"], df["business_name"] + " | " + df["business_address"]))
        if i == 1: s1c = dict(zip(df["entity_id"], df["country"]))
    top = pd.concat([pd.read_parquet(p) for p in sorted(glob.glob(str(ROOT / "top12/top12-fold*.parquet")))], ignore_index=True)
    gt = pd.read_csv(ROOT / "train/train_ground_truth.tsv", sep="\t", quoting=3, dtype=str, keep_default_na=False)
    held = cfg["held_fold"]; held_s1 = set(top.loc[top["fold"] == held, "source1_entity_id"])
    owned = set(t for q, r in zip(gt["source1_entity_id"], gt["matched_entity_ids"]) if q in held_s1 and r for t in r.split(","))
    ctry = top["source1_entity_id"].map(s1c)
    tr = top[top["fold"].isin(cfg["train_folds"]) & ctry.isin(cfg["train_countries"]) & ((top["base_score"] >= cfg["band"]) | (top["label"] == 1))]
    tr = tr[~tr["target_id"].isin(owned)]
    ho = top[(top["fold"] == held) & (top["base_score"] >= cfg["band"])]
    L = cfg["max_len"]
    pre = lambda t: ("s2 " if t.startswith("S2-") else "s3 ") + tx[t]
    Xtr, Str = encode([tx[q] for q in tr["source1_entity_id"]], [pre(t) for t in tr["target_id"]], L); y = torch.tensor(tr["label"].to_numpy(), dtype=torch.float32)
    Xho, Sho = encode([tx[q] for q in ho["source1_entity_id"]], [pre(t) for t in ho["target_id"]], L)
    log(event="pairs", train=len(tr), pos=int(tr["label"].sum()), held=len(ho))
    model = ByteCE(L, cfg["d"], cfg["layers"], cfg["heads"]).to(dev)
    log(event="params", n=sum(p.numel() for p in model.parameters()))
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=0.01)
    bs = cfg["batch"]; total = math.ceil(len(Xtr) / bs) * cfg["epochs"]; warm = max(1, int(0.03 * total))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / warm) * max(0.02, 0.5 * (1 + math.cos(math.pi * s / total))))
    s3 = __import__("boto3").client("s3"); b, k = cfg["s3_out"].replace("s3://", "").split("/", 1); step = 0
    for ep in range(cfg["epochs"]):
        model.train(); perm = np.random.permutation(len(Xtr)); t0 = time.time()
        for i in range(0, len(perm), bs):
            idx = perm[i:i + bs]; x = torch.from_numpy(Xtr[idx]).to(dev); s = torch.from_numpy(Str[idx]).to(dev)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                lg = model(x, s, x == 258)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(lg.float(), y[idx].to(dev))
            opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sched.step(); step += 1
            if step % 500 == 0: log(event="train", ep=ep, step=step, total=total, loss=float(loss), pairs_s=round((i + bs) / (time.time() - t0)))
        model.eval(); out = np.zeros(len(Xho), np.float32)
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            for i in range(0, len(Xho), 4096):
                x = torch.from_numpy(Xho[i:i + 4096]).to(dev); s = torch.from_numpy(Sho[i:i + 4096]).to(dev)
                out[i:i + 4096] = model(x, s, x == 258).float().cpu().numpy()
        name = f"held-fold{held}-ep{ep}.parquet"; p = Path("/tmp") / name
        pd.DataFrame({"q": ho["source1_entity_id"].values, "t": ho["target_id"].values, "logit": out, "label": ho["label"].values}).to_parquet(p, index=False)
        s3.upload_file(str(p), b, k.rstrip("/") + "/" + name); log(event="held", ep=ep, pos_mean=float(out[ho["label"].values == 1].mean()), neg_mean=float(out[ho["label"].values == 0].mean()))
    log(event="complete")


if __name__ == "__main__":
    main()
