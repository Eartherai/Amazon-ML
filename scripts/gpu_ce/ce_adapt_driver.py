"""CL-038: target-domain adaptation of a trained CE with pseudo-labels (SageMaker, 1 GPU).

Starts from a trained CE checkpoint (init channel: encoder/ + head.pt), fine-tunes on
  source labeled pairs: top12 band pairs of config.train_folds in config.source_countries (owner-safe vs held fold), sampled to source_max
  target pseudo pairs: pseudo channel parquet (q, t, y) built WITHOUT target labels
then scores every held-fold band pair (both countries) -> held-fold{h}-adapt.parquet. Channels: code (+ ce_driver.py), train, top12, init, pseudo.
"""
import glob, json, math, random, sys, tarfile
from pathlib import Path
import numpy as np, pandas as pd, torch
sys.path.insert(0, "/opt/ml/input/data/code")
from ce_driver import PairModel, encode, batches_by_length, collate, score, text_map, read_tsv, log, Sink

ROOT = Path("/opt/ml/input/data")


def main():
    cfg = json.loads((ROOT / "code/config.json").read_text()); log(event="config", **cfg)
    random.seed(cfg["seed"]); np.random.seed(cfg["seed"]); torch.manual_seed(cfg["seed"]); dev = torch.device("cuda")
    with tarfile.open(sorted(glob.glob(str(ROOT / "init/*.tar.gz")))[0]) as tf: tf.extractall("/tmp/init")
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("/tmp/init/encoder", use_fast=True); pad = tok.pad_token_id
    model = PairModel("/tmp/init/encoder", None).to(dev); model.head.load_state_dict(torch.load("/tmp/init/head.pt", map_location="cpu"))
    top = pd.concat([pd.read_parquet(p) for p in sorted(glob.glob(str(ROOT / "top12/top12-fold*.parquet")))], ignore_index=True)
    gt = pd.read_csv(ROOT / "train/train_ground_truth.tsv", sep="\t", quoting=3, dtype=str, keep_default_na=False)
    held = cfg["held_fold"]; held_s1 = set(top.loc[top["fold"] == held, "source1_entity_id"])
    owned = set(t for q, r in zip(gt["source1_entity_id"], gt["matched_entity_ids"]) if q in held_s1 and r for t in r.split(","))
    s1c = read_tsv(ROOT / "train/train_source1.tsv", cols=("entity_id", "country")); allowed = set(s1c.loc[s1c["country"].isin(cfg["source_countries"]), "entity_id"])
    src = top[top["fold"].isin(cfg["train_folds"]) & top["source1_entity_id"].isin(allowed) & ((top["base_score"] >= cfg["band"]) | (top["label"] == 1))]
    src = src[~src["target_id"].isin(owned)]
    if len(src) > cfg["source_max"]: src = src.sample(cfg["source_max"], random_state=cfg["seed"])
    ps = pd.concat([pd.read_parquet(p) for p in glob.glob(str(ROOT / "pseudo/*.parquet"))], ignore_index=True)
    tr = pd.concat([src[["source1_entity_id", "target_id", "label"]].rename(columns={"source1_entity_id": "q", "target_id": "t", "label": "y"}), ps[["q", "t", "y"]]], ignore_index=True)
    ho = top[(top["fold"] == held) & (top["base_score"] >= cfg["band"])]
    log(event="pairs", source=len(src), pseudo=len(ps), pseudo_pos=int(ps["y"].sum()), held=len(ho))
    tx = text_map([ROOT / f"train/train_source{i}.tsv" for i in (1, 2, 3)])
    ids_of = lambda qs, ts: sum((encode(tok, [tx[q] for q in qs[i:i + 50000]], [("s2: " if t.startswith("S2-") else "s3: ") + tx[t] for t in ts[i:i + 50000]], cfg["max_len"]) for i in range(0, len(qs), 50000)), [])
    tr_ids = ids_of(tr["q"].tolist(), tr["t"].tolist()); ho_ids = ids_of(ho["source1_entity_id"].tolist(), ho["target_id"].tolist())
    y = torch.tensor(tr["y"].to_numpy(), dtype=torch.float32)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=0.01)
    total = math.ceil(len(tr_ids) / cfg["batch"]) * cfg["epochs"]; warm = max(1, int(0.05 * total))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / warm) * max(0.0, (total - s) / total)); step = 0
    sink = Sink("/opt/ml/output/data/parts", cfg["s3_out"])
    for ep in range(cfg["epochs"]):
        model.train()
        for idx in batches_by_length(tr_ids, cfg["batch"], shuffle=True, seed=cfg["seed"] + ep):
            x, m = collate(tr_ids, idx, pad, dev)
            with torch.autocast("cuda", dtype=torch.bfloat16): lg = model(x, m)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(lg.float(), y[idx].to(dev))
            opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sched.step(); step += 1
            if step % 200 == 0: log(event="train", ep=ep, step=step, total=total, loss=float(loss))
        hs = score(model, ho_ids, pad, dev, cfg["infer_batch"], torch.bfloat16)
        sink.put(f"held-fold{held}-adapt-ep{ep}.parquet", pd.DataFrame({"q": ho["source1_entity_id"].values, "t": ho["target_id"].values, "logit": hs, "label": ho["label"].values}))
    log(event="complete")


if __name__ == "__main__":
    main()
