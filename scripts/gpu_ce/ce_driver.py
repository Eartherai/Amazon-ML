"""CL-014 scaled cross-encoder: train on held-fold-OOF top-12 hard candidates, score held fold + test band.

Runs as a SageMaker Training job (single GPU) or locally for smoke tests (--local).
Channels (under /opt/ml/input/data): code (this file + config.json), train (raw train TSVs + ground
truth), top12 (top12-fold{1,2,3}.parquet: source1_entity_id,target_id,label,base_score,rank,fold),
test12 (192 *-scores.tsv.gz: source1_entity_id,target_id,score), test (raw test TSVs).

Leakage control: training S1 are in config.train_folds only; every pair whose target is owned (ground
truth) by an S1 of the held fold is removed from training. Fold4 is absent from the 200k store.
Pair text: query '<name> | <address>', target '<s2|s3>: <name> | <address>' (raw fields, EXP-050 format).
Outputs (streamed to S3 as each part completes): held-fold and test logits as parquet parts.
"""
from __future__ import annotations

import argparse
import glob
import gzip
import io
import json
import math
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch


def log(**kw):
    print(json.dumps({"t": round(time.time() - T0, 1), **kw}), flush=True)


T0 = time.time()


def read_tsv(path, cols=("entity_id", "business_name", "business_address")):
    df = pd.read_csv(path, sep="\t", quoting=3, dtype=str, keep_default_na=False, na_filter=False, usecols=list(cols))
    return df


def text_map(files):
    out = {}
    for f in files:
        df = read_tsv(f)
        out.update(zip(df["entity_id"], (df["business_name"] + " | " + df["business_address"])))
    return out


class PairModel(torch.nn.Module):
    def __init__(self, model_id, revision):
        super().__init__()
        from transformers import AutoModel
        self.encoder = AutoModel.from_pretrained(model_id, revision=revision)
        self.head = torch.nn.Linear(self.encoder.config.hidden_size, 1)

    def forward(self, ids, mask):
        h = self.encoder(input_ids=ids, attention_mask=mask).last_hidden_state[:, 0]
        return self.head(h.float()).squeeze(-1)


def encode(tok, q, t, max_len):
    enc = tok(q, t, truncation="longest_first", max_length=max_len, padding=False)
    return enc["input_ids"]


def batches_by_length(ids, bs, shuffle, seed=0):
    order = np.argsort([len(x) for x in ids], kind="stable")
    chunks = [order[i:i + bs] for i in range(0, len(order), bs)]
    if shuffle:
        random.Random(seed).shuffle(chunks)
    return chunks


def collate(ids, idx, pad_id, device):
    seqs = [ids[i] for i in idx]
    L = max(len(s) for s in seqs)
    arr = np.full((len(seqs), L), pad_id, dtype=np.int64)
    mask = np.zeros((len(seqs), L), dtype=np.int64)
    for j, s in enumerate(seqs):
        arr[j, :len(s)] = s
        mask[j, :len(s)] = 1
    return torch.from_numpy(arr).to(device, non_blocking=True), torch.from_numpy(mask).to(device, non_blocking=True)


def score(model, ids, pad_id, device, bs, amp_dtype):
    model.eval()
    out = np.zeros(len(ids), dtype=np.float32)
    with torch.inference_mode():
        for idx in batches_by_length(ids, bs, shuffle=False):
            x, m = collate(ids, idx, pad_id, device)
            with torch.autocast(device.type, dtype=amp_dtype, enabled=device.type == "cuda"):
                out[idx] = model(x, m).float().cpu().numpy()
    return out


class Sink:
    """Write parquet parts locally and (on SageMaker) upload each immediately."""

    def __init__(self, local_dir, s3_prefix):
        self.dir = Path(local_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.s3_prefix = s3_prefix
        self.s3 = None
        if s3_prefix:
            import boto3
            self.s3 = boto3.client("s3")

    def put(self, name, df):
        path = self.dir / name
        df.to_parquet(path, index=False)
        if self.s3:
            bucket, key = self.s3_prefix.replace("s3://", "").split("/", 1)
            self.s3.upload_file(str(path), bucket, key.rstrip("/") + "/" + name)
        log(event="part", name=name, rows=len(df))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--local", action="store_true")
    ap.add_argument("--root", default="/opt/ml/input/data")
    a, _ = ap.parse_known_args()
    root = Path(a.root)
    cfg = json.loads((root / "code/config.json").read_text())
    log(event="config", **cfg)
    random.seed(cfg["seed"]); np.random.seed(cfg["seed"]); torch.manual_seed(cfg["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_dtype = torch.bfloat16
    sink = Sink(cfg.get("local_out", "/opt/ml/output/data/parts"), None if a.local else cfg["s3_out"])

    if cfg.get("score_only"):
        import tarfile
        from transformers import AutoTokenizer
        tar = sorted(glob.glob(str(root / "init" / "*.tar.gz")))[0]
        with tarfile.open(tar) as tf: tf.extractall("/tmp/init")
        tok = AutoTokenizer.from_pretrained("/tmp/init/encoder", use_fast=True); pad_id = tok.pad_token_id
        model = PairModel("/tmp/init/encoder", None).to(device)
        model.head.load_state_dict(torch.load("/tmp/init/head.pt", map_location="cpu"))
        log(event="loaded_ce", tar=tar)
        for spec in cfg["pair_sets"]:
            split = spec["split"]
            texts = text_map([root / f"{split}/{split}_source{i}.tsv" for i in (1, 2, 3)])
            for fp in sorted(glob.glob(str(root / "pairs" / spec["glob"]))):
                df = pd.read_parquet(fp)
                qs = [texts[q] for q in df["q"]]
                ts = [("s2: " if t.startswith("S2-") else "s3: ") + texts[t] for t in df["t"]]
                ids = []
                for i in range(0, len(qs), 50000):
                    ids += encode(tok, qs[i:i + 50000], ts[i:i + 50000], cfg["max_len"])
                sc = score(model, ids, pad_id, device, cfg["infer_batch"], amp_dtype)
                sink.put(f"{spec['name']}-{Path(fp).stem}.parquet", pd.DataFrame({"q": df["q"].values, "t": df["t"].values, "logit": sc}))
        log(event="complete"); return
    top = pd.concat([pd.read_parquet(p) for p in sorted(glob.glob(str(root / "top12/top12-fold*.parquet")))], ignore_index=True)
    if cfg.get("max_s1"):
        keep = set(sorted(top["source1_entity_id"].unique())[: cfg["max_s1"]])
        top = top[top["source1_entity_id"].isin(keep)]
    held = cfg["held_fold"]
    gt = pd.read_csv(root / "train/train_ground_truth.tsv", sep="\t", quoting=3, dtype=str, keep_default_na=False)
    held_s1 = set(top.loc[top["fold"] == held, "source1_entity_id"])
    owned_by_held = set()
    for q, raw in zip(gt["source1_entity_id"], gt["matched_entity_ids"]):
        if q in held_s1 and raw:
            owned_by_held.update(raw.split(","))
    band = cfg["band"]
    if cfg.get("train_countries"):
        s1c = read_tsv(root / "train/train_source1.tsv", cols=("entity_id", "country"))
        allowed = set(s1c.loc[s1c["country"].isin(cfg["train_countries"]), "entity_id"])
        top = top.assign(_tr_ok=top["source1_entity_id"].isin(allowed) | (top["fold"] == held))
    else:
        top = top.assign(_tr_ok=True)
    tr = top[top["fold"].isin(cfg["train_folds"]) & top["_tr_ok"] & ((top["base_score"] >= band) | (top["label"] == 1))]
    before = len(tr)
    tr = tr[~tr["target_id"].isin(owned_by_held)]
    ho = top[(top["fold"] == held) & (top["base_score"] >= band)]
    log(event="pairs", train=len(tr), train_removed_owner_safe=before - len(tr), train_pos=int(tr["label"].sum()), held=len(ho), held_pos=int(ho["label"].sum()))

    train_text = text_map([root / f"train/train_source{i}.tsv" for i in (1, 2, 3)])
    log(event="texts", n=len(train_text))
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(cfg["model_id"], revision=cfg["revision"], use_fast=True)
    pad_id = tok.pad_token_id

    def pair_ids(df, texts):
        qs = [texts[q] for q in df["source1_entity_id"]]
        ts = [("s2: " if t.startswith("S2-") else "s3: ") + texts[t] for t in df["target_id"]]
        out = []
        for i in range(0, len(qs), 50000):
            out += encode(tok, qs[i:i + 50000], ts[i:i + 50000], cfg["max_len"])
        return out

    tr_ids = pair_ids(tr, train_text)
    ho_ids = pair_ids(ho, train_text)
    y = torch.tensor(tr["label"].to_numpy(), dtype=torch.float32)
    log(event="tokenized", train=len(tr_ids), held=len(ho_ids), mean_len=float(np.mean([len(x) for x in tr_ids])))

    model = PairModel(cfg["model_id"], cfg["revision"]).to(device)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=cfg["lr"], weight_decay=0.01)
    steps_per_epoch = math.ceil(len(tr_ids) / cfg["batch"])
    total = steps_per_epoch * cfg["epochs"]
    warm = max(1, int(0.05 * total))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / warm) * max(0.0, (total - s) / total))
    step = 0
    for ep in range(cfg["epochs"]):
        model.train()
        losses = []
        t_ep = time.time()
        for idx in batches_by_length(tr_ids, cfg["batch"], shuffle=True, seed=cfg["seed"] + ep):
            x, m = collate(tr_ids, idx, pad_id, device)
            with torch.autocast(device.type, dtype=amp_dtype, enabled=device.type == "cuda"):
                logit = model(x, m)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(logit.float(), y[idx].to(device))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step(); step += 1
            losses.append(float(loss.detach()))
            if step % cfg["log_every"] == 0:
                el = time.time() - t_ep
                log(event="train", epoch=ep, step=step, total=total, loss=float(np.mean(losses[-cfg["log_every"]:])),
                    pairs_per_s=round(len(losses) * cfg["batch"] / el, 1), eta_epoch_min=round((steps_per_epoch - len(losses)) * el / len(losses) / 60, 1))
        # held-fold scores after every epoch (lets us pick the epoch without retraining)
        hs = score(model, ho_ids, pad_id, device, cfg["infer_batch"], amp_dtype)
        sink.put(f"held-fold{held}-ep{ep}.parquet", pd.DataFrame({"q": ho["source1_entity_id"].values, "t": ho["target_id"].values, "logit": hs, "label": ho["label"].values}))
        yh = ho["label"].to_numpy()
        rng = np.random.default_rng(0)
        pos, neg = hs[yh == 1], hs[yh == 0]
        auc_proxy = float((rng.choice(pos, 5000)[:, None] > rng.choice(neg, 5000)[None, :]).mean()) if len(pos) and len(neg) else None
        log(event="held", epoch=ep, pos_mean=float(hs[yh == 1].mean()), neg_mean=float(hs[yh == 0].mean()), auc_sample=auc_proxy)
    if not a.local:
        out = Path("/opt/ml/model")
        model.encoder.save_pretrained(out / "encoder")
        tok.save_pretrained(out / "encoder")
        torch.save(model.head.state_dict(), out / "head.pt")
    if cfg.get("score_test"):
        test_text = text_map([root / f"test/test_source{i}.tsv" for i in (1, 2, 3)])
        shards = sorted(glob.glob(str(root / "test12/*-scores.tsv.gz")))
        if cfg.get("max_test_shards"):
            shards = shards[: cfg["max_test_shards"]]
        for sp in shards:
            df = pd.read_csv(sp, sep="\t", quoting=3, dtype={"source1_entity_id": str, "target_id": str, "score": float})
            df = df[df["score"] >= band]
            if cfg.get("max_test_rows"):
                df = df.head(cfg["max_test_rows"])
            ids = pair_ids(df, test_text)
            s = score(model, ids, pad_id, device, cfg["infer_batch"], amp_dtype)
            sink.put(Path(sp).name.replace("-scores.tsv.gz", ".parquet"), pd.DataFrame({"q": df["source1_entity_id"].values, "t": df["target_id"].values, "base": df["score"].values, "logit": s}))
    log(event="complete")


if __name__ == "__main__":
    main()
