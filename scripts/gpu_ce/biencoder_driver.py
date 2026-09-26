"""CL-015 dense bi-encoder retrieval rescue (SageMaker single GPU).

Train: InfoNCE with in-batch negatives on ground-truth positive (S1, target) pairs of S1 in
train_folds (all positives, including those the sparse retrieval missed). Pairs whose target is
owned by a held-fold S1 are excluded. Evaluate: embed held-fold S1 and ALL same-country train
targets; exact GPU top-K; report link recall overall and, crucially, recall of true links that are
NOT in the existing sparse top-12 (the misses) and candidates added per recovered link.
Optional test: embed test S1 + targets per country, emit top-K pairs not in the existing test
top-12 set, with cosine, streamed to S3.
E5 convention: 'query: ' / 'passage: ' prefixes, mean pooling, L2 normalisation.
"""
from __future__ import annotations

import argparse, glob, json, math, random, time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

T0 = time.time()


def log(**kw):
    print(json.dumps({"t": round(time.time() - T0, 1), **kw}), flush=True)


def read_src(path):
    return pd.read_csv(path, sep="\t", quoting=3, dtype=str, keep_default_na=False, na_filter=False)


class Enc(torch.nn.Module):
    def __init__(self, mid, rev):
        super().__init__()
        from transformers import AutoModel
        self.m = AutoModel.from_pretrained(mid, revision=rev)

    def forward(self, ids, mask):
        h = self.m(input_ids=ids, attention_mask=mask).last_hidden_state
        e = (h * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True).clamp(min=1)
        return torch.nn.functional.normalize(e.float(), dim=-1)


def tok_all(tok, texts, max_len):
    out = []
    for i in range(0, len(texts), 100000):
        out += tok(texts[i:i + 100000], truncation=True, max_length=max_len, padding=False)["input_ids"]
    return out


def collate(ids, idx, pad, dev):
    L = max(len(ids[i]) for i in idx)
    a = np.full((len(idx), L), pad, dtype=np.int64); m = np.zeros((len(idx), L), dtype=np.int64)
    for j, i in enumerate(idx):
        a[j, :len(ids[i])] = ids[i]; m[j, :len(ids[i])] = 1
    return torch.from_numpy(a).to(dev), torch.from_numpy(m).to(dev)


def embed(model, ids, pad, dev, bs):
    model.eval(); out = torch.empty((len(ids), model.m.config.hidden_size), dtype=torch.float16, device=dev)
    order = np.argsort([len(x) for x in ids], kind="stable")
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev.type == "cuda"):
        for s in range(0, len(order), bs):
            idx = order[s:s + bs]
            x, m = collate(ids, idx, pad, dev)
            out[torch.from_numpy(idx).to(dev)] = model(x, m).half()
    return out


def topk(Q, D, k, chunk=256):
    vals, idxs = [], []
    for s in range(0, Q.shape[0], chunk):
        sc = Q[s:s + chunk] @ D.T
        v, i = torch.topk(sc, k, dim=1)
        vals.append(v.float().cpu()); idxs.append(i.cpu())
    return torch.cat(vals).numpy(), torch.cat(idxs).numpy()


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--root", default="/opt/ml/input/data"); a, _ = ap.parse_known_args()
    root = Path(a.root); cfg = json.loads((root / "code/config.json").read_text()); log(event="config", **cfg)
    random.seed(cfg["seed"]); np.random.seed(cfg["seed"]); torch.manual_seed(cfg["seed"])
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    import boto3
    s3 = boto3.client("s3") if cfg.get("s3_out") else None

    def put(name, df):
        p = Path(cfg.get("out_dir", "/opt/ml/output/data")) / name; p.parent.mkdir(parents=True, exist_ok=True); df.to_parquet(p, index=False)
        if s3:
            b, k = cfg["s3_out"].replace("s3://", "").split("/", 1); s3.upload_file(str(p), b, k.rstrip("/") + "/" + name)
        log(event="part", name=name, rows=len(df))

    top = pd.concat([pd.read_parquet(p) for p in sorted(glob.glob(str(root / "top12/top12-fold*.parquet")))], ignore_index=True)
    if cfg.get("max_s1"):
        keep = set(sorted(top["source1_entity_id"].unique())[: cfg["max_s1"]]); top = top[top["source1_entity_id"].isin(keep)]
    fold = dict(zip(top["source1_entity_id"], top["fold"]))
    gt = read_src(root / "train/train_ground_truth.tsv")
    s1 = read_src(root / "train/train_source1.tsv"); tg = pd.concat([read_src(root / f"train/train_source{i}.tsv") for i in (2, 3)])
    s1_text = dict(zip(s1["entity_id"], "query: " + s1["business_name"] + " | " + s1["business_address"]))
    s1_c = dict(zip(s1["entity_id"], s1["country"]))
    tg_text = dict(zip(tg["entity_id"], "passage: " + tg["entity_id"].str[:2].str.lower() + ": " + tg["business_name"] + " | " + tg["business_address"]))
    tg_c = dict(zip(tg["entity_id"], tg["country"]))
    held = cfg["held_fold"]
    pos = [(q, t) for q, raw in zip(gt["source1_entity_id"], gt["matched_entity_ids"]) if raw and q in fold for t in raw.split(",")]
    held_owned = {t for q, t in pos if fold[q] == held}
    train_pos = [(q, t) for q, t in pos if fold[q] in cfg["train_folds"] and t not in held_owned]
    if cfg.get("train_countries"):  # CL-046 retrieval transfer: source-country positives only
        train_pos = [(q, t) for q, t in train_pos if s1_c[q] in cfg["train_countries"]]
    held_pos = [(q, t) for q, t in pos if fold[q] == held]
    in_top = set(zip(top["source1_entity_id"], top["target_id"]))
    held_miss = [(q, t) for q, t in held_pos if (q, t) not in in_top]
    log(event="pairs", train_pos=len(train_pos), held_pos=len(held_pos), held_miss_outside_top12=len(held_miss))
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(cfg["model_id"], revision=cfg["revision"], use_fast=True); pad = tok.pad_token_id
    model = Enc(cfg["model_id"], cfg["revision"]).to(dev)
    if cfg.get("init_tar"):
        import tarfile
        tar = sorted(glob.glob(str(root / "init" / "*.tar.gz")))[0]
        with tarfile.open(tar) as tf: tf.extractall("/tmp/init")
        from transformers import AutoModel
        model.m = AutoModel.from_pretrained("/tmp/init/encoder").to(dev)
        tok = AutoTokenizer.from_pretrained("/tmp/init/encoder", use_fast=True); pad = tok.pad_token_id
        log(event="loaded_init", tar=tar)
    if cfg.get("init_tar"):
        train_pos = train_pos[:0]
    if cfg.get("pseudo_glob"):  # CL-059: self-training on label-free target-domain pseudo-positives (q, t[, y])
        ps = pd.concat([pd.read_parquet(p) for p in glob.glob(str(root / "pseudo" / cfg["pseudo_glob"]))], ignore_index=True)
        if "y" in ps.columns: ps = ps[ps["y"] == 1]
        if cfg.get("pseudo_test"):  # CL-067: France pseudo pairs reference TEST records; add their texts (same formats)
            ts1_ = read_src(root / "test/test_source1.tsv"); ttg_ = pd.concat([read_src(root / f"test/test_source{i}.tsv") for i in (2, 3)])
            s1_text.update(dict(zip(ts1_["entity_id"], "query: " + ts1_["business_name"] + " | " + ts1_["business_address"])))
            tg_text.update(dict(zip(ttg_["entity_id"], "passage: " + ttg_["entity_id"].str[:2].str.lower() + ": " + ttg_["business_name"] + " | " + ttg_["business_address"])))
        train_pos = [(q, t) for q, t in zip(ps["q"], ps["t"]) if q in s1_text and t in tg_text]
        log(event="pseudo", pairs=len(train_pos))
    random.shuffle(train_pos)
    qi = tok_all(tok, [s1_text[q] for q, _ in train_pos], cfg["max_len"]); ti = tok_all(tok, [tg_text[t] for _, t in train_pos], cfg["max_len"])
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=0.01)
    bs = cfg["batch"]; steps = max(1, math.ceil(len(train_pos) / bs) * cfg["epochs"]); warm = max(1, int(0.05 * steps))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / warm) * max(0.0, (steps - s) / steps))
    step = 0; scale = 1.0 / cfg["temperature"]
    for ep in range(cfg["epochs"]):
        model.train(); perm = np.random.permutation(len(train_pos))
        for s in range(0, len(perm) - bs + 1, bs):
            idx = perm[s:s + bs]
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev.type == "cuda"):
                eq = model(*collate(qi, idx, pad, dev)); et = model(*collate(ti, idx, pad, dev))
            sim = eq @ et.T * scale
            # the same S1 can appear twice in a batch with different targets: mask those off-diagonal positives
            qs = np.array([train_pos[i][0] for i in idx]); same = torch.from_numpy(qs[:, None] == qs[None, :]).to(dev)
            sim = sim.masked_fill(same & ~torch.eye(len(idx), dtype=torch.bool, device=dev), -1e4)
            lab = torch.arange(len(idx), device=dev)
            loss = (torch.nn.functional.cross_entropy(sim, lab) + torch.nn.functional.cross_entropy(sim.T, lab)) / 2
            opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sched.step(); step += 1
            if step % cfg["log_every"] == 0:
                log(event="train", step=step, total=steps, loss=float(loss.detach()))
    K = cfg["k"]; report = {}
    held_s1 = sorted({q for q, _ in held_pos} | {q for q in top.loc[top["fold"] == held, "source1_entity_id"]})
    for country in ([] if cfg.get("skip_eval") else sorted(set(s1_c[q] for q in held_s1))):
        qs = [q for q in held_s1 if s1_c[q] == country]; ts = [t for t in tg_text if tg_c[t] == country]
        if cfg.get("max_targets"):
            need = {t for q, t in held_pos if s1_c[q] == country}
            ts = sorted(need | set(random.Random(0).sample(ts, min(cfg["max_targets"], len(ts)))))
        E_t = embed(model, tok_all(tok, [tg_text[t] for t in ts], cfg["max_len"]), pad, dev, cfg["infer_batch"])
        E_q = embed(model, tok_all(tok, [s1_text[q] for q in qs], cfg["max_len"]), pad, dev, cfg["infer_batch"])
        v, ix = topk(E_q, E_t, K); del E_t, E_q; torch.cuda.empty_cache()
        cand = {(qs[i], ts[j]) for i in range(len(qs)) for j in ix[i]}
        hp = [(q, t) for q, t in held_pos if s1_c[q] == country]; hm = [(q, t) for q, t in held_miss if s1_c[q] == country]
        new = [p for p in cand if p not in in_top]
        rec = {"s1": len(qs), "targets": len(ts), "link_recall_dense": sum(p in cand for p in hp) / max(len(hp), 1),
               "misses": len(hm), "misses_recovered": sum(p in cand for p in hm), "new_pairs": len(new)}
        for kk in (10, 20, 50):
            if kk <= K:
                ck = {(qs[i], ts[j]) for i in range(len(qs)) for j in ix[i][:kk]}
                rec[f"misses_recovered@{kk}"] = sum(p in ck for p in hm); rec[f"new_pairs@{kk}"] = sum(1 for p in ck if p not in in_top)
        report[country] = rec; log(event="eval", country=country, **rec)
        rows = [(qs[i], ts[j], float(v[i][r]), r + 1) for i in range(len(qs)) for r, j in enumerate(ix[i]) if (qs[i], ts[j]) not in in_top]
        put(f"held-fold{held}-{country}-dense.parquet", pd.DataFrame(rows, columns=["q", "t", "cos", "rank"]))
    (Path(cfg.get("out_dir", "/opt/ml/output/data"))).mkdir(parents=True, exist_ok=True)
    (Path(cfg.get("out_dir", "/opt/ml/output/data")) / "eval.json").write_text(json.dumps(report, indent=2))
    if s3:
        b, k = cfg["s3_out"].replace("s3://", "").split("/", 1); s3.upload_file(str(Path(cfg.get("out_dir", "/opt/ml/output/data")) / "eval.json"), b, k.rstrip("/") + "/eval.json")
    md = cfg.get("model_dir", "/opt/ml/model/encoder"); model.m.save_pretrained(md); tok.save_pretrained(md)
    if cfg.get("score_test"):
        ts1 = read_src(root / "test/test_source1.tsv"); ttg = pd.concat([read_src(root / f"test/test_source{i}.tsv") for i in (2, 3)])
        t12 = set()
        for sp in sorted(glob.glob(str(root / "test12/*-scores.tsv.gz"))):
            d = pd.read_csv(sp, sep="\t", quoting=3, dtype=str, usecols=["source1_entity_id", "target_id"]); t12.update(zip(d["source1_entity_id"], d["target_id"]))
        for country in [c for c in sorted(ts1["country"].unique()) if not cfg.get("test_countries") or c in cfg["test_countries"]]:
            qd = ts1[ts1["country"] == country]; td = ttg[ttg["country"] == country]
            qs = qd["entity_id"].tolist(); ts = td["entity_id"].tolist()
            E_t = embed(model, tok_all(tok, ("passage: " + td["entity_id"].str[:2].str.lower() + ": " + td["business_name"] + " | " + td["business_address"]).tolist(), cfg["max_len"]), pad, dev, cfg["infer_batch"])
            E_q = embed(model, tok_all(tok, ("query: " + qd["business_name"] + " | " + qd["business_address"]).tolist(), cfg["max_len"]), pad, dev, cfg["infer_batch"])
            for s in range(0, len(qs), 200000):
                v, ix = topk(E_q[s:s + 200000], E_t, cfg["test_k"])
                rows = [(qs[s + i], ts[j], float(v[i][r]), r + 1) for i in range(len(ix)) for r, j in enumerate(ix[i]) if (qs[s + i], ts[j]) not in t12]
                put(f"test-{country}-{s // 200000:03d}-dense.parquet", pd.DataFrame(rows, columns=["q", "t", "cos", "rank"]))
            del E_t, E_q; torch.cuda.empty_cache()
    log(event="complete")


if __name__ == "__main__":
    main()
