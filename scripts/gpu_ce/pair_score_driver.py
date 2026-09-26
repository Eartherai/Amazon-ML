"""CL-029: score arbitrary (q, t) pair sets with a trained CE checkpoint or a zero-shot HF reranker (SageMaker, 1 GPU).

config.json keys:
  model: {"kind": "ce_tar"} (init channel *.tar.gz with encoder/ + head.pt, EXP-050 text format)
      or {"kind": "hf_seqcls", "model_id": ..., "revision": ...} (zero-shot reranker; plain 'name, address' text)
  pair_sets: list of {"name", "split": train|test, "source": "top12" (fold, band) | channel name (glob, max_rank; *.tsv.gz
             test12 shards filtered at band)}, optional "accent_fold": true applies NFKC + accent stripping to both texts
  max_len, infer_batch, s3_out
Channels: code, train (raw train TSVs), test (raw test TSVs), top12, init, and any pair channels named in pair_sets.
Output parquet parts q, t, logit (streamed to s3_out). Fold4 is absent from every input.
"""
import glob, json, time, tarfile, unicodedata
from pathlib import Path
import numpy as np, pandas as pd, torch

T0 = time.time()
ROOT = Path("/opt/ml/input/data")


def log(**kw):
    print(json.dumps({"t": round(time.time() - T0, 1), **kw}), flush=True)


def fold(x):
    return "".join(c for c in unicodedata.normalize("NFKD", unicodedata.normalize("NFKC", x)) if not unicodedata.combining(c))


def texts(split, fmt, folded=False):
    out = {}
    for i in (1, 2, 3):
        df = pd.read_csv(ROOT / f"{split}/{split}_source{i}.tsv", sep="\t", quoting=3, dtype=str, keep_default_na=False,
                         na_filter=False, usecols=["entity_id", "business_name", "business_address"])
        sep = " | " if fmt == "ce" else ", "
        vals = df["business_name"] + sep + df["business_address"]
        out.update(zip(df["entity_id"], map(fold, vals) if folded else vals))
    return out


def main():
    cfg = json.loads((ROOT / "code/config.json").read_text()); log(event="config", **cfg)
    dev = torch.device("cuda"); from transformers import AutoTokenizer
    kind = cfg["model"]["kind"]
    if kind == "ce_tar":
        with tarfile.open(sorted(glob.glob(str(ROOT / "init/*.tar.gz")))[0]) as tf: tf.extractall("/tmp/init")
        from transformers import AutoModel
        tok = AutoTokenizer.from_pretrained("/tmp/init/encoder", use_fast=True)
        enc = AutoModel.from_pretrained("/tmp/init/encoder").to(dev).eval()
        head = torch.nn.Linear(enc.config.hidden_size, 1); head.load_state_dict(torch.load("/tmp/init/head.pt", map_location="cpu")); head = head.to(dev)
        fwd = lambda x, m: head(enc(input_ids=x, attention_mask=m).last_hidden_state[:, 0].float()).squeeze(-1)
    else:
        from transformers import AutoModelForSequenceClassification
        from huggingface_hub import HfApi
        mid, rev = cfg["model"]["model_id"], cfg["model"].get("revision")
        log(event="hf_model", model_id=mid, resolved_sha=HfApi().model_info(mid, revision=rev).sha)
        tok = AutoTokenizer.from_pretrained(mid, revision=rev, use_fast=True)
        net = AutoModelForSequenceClassification.from_pretrained(mid, revision=rev, torch_dtype=torch.float16).to(dev).eval()
        fwd = lambda x, m: net(input_ids=x, attention_mask=m).logits[:, 0].float()
    pad = tok.pad_token_id; s3 = __import__("boto3").client("s3"); cache = {}
    for spec in cfg["pair_sets"]:
        split = spec["split"]
        ck = (split, bool(spec.get("accent_fold")))
        if ck not in cache: cache[ck] = texts(split, "ce" if kind == "ce_tar" else "plain", folded=ck[1])
        tx = cache[ck]
        if spec["source"] == "top12":
            top = pd.concat([pd.read_parquet(p) for p in sorted(glob.glob(str(ROOT / "top12/top12-fold*.parquet")))])
            top = top[(top["fold"] == spec["fold"]) & (top["base_score"] >= spec["band"])]
            frames = [("top12", top.rename(columns={"source1_entity_id": "q", "target_id": "t"})[["q", "t"]])]
        else:
            frames = []
            for fp in sorted(glob.glob(str(ROOT / spec["source"] / spec["glob"]))):
                if fp.endswith(".tsv.gz"):  # test12 shard: source1_entity_id, target_id, score (stage-1)
                    df = pd.read_csv(fp, sep="\t", quoting=3, dtype={"source1_entity_id": str, "target_id": str, "score": float})
                    df = df[df["score"] >= spec.get("band", 0.0)].rename(columns={"source1_entity_id": "q", "target_id": "t"})
                else:
                    df = pd.read_parquet(fp)
                if spec.get("max_rank"): df = df[df["rank"] <= spec["max_rank"]]
                frames.append((Path(fp).stem, df[["q", "t"]]))
        for stem, df in frames:
            for c0 in range(0, len(df), 500000):
                part = df.iloc[c0:c0 + 500000]
                if kind == "ce_tar":
                    qs = [tx[q] for q in part["q"]]; ts = [("s2: " if t.startswith("S2-") else "s3: ") + tx[t] for t in part["t"]]
                else:
                    qs = [tx[q] for q in part["q"]]; ts = [tx[t] for t in part["t"]]
                ids = tok(qs, ts, truncation="longest_first", max_length=cfg["max_len"], padding=False)["input_ids"]
                order = np.argsort([len(x) for x in ids], kind="stable"); out = np.zeros(len(ids), np.float32)
                with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16 if kind != "ce_tar" else torch.bfloat16):
                    for i in range(0, len(order), cfg["infer_batch"]):
                        idx = order[i:i + cfg["infer_batch"]]; L = max(len(ids[j]) for j in idx)
                        x = np.full((len(idx), L), pad, np.int64); m = np.zeros((len(idx), L), np.int64)
                        for r, j in enumerate(idx): x[r, :len(ids[j])] = ids[j]; m[r, :len(ids[j])] = 1
                        out[idx] = fwd(torch.from_numpy(x).to(dev), torch.from_numpy(m).to(dev)).float().cpu().numpy()
                name = f"{spec['name']}-{stem}-{c0 // 500000:03d}.parquet"; p = Path("/tmp") / name
                pd.DataFrame({"q": part["q"].values, "t": part["t"].values, "logit": out}).to_parquet(p, index=False)
                b, k = cfg["s3_out"].replace("s3://", "").split("/", 1); s3.upload_file(str(p), b, k.rstrip("/") + "/" + name)
                log(event="part", name=name, rows=len(part))
    log(event="complete")


if __name__ == "__main__":
    main()
