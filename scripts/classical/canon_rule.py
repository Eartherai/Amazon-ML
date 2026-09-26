"""CL-075: country-invariant canonical equality rules for the uncertain band (France diagnosis).

Canonical name: NFKD accent strip, casefold, punctuation to space, drop legal-form tokens (FR/US/IN lists), sorted token set.
Canonical address: NFKD, casefold, punctuation to space, street-type synonyms mapped, leading zeros stripped from numbers,
region/department/state tokens not used (token set of numbers + street words + city), sorted token set.
Measures, on India/US train fold tables (labels), the precision/coverage of rules by probability band and decision, and on
test the number of France/IU pairs each rule would flip. Label-free on test.
"""
import re, sys, unicodedata
from pathlib import Path
import polars as pl

ROOT = Path(__file__).resolve().parents[2]
LEGAL = set("sarl sas sasu eurl sa sci snc scop scs sca selarl sem gie cie compagnie fils societe ste llc inc incorporated ltd limited pvt private corp corporation co company plc llp lp pllc pc".split())
STREET = {"r": "rue", "rue": "rue", "av": "avenue", "ave": "avenue", "avenue": "avenue", "bd": "boulevard", "blvd": "boulevard", "boulevard": "boulevard",
          "all": "allee", "allee": "allee", "pl": "place", "place": "place", "imp": "impasse", "impasse": "impasse", "ch": "chemin", "chemin": "chemin",
          "rte": "route", "route": "route", "qu": "quai", "quai": "quai", "crs": "cours", "cours": "cours", "sq": "square", "square": "square",
          "st": "street", "street": "street", "rd": "road", "road": "road", "dr": "drive", "drive": "drive", "ln": "lane", "lane": "lane",
          "ct": "court", "court": "court", "hwy": "highway", "highway": "highway", "pkwy": "parkway", "parkway": "parkway", "nagar": "nagar", "marg": "marg",
          "n": "", "no": "", "bis": "bis", "ter": "ter", "and": "et", "et": "et", "&": "et"}


def base(s):
    s = unicodedata.normalize("NFKD", s or "")
    s = "".join(ch for ch in s if not unicodedata.combining(ch)).casefold()
    return re.sub(r"[^\w]+", " ", s).strip()


def cname(s):
    toks = [t for t in base(s).split() if t not in LEGAL]
    return " ".join(sorted(set(toks)))


def caddr(s, regions):
    out = []
    for t in base(s).split():
        if t.isdigit(): t = t.lstrip("0") or "0"
        t = STREET.get(t, t)
        if t and t not in regions: out.append(t)
    return " ".join(sorted(set(out)))


def main():
    T = ROOT / "student_resource/dataset"
    regions = set()
    for s in ("hauts de france pays de la loire nouvelle aquitaine nord gironde loire atlantique pas de calais ile de france auvergne rhone alpes provence alpes cote d azur "
              "occitanie grand est normandie bretagne bourgogne franche comte centre val de loire").split():
        pass
    regions = set("hauts de france pays la loire nouvelle aquitaine nord gironde atlantique pas calais".split())  # FR regions/departments seen in data
    txt = {}
    for split in ("train", "test"):
        for i in (1, 2, 3):
            d = pl.read_csv(T / f"{split}/{split}_source{i}.tsv", separator="\t", quote_char=None, infer_schema_length=0).fill_null("")
            txt.update(dict(zip(d["entity_id"].to_list(), zip(d["business_name"].to_list(), d["business_address"].to_list()))))
    def feats(df):
        qn, qa, tn, ta = [], [], [], []
        for q, t in df.select("q", "t").iter_rows():
            a, b = txt[q], txt[t]; qn.append(cname(a[0])); tn.append(cname(b[0])); qa.append(caddr(a[1], regions)); ta.append(caddr(b[1], regions))
        return df.with_columns(pl.Series("qn", qn), pl.Series("tn", tn), pl.Series("qa", qa), pl.Series("ta", ta)).with_columns(
            (pl.col("qn") == pl.col("tn")).alias("name_eq"), ((pl.col("qa") == pl.col("ta")) & (pl.col("ta") != "")).alias("addr_eq"), (pl.col("ta") == "").alias("t_noaddr"))
    tr = pl.concat([pl.read_parquet(f"/Users/earther/Desktop/aml-shared/candidates_mfu_f{h}.parquet", columns=["q", "t", "prob", "decision", "label", "country"]) for h in (1, 2, 3)])
    tr = feats(tr)
    band = lambda p: pl.when(p >= 0.72).then(pl.lit("hi")).when(p >= 0.2).then(pl.lit("band")).when(p >= 0.02).then(pl.lit("low")).otherwise(pl.lit("vlow"))
    tr = tr.with_columns(band(pl.col("prob")).alias("b"))
    print("TRAIN rule precision by band (rule: name_eq & addr_eq; name_eq & t_noaddr; addr_eq only; name_eq only)")
    print(tr.group_by("b").agg(pl.len(), pl.col("label").mean().alias("prec_all"),
                                (pl.col("name_eq") & pl.col("addr_eq")).sum().alias("n_NA"), pl.col("label").filter(pl.col("name_eq") & pl.col("addr_eq")).mean().alias("p_NA"),
                                (pl.col("name_eq") & pl.col("t_noaddr")).sum().alias("n_N0"), pl.col("label").filter(pl.col("name_eq") & pl.col("t_noaddr")).mean().alias("p_N0"),
                                (pl.col("addr_eq") & ~pl.col("name_eq")).sum().alias("n_A"), pl.col("label").filter(pl.col("addr_eq") & ~pl.col("name_eq")).mean().alias("p_A"),
                                (pl.col("name_eq") & ~pl.col("addr_eq") & ~pl.col("t_noaddr")).sum().alias("n_N"), pl.col("label").filter(pl.col("name_eq") & ~pl.col("addr_eq") & ~pl.col("t_noaddr")).mean().alias("p_N")).sort("b"))
    te = pl.read_parquet(ROOT / "outputs/experiments/CL-066/test_cands_frblend050_postuniv.parquet", columns=["q", "t", "prob", "decision", "country"])
    te = feats(te).with_columns(band(pl.col("prob")).alias("b"))
    print("TEST counts by country and band")
    print(te.group_by("country", "b").agg(pl.len(), pl.col("decision").mean().alias("acc"), (pl.col("name_eq") & pl.col("addr_eq")).sum().alias("n_NA"),
                                          (pl.col("name_eq") & pl.col("t_noaddr")).sum().alias("n_N0"), (pl.col("addr_eq") & ~pl.col("name_eq")).sum().alias("n_A"),
                                          (pl.col("name_eq") & ~pl.col("addr_eq") & ~pl.col("t_noaddr")).sum().alias("n_N")).sort("country", "b"))
    tr.select("q", "t", "name_eq", "addr_eq", "t_noaddr").write_parquet(ROOT / "outputs/experiments/CL-075/train_canon.parquet")
    te.select("q", "t", "country", "prob", "decision", "name_eq", "addr_eq", "t_noaddr").write_parquet(ROOT / "outputs/experiments/CL-075/test_canon.parquet")


if __name__ == "__main__":
    main()
