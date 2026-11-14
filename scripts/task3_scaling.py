"""Task 3 analysis: validation bpb vs non-embedding parameters and vs training FLOPs, at a fixed token budget.
Usage: python task3_scaling.py OUT.png OUT.csv SWEEP_DIR [SWEEP_DIR ...]
Each SWEEP_DIR holds run folders <name>_s<seed> with metrics.json. Runs used: 'plus_swiglu' (the 256x4 point,
from the recipe sweep) and every run whose name starts with 'size_'.
FLOPs = 6 x total parameters x tokens seen (the tied output layer is compute, so embeddings are included).
"""
import csv, glob, json, os, re, sys
from collections import defaultdict
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def main(out_png, out_csv, dirs):
    pts = defaultdict(lambda: {"bpb": {}, "ne": None, "tot": None, "tok": None})
    for sd in dirs:
        for d in sorted(glob.glob(os.path.join(sd, "*_s*"))):
            m = re.match(r"(.+)_s(\d+)$", os.path.basename(d))
            mf = os.path.join(d, "metrics.json")
            if not m or not os.path.exists(mf):
                continue
            name, seed = m.group(1), int(m.group(2))
            if name != "plus_swiglu" and not name.startswith("size_"):
                continue
            j = json.load(open(mf))
            p = pts[j["params_non_embedding"]]
            p["bpb"][seed] = j["val_bpb"]
            p["ne"], p["tot"], p["tok"] = j["params_non_embedding"], j["params"], j["tokens_seen"]
    if len(pts) < 2:
        sys.exit("Need at least two model sizes with finished runs.")
    toks = {p["tok"] for p in pts.values()}
    if len(toks) > 1:
        print("WARNING: token budgets differ across runs:", sorted(toks))
    keys = sorted(pts)
    rows = []
    print(f"{'non-emb params':>15s} {'total':>10s} {'FLOPs':>10s} {'seeds':>6s} {'val bpb':>9s} {'gain/doubling of N':>20s}")
    prev = None
    for k in keys:
        p = pts[k]
        v = list(p["bpb"].values())
        mu = float(np.mean(v))
        fl = 6.0 * p["tot"] * p["tok"]
        gain = ""
        if prev is not None:
            doublings = np.log2(k / prev[0])
            gain = f"{(prev[1] - mu) / doublings:+.4f}"
        print(f"{k:15,d} {p['tot']:10,d} {fl:10.2e} {len(v):6d} {mu:9.4f} {gain:>20s}")
        rows.append([k, p["tot"], fl, len(v), round(mu, 4), gain])
        prev = (k, mu)
    fig, axs = plt.subplots(1, 2, figsize=(10, 4.2))
    xs_n = keys
    xs_f = [6.0 * pts[k]["tot"] * pts[k]["tok"] for k in keys]
    ys = [float(np.mean(list(pts[k]["bpb"].values()))) for k in keys]
    for ax, xs, xl in [(axs[0], xs_n, "non-embedding parameters"), (axs[1], xs_f, "training FLOPs (6 x params x tokens)")]:
        ax.plot(xs, ys, "-o", color="C0")
        for k, x in zip(keys, xs):
            ax.plot([x] * len(pts[k]["bpb"]), list(pts[k]["bpb"].values()), ".", color="C0", alpha=0.4)
        ax.set_xscale("log"); ax.set_xlabel(xl); ax.set_ylabel("validation bits per byte"); ax.grid(alpha=0.3, which="both")
    axs[0].set_title("bpb vs non-embedding parameters"); axs[1].set_title("bpb vs training FLOPs (same token budget)")
    fig.tight_layout(); fig.savefig(out_png, dpi=150)
    with open(out_csv, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["non_embedding_params", "total_params", "train_flops", "n_seeds", "val_bpb_mean", "gain_per_doubling_of_N"]); w.writerows(rows)
    print("\nwrote", out_png, "and", out_csv)

if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3:])