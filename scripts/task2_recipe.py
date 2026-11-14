"""Task 2 analysis: learning curves and paired differences for the recipe ladder.
Usage: python task2_recipe.py SWEEP_DIR OUT.png OUT.csv
SWEEP_DIR holds one folder per run, named <row>_s<seed>, each with metrics.json and eval_log.jsonl.
Headline numbers use metrics.json (final eval over 1M validation tokens); curves use eval_log.jsonl (131k-token evals).
"""
import csv, glob, json, os, re, sys
from collections import defaultdict
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ORDER = ["gpt2_textbook", "plus_rmsnorm_rope", "plus_swiglu", "plus_wsd_adamw", "plus_muon"]

def read(p):
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]

def main(sweep_dir, out_png, out_csv, tokens_per_step=16384):
    final, curves = defaultdict(dict), defaultdict(dict)
    for d in sorted(glob.glob(os.path.join(sweep_dir, "*_s*"))):
        m = re.match(r"(.+)_s(\d+)$", os.path.basename(d))
        mf, ef = os.path.join(d, "metrics.json"), os.path.join(d, "eval_log.jsonl")
        if not m or not os.path.exists(mf):
            continue
        name, seed = m.group(1), int(m.group(2))
        final[name][seed] = json.load(open(mf))["val_bpb"]
        if os.path.exists(ef):
            curves[name][seed] = {r["step"]: r["bpb"] for r in read(ef)}
    names = [n for n in ORDER if n in final]
    if not names:
        sys.exit(f"No finished runs in {sweep_dir}")
    seeds = sorted({s for n in names for s in final[n]})

    print(f"{'row':20s}" + "".join(f"{'seed '+str(s):>11s}" for s in seeds) + f"{'mean':>9s}{'std':>8s}")
    rows = []
    for n in names:
        v = [final[n][s] for s in seeds if s in final[n]]
        sd = float(np.std(v, ddof=1)) if len(v) > 1 else float("nan")
        print(f"{n:20s}" + "".join(f"{final[n][s]:11.4f}" if s in final[n] else f"{'-':>11s}" for s in seeds) + f"{np.mean(v):9.4f}{sd:8.4f}")
        rows.append([n] + [final[n].get(s, "") for s in seeds] + [round(float(np.mean(v)), 4), round(sd, 4) if sd == sd else ""])

    def diff(a, b):
        ss = [s for s in seeds if s in final.get(a, {}) and s in final.get(b, {})]
        return ss, [final[b][s] - final[a][s] for s in ss]
    print("\nPaired differences (B - A; negative = B is better), per seed and mean:")
    pairs = [(names[i], names[i + 1]) for i in range(len(names) - 1)]
    named = {("plus_swiglu", "plus_wsd_adamw"): "schedule only (cosine -> WSD, AdamW kept)",
             ("plus_wsd_adamw", "plus_muon"): "optimizer only (AdamW -> Muon, WSD kept)",
             ("gpt2_textbook", "plus_muon"): "total, textbook -> Muon"}
    allp = pairs + ([("gpt2_textbook", "plus_muon")] if "gpt2_textbook" in names and "plus_muon" in names else [])
    drows = []
    for a, b in allp:
        ss, d = diff(a, b)
        if not d:
            continue
        agree = "same sign in every seed" if len(d) > 1 and (all(x < 0 for x in d) or all(x > 0 for x in d)) else ("one seed" if len(d) == 1 else "seeds disagree")
        tag = f"   [{named[(a, b)]}]" if (a, b) in named else ""
        print(f"  {a:18s} -> {b:18s} " + "  ".join(f"s{s}:{x:+.4f}" for s, x in zip(ss, d)) + f"   mean {np.mean(d):+.4f} ({agree}){tag}")
        drows.append([a, b] + [round(x, 4) for x in d] + [round(float(np.mean(d)), 4), agree])

    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    for i, n in enumerate(names):
        cs = list(curves[n].values())
        if not cs:
            continue
        steps = sorted(set.intersection(*[set(c) for c in cs]))
        xs = [s * tokens_per_step / 1e6 for s in steps]
        for c in cs:
            ax.plot(xs, [c[s] for s in steps], "-", color=f"C{i}", alpha=0.25, lw=1)
        ax.plot(xs, [np.mean([c[s] for c in cs]) for s in steps], "-o", ms=4, color=f"C{i}", label=n)
    ax.set_xlabel("tokens seen (M)"); ax.set_ylabel("validation bits per byte (lower is better)")
    ax.set_title("Task 2: recipe ladder on FineWeb (thin lines: seeds, thick: mean)")
    ax.grid(alpha=0.3); ax.legend(fontsize=8); fig.tight_layout(); fig.savefig(out_png, dpi=150)

    with open(out_csv, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["row"] + [f"seed_{s}" for s in seeds] + ["mean", "std"]); w.writerows(rows)
        w.writerow([]); w.writerow(["A", "B", "diff_per_seed...", "mean", "agreement"]); w.writerows(drows)
    print("\nwrote", out_png, "and", out_csv)

if __name__ == "__main__":
    main(*sys.argv[1:4])