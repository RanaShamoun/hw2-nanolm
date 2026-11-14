"""Extra credit + '2x model or 2x tokens' analysis.
Usage: python extra_credit.py OUT.png OUT.csv RECIPE_DIR LR_DIR TOKENS_DIR DOUBLE_DIR
RECIPE_DIR: Task 2 sweep (plus_rmsnorm_rope = GELU, plus_swiglu = SwiGLU, lr 1e-3, 300 steps)
LR_DIR: gelu_lr5e-4, swiglu_lr5e-4 (300 steps)   TOKENS_DIR: gelu_600steps, swiglu_600steps (lr 1e-3)
DOUBLE_DIR: size_d256_L8 (SwiGLU, 300 steps, 2x non-embedding parameters)
All runs read metrics.json val_bpb. Differences are SwiGLU - GELU (negative = SwiGLU better), paired by seed.
"""
import csv, glob, json, os, re, sys
from collections import defaultdict
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def load(d):
    out = defaultdict(dict)
    for f in sorted(glob.glob(os.path.join(d, "*_s*"))):
        m = re.match(r"(.+)_s(\d+)$", os.path.basename(f))
        mf = os.path.join(f, "metrics.json")
        if m and os.path.exists(mf):
            j = json.load(open(mf))
            out[m.group(1)][int(m.group(2))] = (j["val_bpb"], j.get("params_non_embedding"), j.get("tokens_seen"))
    return out

def cond(label, gelu, swi, rows):
    seeds = sorted(set(gelu) & set(swi))
    if not seeds:
        print(f"  {label}: no common seeds"); return None
    d = [swi[s][0] - gelu[s][0] for s in seeds]
    print(f"  {label:22s} GELU " + " ".join(f"s{s}:{gelu[s][0]:.4f}" for s in seeds) + "   SwiGLU " + " ".join(f"s{s}:{swi[s][0]:.4f}" for s in seeds)
          + "   diff " + " ".join(f"s{s}:{x:+.4f}" for s, x in zip(seeds, d)) + f"   mean {np.mean(d):+.4f}")
    rows.append([label] + [round(x, 4) for x in d] + [round(float(np.mean(d)), 4)])
    return d

def main(out_png, out_csv, recipe, lr, tokens, double):
    R, L, T, D = load(recipe), load(lr), load(tokens), load(double)
    rows = []
    print("SwiGLU - GELU (negative = SwiGLU better):")
    d_lr1 = cond("lr 1e-3 (300 steps)", R["plus_rmsnorm_rope"], R["plus_swiglu"], rows)
    d_lr2 = cond("lr 5e-4 (300 steps)", L["gelu_lr5e-4"], L["swiglu_lr5e-4"], rows)
    d_t1 = d_lr1
    d_t2 = cond("600 steps (9.8M tokens)", T["gelu_600steps"], T["swiglu_600steps"], rows)
    print("\n2x model vs 2x tokens (256x4 SwiGLU, 300 steps is the reference):")
    base = R["plus_swiglu"]
    tok2 = T["swiglu_600steps"]
    mod2 = D["size_d256_L8"]
    for s in sorted(base):
        b = base[s][0]
        line = f"  seed {s}: reference {b:.4f}"
        if s in tok2: line += f" | 2x tokens (600 steps) {tok2[s][0]:.4f} ({tok2[s][0]-b:+.4f})"
        if s in mod2: line += f" | 2x model (256x8) {mod2[s][0]:.4f} ({mod2[s][0]-b:+.4f})  [{mod2[s][1]:,} vs {base[s][1]:,} non-embedding params]"
        print(line)
        rows.append([f"2x comparison seed {s}", round(b, 4), round(tok2[s][0], 4) if s in tok2 else "", round(mod2[s][0], 4) if s in mod2 else ""])
    fig, axs = plt.subplots(1, 3, figsize=(12, 3.8))
    for ax, title, xs, ds in [(axs[0], "SwiGLU - GELU vs learning rate", ["1e-3", "5e-4"], [d_lr1, d_lr2]),
                              (axs[1], "SwiGLU - GELU vs token budget", ["4.9M", "9.8M"], [d_t1, d_t2])]:
        for i, d in enumerate(ds):
            if d is None: continue
            ax.plot([i] * len(d), d, "o", color="C0", alpha=0.6)
            ax.plot([i], [np.mean(d)], "_", color="C3", ms=22, mew=2)
        ax.axhline(0, color="gray", lw=0.8); ax.set_xticks(range(len(xs))); ax.set_xticklabels(xs); ax.set_title(title, fontsize=10)
        ax.set_ylabel("val bpb difference (negative: SwiGLU better)"); ax.grid(alpha=0.3)
    labels, vals = ["reference\n256x4, 300 steps", "2x tokens\n256x4, 600 steps", "2x model\n256x8, 300 steps"], []
    for src in (base, tok2, mod2):
        vals.append([v[0] for v in src.values()])
    axs[2].bar(range(3), [np.mean(v) if v else 0 for v in vals], color=["C7", "C0", "C1"])
    for i, v in enumerate(vals):
        axs[2].plot([i] * len(v), v, "k.", alpha=0.6)
    lo = min(min(v) for v in vals if v); hi = max(max(v) for v in vals if v)
    axs[2].set_ylim(lo - 0.05, hi + 0.03); axs[2].set_xticks(range(3)); axs[2].set_xticklabels(labels, fontsize=8)
    axs[2].set_title("equal-compute comparison", fontsize=10); axs[2].set_ylabel("val bpb")
    fig.tight_layout(); fig.savefig(out_png, dpi=150)
    with open(out_csv, "w", newline="") as fh:
        csv.writer(fh).writerows(rows)
    print("\nwrote", out_png, "and", out_csv)

if __name__ == "__main__":
    main(*sys.argv[1:7])