"""Task 1 overfitting check: training vs validation bits-per-byte for long tokenizer runs.
Usage: python task1_long.py SWEEP_DIR OUT.png OUT.csv [tokens_per_step]
SWEEP_DIR holds <run>_s<seed> folders with metrics.json, eval_log.jsonl, train_log.jsonl.
Training loss is converted to bpb with the validation bytes-per-token of the same run (an approximation:
the training text may have a slightly different bytes/token, and training loss includes dropout noise).
"""
import csv, glob, json, math, os, re, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def read(p):
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]

def main(sweep_dir, out_png, out_csv, tokens_per_step=8192):
    runs = {}
    for d in sorted(glob.glob(os.path.join(sweep_dir, "*_s*"))):
        m = re.match(r"(.+)_s(\d+)$", os.path.basename(d))
        need = [os.path.join(d, f) for f in ("metrics.json", "eval_log.jsonl", "train_log.jsonl")]
        if not m or not all(os.path.exists(p) for p in need):
            continue
        mj = json.load(open(need[0]))
        bpt = mj.get("val_bytes_per_token") or 1.0
        ev = {r["step"]: r["bpb"] for r in read(need[1])}
        tr = {r["step"]: r["loss"] / math.log(2) / bpt for r in read(need[2])}
        runs[m.group(1)] = (ev, tr, mj)
    if not runs:
        sys.exit(f"No finished runs in {sweep_dir}")
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    rows = []
    for i, (n, (ev, tr, mj)) in enumerate(sorted(runs.items())):
        xs = sorted(tr); ax.plot([x * tokens_per_step / 1e6 for x in xs], [tr[x] for x in xs], "--", color=f"C{i}", alpha=0.6, label=f"{n} train")
        xs = sorted(ev); ax.plot([x * tokens_per_step / 1e6 for x in xs], [ev[x] for x in xs], "-o", ms=3, color=f"C{i}", label=f"{n} validation")
        best = min(ev, key=ev.get); last = max(ev)
        print(f"\n{n}: step, validation bpb, training bpb (nearest logged step), gap (val - train)")
        for s in sorted(ev):
            near = min(tr, key=lambda t: abs(t - s)) if tr else None
            t = tr.get(near) if near is not None else None
            print(f"  {s:5d}  {ev[s]:7.4f}  " + (f"{t:7.4f}  {ev[s]-t:+7.4f}" if t is not None else "   (no train point)"))
            rows.append([n, s, round(ev[s], 4), round(t, 4) if t is not None else "", round(ev[s] - t, 4) if t is not None else ""])
        print(f"  best validation {ev[best]:.4f} at step {best} ({best*tokens_per_step/1e6:.1f}M tokens); final {ev[last]:.4f} at step {last}; final - best = {ev[last]-ev[best]:+.4f}")
    ax.set_xlabel("tokens seen (M)"); ax.set_ylabel("bits per byte (lower is better)")
    ax.set_title("Task 1: training vs validation bpb, long tokenizer runs"); ax.grid(alpha=0.3); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(out_png, dpi=150)
    with open(out_csv, "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["run", "step", "val_bpb", "train_bpb", "gap"]); w.writerows(rows)
    print("\nwrote", out_png, "and", out_csv)

if __name__ == "__main__":
    main(*sys.argv[1:4])