"""Task 1 analysis: tokenizer study. Best and final validation bpb, passes over the data, learning curves.
Usage: python task1_tokenizers.py SWEEP_DIR OUT.png OUT.csv [DATA_DIR]
SWEEP_DIR holds <run>_s<seed> folders with metrics.json and eval_log.jsonl.
DATA_DIR (optional, e.g. llm\\data\\nanolm) is used to count training tokens: passes = tokens_seen / training tokens.
'best' and 'final' come from eval_log.jsonl (same validation subset at every evaluation); the official final
number over the whole validation split is metrics.json val_bpb.
"""
import csv, glob, json, os, re, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ORDER = ["char", "byte", "sp_bpe_512", "sp_bpe_2048", "sp_unigram_2048", "word_2048", "word_8192", "gpt2_bpe"]
KEY = {"char": "char", "byte": "byte", "sp_bpe_512": "sp_trainbpe512", "sp_bpe_2048": "sp_trainbpe2048",
       "sp_unigram_2048": "sp_trainunigram2048", "word_2048": "word2048", "word_8192": "word8192", "gpt2_bpe": "gpt2"}
VOCAB = {"char": 65, "byte": 256, "sp_bpe_512": 512, "sp_bpe_2048": 2048, "sp_unigram_2048": 2048,
         "word_2048": 2048, "word_8192": 8192, "gpt2_bpe": 50257}
HEADER_BYTES = 1024          # 256 x int32 header, then uint16 tokens

def read(p):
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]

def train_tokens(data_dir, name):
    if not data_dir:
        return None
    hits = glob.glob(os.path.join(data_dir, f"*{KEY[name]}_tail*", "train.bin"))
    return (os.path.getsize(hits[0]) - HEADER_BYTES) // 2 if hits else None

def main(sweep_dir, out_png, out_csv, data_dir=None):
    runs = {}
    for d in sorted(glob.glob(os.path.join(sweep_dir, "*_s*"))):
        m = re.match(r"(.+)_s(\d+)$", os.path.basename(d))
        mf, ef = os.path.join(d, "metrics.json"), os.path.join(d, "eval_log.jsonl")
        if not m or not os.path.exists(mf) or not os.path.exists(ef) or m.group(1) not in KEY:
            continue
        runs[m.group(1)] = (json.load(open(mf)), read(ef))
    if not runs:
        sys.exit(f"No finished tokenizer runs in {sweep_dir}")
    names = [n for n in ORDER if n in runs]
    chk = train_tokens(data_dir, "char")
    if data_dir and chk is not None and chk != 1003854:
        print(f"WARNING: counted {chk:,} char training tokens, expected 1,003,854; passes may be off")
    print(f"{'tokenizer':16s} {'V':>6s} {'B/tok':>6s} {'passes':>7s} {'best bpb':>9s} {'@step':>6s} {'final bpb':>10s} {'final-best':>11s} {'official final':>15s}")
    rows = []
    for n in names:
        mj, ev = runs[n]
        steps = [r["step"] for r in ev]; bp = [r["bpb"] for r in ev]
        i = int(np.argmin(bp)); best, bstep, fin = bp[i], steps[i], bp[-1]
        ntr = train_tokens(data_dir, n)
        passes = (mj["tokens_seen"] / ntr) if ntr else float("nan")
        print(f"{n:16s} {VOCAB[n]:6d} {mj.get('val_bytes_per_token', float('nan')):6.2f} {passes:7.1f} {best:9.4f} {bstep:6d} {fin:10.4f} {fin-best:+11.4f} {mj['val_bpb']:15.4f}")
        rows.append([n, VOCAB[n], round(mj.get('val_bytes_per_token', float('nan')), 3), round(passes, 2), round(best, 4), bstep, round(fin, 4), round(fin - best, 4), round(mj['val_bpb'], 4)])
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    for i, n in enumerate(names):
        ev = runs[n][1]
        ax.plot([r["step"] for r in ev], [r["bpb"] for r in ev], "-o", ms=3, color=f"C{i}", label=f"{n} (V={VOCAB[n]})")
        j = int(np.argmin([r["bpb"] for r in ev]))
        ax.plot([ev[j]["step"]], [ev[j]["bpb"]], "*", ms=11, color=f"C{i}")
    ax.set_xlabel("training step (8192 tokens per step)"); ax.set_ylabel("validation bits per byte (lower is better)")
    ax.set_title("Task 1: tokenizers on Shakespeare (stars: best checkpoint)"); ax.grid(alpha=0.3); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(out_png, dpi=150)
    with open(out_csv, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["tokenizer", "vocab", "bytes_per_token", "passes", "best_bpb", "best_step", "final_bpb", "final_minus_best", "official_final_bpb"]); w.writerows(rows)
    print("\nwrote", out_png, "and", out_csv)

if __name__ == "__main__":
    main(*sys.argv[1:5])