\# HW 2 - Tokenizers and pre-training on web text (Parts 1-2)



Course code: https://github.com/lkk688/DeepDataMiningLearning (llm/nanolm). Not copied here; this repo holds only my configs, scripts and results.



Machine: HP laptop (CPU), Python 3.13, PyTorch 2.14.



The report submitted on Canvas is `HW2\_report.pdf` in this repo (added when finished).



\## Task 1 - tokenizer study

To be added.



\## Task 2 - training recipe on FineWeb (L4q size: 4 layers x 256)

\- `configs/recipe\_laptop.yaml`: the course ladder (`03\_transformer\_recipe.yaml`) at L4q size, 5 rows x 2 seeds (1337, 42): `gpt2\_textbook`, `plus\_rmsnorm\_rope`, `plus\_swiglu`, `plus\_wsd\_adamw` (the control that separates the schedule from the optimizer), `plus\_muon`.

\- Built on `03\_gpt\_fineweb.yaml` (AdamW + cosine), not on the L4 configs, which default to Muon + WSD. 300 steps (20 warmup), 16,384 tokens per step, 4.9M tokens per run, no time cap, so every run sees exactly the same tokens.

\- `results/task2/`: one folder per run (config, metrics, logs, sample) plus `sweep.yaml`. `results/task2\_recipe.csv` and `figures/task2\_recipe.png` come from `scripts/task2\_recipe.py`.

\- Final validation bpb (mean of 2 seeds): textbook 2.647, +RMSNorm/RoPE 2.454, +SwiGLU 2.389, +WSD 2.283, +Muon 2.138. Every step has the same sign in both seeds. Schedule only (WSD minus SwiGLU): -0.106; optimizer only (Muon minus WSD): -0.145.



\## Task 3 - scaling at a fixed token budget

\- `configs/scaling\_laptop.yaml`: model sizes 64x2, 128x2, 192x3, 384x6 (dim x layers), same recipe as `plus\_swiglu`, same 4.9M tokens. The 256x4 point is the `plus\_swiglu` runs of Task 2.

\- `results/task3/`: run folders. `results/task3\_scaling.csv` and `figures/task3\_scaling.png` come from `scripts/task3\_scaling.py` (bpb against non-embedding parameters and against 6 x parameters x tokens).

\- Validation bpb: 64x2 3.081, 128x2 2.758, 192x3 2.536, 256x4 2.389, 384x6 2.257. Gain per doubling of non-embedding parameters falls from 0.16 to 0.08.



\## Extra credit - SwiGLU

To be added.



\## Reduced budget

The course sweeps are run at reduced size: 300 steps (4.9M tokens) per FineWeb run instead of the L4q config's 1000 steps and 25-minute time cap, which would give runs different numbers of tokens. Every comparison uses identical token budgets.

