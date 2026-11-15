\# HW 2 - Tokenizers and pre-training on web text (Parts 1-2)



Course code: https://github.com/lkk688/DeepDataMiningLearning (llm/nanolm). Not copied here; this repo holds only my configs, scripts and results.



Machine: HP laptop (CPU), Python 3.13, PyTorch 2.14.



The final report submitted on Canvas is `HW2_report.pdf`.



\## Task 1 - tokenizer study

\- `configs/tokenizers_laptop.yaml`: the course sweep `laptop_tokenizers.yaml` with 6 tokenizers (char, BPE-512, BPE-2048, unigram-2048, word-2048, word-8192; byte and GPT-2 BPE left out), 600 steps (30 warmup), 8192 tokens per step (4.9M tokens), seed 1337, no time cap. Results: `results/task1/`, `results/task1_tokenizers.csv`, `figures/task1_tokenizers.png` (from `scripts/task1_tokenizers.py`).

\- Final validation bpb, equal to the best checkpoint (step 600) for all six: unigram-2048 2.0872, BPE-2048 2.0901, BPE-512 2.1060, word-2048 2.1534, word-8192 2.1699, char 2.2681. Passes over the training text are in the CSV. No tokenizer overfits at this budget.

\- Overfitting run: `configs/tokenizer_overfit_laptop.yaml`, word-8192 for the full 2000 steps. Best validation bpb 2.1001 at step 800, final 2.3001 (+0.200), while training bpb keeps falling. Results: `results/task1_long/`, `results/task1_long.csv`, `figures/task1_long.png` (from `scripts/task1_long.py`).

\- Fix experiment: `configs/tokenizer_fix_laptop.yaml` repeats the
  word-8192 run with dropout increased from 0.1 to 0.3.

- Results:
  - Original run: best validation bpb 2.1001 (step 800),
    final validation bpb 2.3001.
  - Dropout 0.3: best validation bpb 2.0593 (step 1800),
    final validation bpb 2.0693.

- The stronger regularization removed most of the overfitting and
  improved validation performance by 0.231 bpb relative to the
  original final checkpoint.

- Outputs:
  `results/task1_fix.csv`,
  `figures/task1_fix.png`.
  


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

### Learning-rate test

`swiglu_lr_laptop.yaml`

Compare GELU and SwiGLU at learning rate 5e-4
(300 steps, 2 seeds).

Results:

| Condition | Mean validation bpb |
|------------|------------|
| GELU, lr=5e-4 | 2.6116 |
| SwiGLU, lr=5e-4 | 2.5768 |

SwiGLU remained better by 0.035 bpb.

### Token-budget test

`swiglu_tokens_laptop.yaml`

Compare GELU and SwiGLU at 600 steps (9.8M tokens).

Results:

| Condition | Mean validation bpb |
|------------|------------|
| GELU, 600 steps | 2.2244 |
| SwiGLU, 600 steps | 2.1320 |

SwiGLU remained better by 0.092 bpb and the advantage increased relative
to the 300-step runs.

### 2× model versus 2× tokens

`double_model_laptop.yaml`

Reference:

- 256×4, 300 steps: 2.3890 bpb

2× tokens:

- 256×4, 600 steps: 2.1320 bpb

2× model:

- 256×8, 300 steps: 2.4328 bpb

At this budget, additional tokens were substantially more valuable than
doubling model depth.

Outputs:

- `figures/extra_credit.png`
- `results/extra_credit.csv`



\## Reduced budget

The course sweeps are run at reduced size: 300 steps (4.9M tokens) per FineWeb run instead of the L4q config's 1000 steps and 25-minute time cap, which would give runs different numbers of tokens. Every comparison uses identical token budgets.

