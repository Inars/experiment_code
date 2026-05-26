# experiment_code — Distillation + Growing for NLI

Knowledge distillation from **`FacebookAI/roberta-large-mnli`** (teacher) into
**`FacebookAI/roberta-base`** (student) on the **`SetFit/mnli`** dataset, with the
student grown during training using the **TINY** algorithm
([paper](https://openreview.net/pdf?id=hbtG6s6e7r)).

Two kinds of growth, both toggleable per layer in `configs/grow.yaml`:

- **FFN growth** via `gromo` — adds neurons to each transformer block's
  intermediate dim (Linear E→4E → activation → Linear 4E→E).
- **Attention growth** — adds rank to each head's Q/K projection (`k_dim`),
  ported from
  [stephane-rivaud/growing-attention](https://github.com/stephane-rivaud/growing-attention)'s
  `TwoShotsIterative` (matrix-free Conjugate Gradient, scales to RoBERTa-base).

`gromo` is consumed as a sibling editable install at `../gromo` and is never
modified from this repo.

## 2×2 experiment design

|                | no grow                | grow                          |
|----------------|------------------------|-------------------------------|
| **no distill** | `train_baseline.py`    | `train_grow.py`               |
| **distill**    | `train_distill.py`     | `train_distill_grow.py`       |

## Setup

Requires `uv` and a sibling clone of `gromo` at `../gromo` (this repo installs
it as an editable dep).

```bash
uv sync
```

Pins Python 3.11; installs torch, transformers, datasets, accelerate, evaluate,
gromo (editable), and dev tooling.

## Running on Apple Silicon (MPS)

Apple-Silicon MPS is supported end-to-end. `gromo`'s default device is set via
`gromo.utils.utils.set_device("mps")` at process start (handled by every
training entrypoint).

Two MPS limitations the project routes around (already wired in the scripts):

1. **`torch.linalg.eigh` is not implemented on MPS** in torch 2.12. `gromo`'s
   FFN growth math calls it. Every script that may grow sets
   `PYTORCH_ENABLE_MPS_FALLBACK=1` at module load so only the unsupported op
   falls back to CPU — forward/backward stay on MPS. The same env var is set in
   `tests/conftest.py`.
2. **MPS does not support fp64.** The attention grow step needs fp64 for
   numerical stability of the CG solves, so the entire grow-step math runs on
   CPU+fp64 (orchestrated from MPS-side forward/backward hooks). The training
   loop is unaffected.

Wall-clock expectations on a MacBook Air with MPS (full MNLI, batch=16, fp32):

- **Teacher eval (one pass over MNLI val):** ~5 min (verified — accuracy 0.906)
- **Training, 3 epochs:** ~15–16 hours. Reasonable as an overnight run.
- **One attention grow step (all 12 layers wrapped):** ~10–20 min, dominated by
  the per-head CG solves. Override `attention.layers` in `configs/grow.yaml` to
  wrap a subset if you want a faster cycle.

## Layout

```
experiment_code/
├── configs/
│   ├── data.yaml           dataset/splits/tokenization/batch sizes
│   ├── teacher.yaml        FacebookAI/roberta-large-mnli + label-order remap
│   ├── student.yaml        FacebookAI/roberta-base + canonical 3-way head
│   ├── train.yaml          shared optimizer/scheduler/eval/logging
│   ├── distill.yaml        soft-target loss (temperature, alpha)
│   └── grow.yaml           growth schedule + per-kind (ffn/attention) knobs
├── src/distill_nli/
│   ├── data/mnli.py        SetFit/mnli loader, tokenization, collator
│   ├── models/
│   │   ├── teacher.py      frozen wrapper + canonical-order logit permutation
│   │   ├── student.py      RoBERTa-base + fresh 3-way head in canonical order
│   │   └── growing.py      surgery — swaps HF FFN/attention with growable variants
│   ├── distillation/losses.py    Hinton-style soft-target loss
│   ├── growing/
│   │   ├── schedule.py     epoch-based warmup/interval/max_grows scheduler
│   │   ├── ffn.py          GrowableRobertaFFN (gromo-backed) + grow_step_ffn
│   │   ├── attention.py    GrowableRobertaSelfAttention + TwoShotsIterative port
│   │   ├── _math.py        vec, unvec, batched_cg
│   │   └── __init__.py     combined grow_step(...) entry point
│   ├── training/loop.py    shared training loop with optional growth hook
│   └── utils/              seed, rich logger, accuracy/F1, yaml loader
├── scripts/
│   ├── eval_teacher.py
│   ├── train_baseline.py
│   ├── train_distill.py
│   ├── train_grow.py
│   └── train_distill_grow.py
├── tests/                  unit + integration (13 tests; pytest)
└── experiments/            run artifacts (gitignored)
```

## Scripts

All scripts accept `--train-limit N` / `--val-limit N` / `--epochs E` /
`--eval-every N` / `--run-name NAME` / `--no-save` for quick iteration.

```bash
# Teacher sanity check on MNLI val (~5 min).
uv run python scripts/eval_teacher.py

# Student only, hard-label CE.
uv run python scripts/train_baseline.py

# Distillation, no growing.
uv run python scripts/train_distill.py

# Growing, no distillation.
uv run python scripts/train_grow.py

# Headline: distillation + growing.
uv run python scripts/train_distill_grow.py
```

Quick smoke test (a few hundred examples, 1–2 epochs):

```bash
uv run python scripts/train_distill_grow.py \
  --train-limit 256 --val-limit 128 --epochs 2 --eval-every 16 \
  --run-name smoke --no-save
```

## Tests

```bash
uv run pytest
```

13 tests on MPS, ~30 s total:

- Schedule (warmup / interval / max_grows / counters)
- FFN: wrap preserves forward output through all 12 layers; one grow step
  increases the intermediate dim.
- Attention: same forward-equivalence test; **CG Step 1 matches an explicit
  dense Ω solve to 1e-6 at fp64** (the strongest math correctness check);
  one full grow step on the real RoBERTa-base student.
- Combined: one `grow_step` dispatches to both growers correctly.

## Config knobs that matter

In `configs/grow.yaml`:

- `enabled`, `warmup_epochs`, `interval_epochs`, `max_grows`,
  `num_probe_batches` — when and how often growth fires.
- `ffn.layers` / `attention.layers` — `"all"` or a list of encoder-layer
  indices. Controls memory footprint of the grow step.
- `ffn.neurons_per_grow` / `attention.p_per_grow` — how much each layer/head
  can grow per step.
- `attention.top_k` — number of heads selected (across all wrapped layers) per
  grow step, ranked by the TwoShots score.
- `attention.precision` — `float64` (default, recommended) or `float32`.
- `optimizer.reset_after_grow` — rebuild the optimizer + scheduler after a
  successful grow step (default `true`; loses Adam momentum).

In `configs/distill.yaml`:

- `loss.temperature`, `loss.alpha` — Hinton-style temperature and the
  soft-target mixing weight (`hard_loss_weight = 1 - alpha`).

## License / acknowledgement

- TINY algorithm: Verbockhaven et al.,
  *Growing Tiny Networks: Spotting Expressivity Bottlenecks and Fixing Them
  Optimally*, <https://openreview.net/pdf?id=hbtG6s6e7r>.
  Reference implementation: <https://gitlab.inria.fr/mverbock/tinypub>.
- Attention-growth math (TwoShotsIterative — matrix-free CG variant of TINY
  on attention) ported from
  <https://github.com/stephane-rivaud/growing-attention>.
- `gromo` is consumed as an editable local install and is unchanged from
  upstream <https://github.com/growingnet/gromo>.
