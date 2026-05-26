# experiment_code — Distillation + Growing for NLI

Knowledge distillation from **`FacebookAI/roberta-large-mnli`** (teacher) into
**`FacebookAI/roberta-base`** (student) on the **`SetFit/mnli`** dataset, with the
student grown during training using the **TINY** algorithm
([paper](https://openreview.net/pdf?id=hbtG6s6e7r)).

Growing primitives come from a sibling project **gromo**
(local path `../gromo`, installed editable, never modified from this repo).

## Layout

```
experiment_code/
├── configs/           data, teacher, student, distillation + growing schedule
├── src/distill_nli/
│   ├── data/          SetFit/mnli loading, tokenization, dataloaders
│   ├── models/        teacher, student, gromo-wrap bridge
│   ├── distillation/  KL/CE losses, training loop
│   ├── growing/       TINY growing step
│   └── utils/         seed, logging, metrics
├── scripts/           eval_teacher, train_baseline, train_distill, train_distill_grow
├── tests/             unit tests
├── notebooks/         exploratory
└── experiments/       run artifacts (gitignored)
```

## Setup

Requires `uv` and a sibling clone of `gromo` at `../gromo` (this repo installs it
as an editable dep).

```bash
uv sync
```

This pins Python 3.11 and installs torch, transformers, datasets, accelerate,
evaluate, gromo (editable), and dev tooling.

## Running on a MacBook Air (MPS)

Apple-Silicon MPS is supported end-to-end. gromo's default device is set via
`gromo.utils.utils.set_device("mps")` at process start (called by the training
entrypoints). Configs default to bf16 and small batch sizes; bump them on a
bigger machine.

## Scripts

```bash
uv run python scripts/eval_teacher.py         # sanity-check teacher on MNLI val
uv run python scripts/train_baseline.py       # student only — control
uv run python scripts/train_distill.py        # distill, no grow — ablation
uv run python scripts/train_distill_grow.py   # distill + grow — headline
```

## License / acknowledgement

TINY growing implementation follows
<https://gitlab.inria.fr/mverbock/tinypub> (Verbockhaven et al.).
