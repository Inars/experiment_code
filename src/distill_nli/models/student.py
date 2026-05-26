"""Student: FacebookAI/roberta-base + a fresh 3-way NLI classification head.

The head is initialized in the project's canonical label order
(0=entailment, 1=neutral, 2=contradiction), matching SetFit/mnli and the
permuted teacher output from models.teacher.FrozenTeacher.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
)
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from distill_nli.data.mnli import CANONICAL_LABEL_ORDER


def load_student(
    cfg: dict[str, Any],
    device: torch.device | str,
) -> tuple[nn.Module, PreTrainedTokenizerBase]:
    name = cfg["model_name"]
    num_labels = int(cfg.get("num_labels", 3))
    assert num_labels == len(CANONICAL_LABEL_ORDER), (
        f"num_labels={num_labels} but canonical order has {len(CANONICAL_LABEL_ORDER)} classes"
    )

    id2label = dict(enumerate(CANONICAL_LABEL_ORDER))
    label2id = {v: k for k, v in id2label.items()}

    base_cfg = AutoConfig.from_pretrained(
        name,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id,
    )
    base_cfg.classifier_dropout = float(cfg.get("classifier_dropout", 0.1))

    model = AutoModelForSequenceClassification.from_pretrained(name, config=base_cfg)
    tokenizer = AutoTokenizer.from_pretrained(name)

    model.to(device)
    return model, tokenizer
