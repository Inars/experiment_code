"""Surgery: in-place replacement of HF RoBERTa modules with growable variants.

This is the ONLY place in the project that touches HF's RobertaLayer internals.

FFN surgery replaces the path
    layer.intermediate(dense -> activation) -> layer.output.dense
with a single GrowableRobertaFFN, while keeping layer.output.dropout,
layer.output.LayerNorm, and the residual connection intact. It does so by
monkey-patching layer.feed_forward_chunk.

Attention surgery comes in Phase 3.

gromo (../gromo) is read-only.
"""

from __future__ import annotations

import types
from typing import Any

import torch
from torch import nn

from distill_nli.growing.attention import GrowableRobertaSelfAttention
from distill_nli.growing.ffn import GrowableRobertaFFN


def _resolve_layer_indices(spec: Any, n_layers: int) -> list[int]:
    if spec == "all":
        return list(range(n_layers))
    if isinstance(spec, (list, tuple)):
        return [int(i) for i in spec]
    raise ValueError(f"Unsupported layers spec: {spec!r}; expected 'all' or a list of ints.")


def _wrap_layer_attention(layer: nn.Module, growable: GrowableRobertaSelfAttention) -> None:
    """Swap layer.attention.self with `growable`. HF's RobertaAttention.forward
    calls `self.self(...)` and gets back (attn_output, attn_weights); our
    replacement returns the same tuple shape, so no further patching is needed.
    """
    layer.attention.self = growable
    # nn.Module.__setattr__ does NOT propagate the parent's train/eval state to
    # newly assigned children. Without this, an eval-mode model would still run
    # the new module's dropout because growable defaults to training=True.
    growable.train(layer.training)


def _wrap_layer_ffn(layer: nn.Module, growable: GrowableRobertaFFN) -> None:
    """Route layer.feed_forward_chunk through `growable`, preserving dropout+LN+residual.

    The original HF chunk:
        intermediate_output = self.intermediate(attention_output)           # dense + GELU
        layer_output = self.output(intermediate_output, attention_output)   # dense + dropout + LN(+residual)

    The replacement:
        ffn_output = self.growable_ffn(attention_output)                    # dense + GELU + dense
        return self.output.LayerNorm(self.output.dropout(ffn_output) + attention_output)
    """
    layer.growable_ffn = growable

    # Drop dead params (intermediate.* and output.dense) by replacing with Identity.
    # output.LayerNorm and output.dropout stay because we still call them above.
    layer.intermediate = nn.Identity()
    layer.output.dense = nn.Identity()

    def _new_feed_forward_chunk(self: nn.Module, attention_output: torch.Tensor) -> torch.Tensor:
        ffn_output = self.growable_ffn(attention_output)
        return self.output.LayerNorm(self.output.dropout(ffn_output) + attention_output)

    layer.feed_forward_chunk = types.MethodType(_new_feed_forward_chunk, layer)
    growable.train(layer.training)


def make_student_growable(
    student: nn.Module,
    grow_cfg: dict[str, Any],
) -> dict[str, list[nn.Module]]:
    """In-place surgery on a HF RoBERTa student. Returns a registry of growable modules.

    Args:
        student: a RobertaForSequenceClassification (or similar with `.roberta.encoder.layer`).
        grow_cfg: dict matching configs/grow.yaml. Specifically reads `ffn.enabled`,
            `ffn.layers`, and (Phase 3) `attention.*`.

    Returns:
        {"ffn": [GrowableRobertaFFN, ...], "attention": []}.
    """
    registry: dict[str, list[nn.Module]] = {"ffn": [], "attention": []}

    # Resolve encoder layers (handles RobertaModel / RobertaForSequenceClassification).
    if hasattr(student, "roberta"):
        encoder = student.roberta.encoder
    elif hasattr(student, "encoder"):
        encoder = student.encoder
    else:
        raise AttributeError(
            "Could not locate encoder layers on the student model; expected `.roberta.encoder` or `.encoder`.",
        )

    n_layers = len(encoder.layer)

    ffn_cfg = grow_cfg.get("ffn", {})
    if ffn_cfg.get("enabled", False):
        for idx in _resolve_layer_indices(ffn_cfg.get("layers", "all"), n_layers):
            layer = encoder.layer[idx]
            growable = GrowableRobertaFFN.from_pretrained(
                intermediate_dense=layer.intermediate.dense,
                output_dense=layer.output.dense,
                activation=layer.intermediate.intermediate_act_fn,
            )
            _wrap_layer_ffn(layer, growable)
            registry["ffn"].append(growable)

    attn_cfg = grow_cfg.get("attention", {})
    if attn_cfg.get("enabled", False):
        for idx in _resolve_layer_indices(attn_cfg.get("layers", "all"), n_layers):
            layer = encoder.layer[idx]
            growable = GrowableRobertaSelfAttention.from_hf(layer.attention.self)
            _wrap_layer_attention(layer, growable)
            registry["attention"].append(growable)

    return registry
