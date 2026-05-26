"""TINY-style FFN growth via gromo.

Targets the FFN block of each transformer layer:
    intermediate.dense (E -> 4E) -> activation -> output.dense (4E -> E)

Growth adds neurons to the bottleneck (4E hidden dim). gromo's LinearGrowingModule
handles the natural-gradient computation (see compute_optimal_updates docstring at
gromo/modules/growing_module.py:2424). We use TINY-paper hyperparameters as
documented in that file.

The surgery surrounding this module (replacing layer.feed_forward_chunk) lives in
src/distill_nli/models/growing.py.
"""

from __future__ import annotations

from typing import Callable, Iterable

import torch
from torch import nn

from gromo.containers.growing_container import GrowingContainer
from gromo.modules.linear_growing_module import LinearGrowingModule


# TINY-paper hyperparameters per gromo's docstring.
_TINY_KWARGS = dict(
    compute_delta=False,
    use_covariance=True,
    alpha_zero=False,
    omega_zero=False,
    use_projection=True,
    ignore_singular_values=False,
)


class GrowableRobertaFFN(GrowingContainer):
    """Two-layer growable FFN: Linear(E -> H) -> act -> Linear(H -> E).

    Wraps the part of RobertaLayer that lives between attention_output and the
    residual+LayerNorm in `output`. The surrounding dropout/LayerNorm/residual
    stay outside this module (managed by the surgery in models/growing.py).
    """

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        activation: nn.Module,
        device: torch.device | None = None,
    ) -> None:
        super().__init__(
            in_features=hidden_size, out_features=hidden_size, device=device,
        )
        self.first = LinearGrowingModule(
            in_features=hidden_size,
            out_features=intermediate_size,
            use_bias=True,
            post_layer_function=activation,
            allow_growing=False,
            name="ffn_first",
            device=device,
        )
        self.second = LinearGrowingModule(
            in_features=intermediate_size,
            out_features=hidden_size,
            use_bias=True,
            post_layer_function=nn.Identity(),
            previous_module=self.first,
            allow_growing=True,
            name="ffn_second",
            device=device,
        )
        # The growable boundary is between first and second; gromo expects the
        # call on the downstream module (second).
        self._growing_layers = [self.second]

    def set_growing_layers(self) -> None:
        self._growing_layers = [self.second]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.second(self.first(x))

    @property
    def intermediate_size(self) -> int:
        return self.first.out_features

    @classmethod
    def from_pretrained(
        cls,
        intermediate_dense: nn.Linear,
        output_dense: nn.Linear,
        activation: nn.Module,
    ) -> "GrowableRobertaFFN":
        """Build from a pretrained RoBERTa FFN pair, copying weights and bias."""
        device = intermediate_dense.weight.device
        instance = cls(
            hidden_size=intermediate_dense.in_features,
            intermediate_size=intermediate_dense.out_features,
            activation=activation,
            device=device,
        )
        instance.to(device)
        with torch.no_grad():
            instance.first.layer.weight.copy_(intermediate_dense.weight)
            instance.first.layer.bias.copy_(intermediate_dense.bias)
            instance.second.layer.weight.copy_(output_dense.weight)
            instance.second.layer.bias.copy_(output_dense.bias)
        return instance


LossFn = Callable[[nn.Module, object], torch.Tensor]


def grow_step_ffn(
    student: nn.Module,
    growable_ffns: list[GrowableRobertaFFN],
    probe_batches: Iterable[object],
    loss_fn: LossFn,
    *,
    neurons_per_grow: int,
    alpha: float,
    max_intermediate: int,
) -> dict[int, int]:
    """One TINY-style growth step over all `growable_ffns`.

    Iteration plan (mirrors growing-attention's accumulate-then-grow shape):
        1. set_growing_layers + init_computation on every FFN.
        2. for each probe batch: student.zero_grad(); loss_fn(student, batch).backward();
           then update_computation() on every FFN.
        3. for each FFN: compute_optimal_updates(maximum_added_neurons=cap) and
           apply_change(). Reset computation buffers.

    Args:
        student: the model that loss_fn back-propagates through.
        growable_ffns: list of GrowableRobertaFFN modules embedded in `student`.
        probe_batches: small iterable of batches (typically num_probe_batches).
        loss_fn: callable (student, batch) -> Tensor with grad. Set up by the
            training script: distillation loss when a teacher is present,
            hard-label CE otherwise.
        neurons_per_grow: cap on neurons added per FFN this step.
        alpha: Tikhonov regularization passed to gromo (currently unused at the
            container level; reserved for future when we surface alpha through
            gromo's compute_optimal_updates).
        max_intermediate: hard cap on the bottleneck dim; growth past it is skipped.

    Returns:
        dict {layer_index_in_growable_ffns: neurons_added}.
    """
    del alpha  # gromo's TINY kwargs do not currently take an explicit alpha.

    for ffn in growable_ffns:
        ffn.set_growing_layers()
        ffn.init_computation()

    was_training = student.training
    student.eval()
    try:
        for batch in probe_batches:
            student.zero_grad(set_to_none=True)
            loss = loss_fn(student, batch)
            loss.backward()
            for ffn in growable_ffns:
                ffn.update_computation()
    finally:
        if was_training:
            student.train()

    added: dict[int, int] = {}
    for i, ffn in enumerate(growable_ffns):
        room = max_intermediate - ffn.intermediate_size
        cap = min(neurons_per_grow, max(0, room))
        if cap == 0:
            added[i] = 0
            ffn.reset_computation()
            continue

        size_before = ffn.intermediate_size
        ffn.compute_optimal_updates(maximum_added_neurons=cap, **_TINY_KWARGS)
        ffn.dummy_select_update()  # single growable layer per container
        ffn.apply_change()
        ffn.reset_computation()
        added[i] = ffn.intermediate_size - size_before

    return added
