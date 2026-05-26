"""Epoch-based growth scheduler.

Mirrors growing-attention's loop logic: do nothing for `warmup_epochs`, then
attempt to grow every `interval_epochs`, stopping after `max_grows` successful
attempts. The scheduler is stateful on the count of successful grows.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class GrowthSchedule:
    """Decides whether the current epoch is a growth epoch.

    `epoch` is zero-indexed across the whole run (epoch 0 = first epoch).
    A successful growth call must be reported via `record_grow()` so the
    `max_grows` cap is respected.
    """

    warmup_epochs: int
    interval_epochs: int
    max_grows: int
    _grows_done: int = 0

    def is_growth_epoch(self, epoch: int) -> bool:
        if self._grows_done >= self.max_grows:
            return False
        if epoch < self.warmup_epochs:
            return False
        return (epoch - self.warmup_epochs) % max(1, self.interval_epochs) == 0

    def record_grow(self) -> None:
        self._grows_done += 1

    @property
    def grows_done(self) -> int:
        return self._grows_done

    @property
    def grows_remaining(self) -> int:
        return max(0, self.max_grows - self._grows_done)
