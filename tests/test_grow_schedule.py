"""Tests for distill_nli.growing.schedule.GrowthSchedule."""

from __future__ import annotations

from distill_nli.growing.schedule import GrowthSchedule


def test_warmup_blocks_early_epochs():
    s = GrowthSchedule(warmup_epochs=2, interval_epochs=1, max_grows=5)
    assert not s.is_growth_epoch(0)
    assert not s.is_growth_epoch(1)
    assert s.is_growth_epoch(2)


def test_interval_skips_non_aligned_epochs():
    s = GrowthSchedule(warmup_epochs=1, interval_epochs=3, max_grows=10)
    # epochs 1, 4, 7, ... are growth epochs
    assert not s.is_growth_epoch(0)
    assert s.is_growth_epoch(1)
    assert not s.is_growth_epoch(2)
    assert not s.is_growth_epoch(3)
    assert s.is_growth_epoch(4)
    assert s.is_growth_epoch(7)


def test_max_grows_caps_total():
    s = GrowthSchedule(warmup_epochs=0, interval_epochs=1, max_grows=2)
    assert s.is_growth_epoch(0)
    s.record_grow()
    assert s.is_growth_epoch(1)
    s.record_grow()
    # cap reached
    assert not s.is_growth_epoch(2)
    assert not s.is_growth_epoch(3)
    assert s.grows_remaining == 0


def test_grows_done_and_remaining_consistency():
    s = GrowthSchedule(warmup_epochs=0, interval_epochs=1, max_grows=3)
    assert s.grows_done == 0
    assert s.grows_remaining == 3
    s.record_grow()
    s.record_grow()
    assert s.grows_done == 2
    assert s.grows_remaining == 1


def test_interval_zero_is_treated_as_one():
    # Guard against accidental divide-by-zero from a misconfigured interval.
    s = GrowthSchedule(warmup_epochs=0, interval_epochs=0, max_grows=10)
    assert s.is_growth_epoch(0)
    assert s.is_growth_epoch(1)
    assert s.is_growth_epoch(5)
