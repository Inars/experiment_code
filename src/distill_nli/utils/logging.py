"""Rich-based logger for training/eval scripts."""

from __future__ import annotations

import logging

from rich.logging import RichHandler


def get_logger(name: str = "distill_nli", level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(level)
    handler = RichHandler(rich_tracebacks=True, show_path=False, show_time=True)
    handler.setLevel(level)
    logger.addHandler(handler)
    logger.propagate = False
    return logger
