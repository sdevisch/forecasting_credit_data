from __future__ import annotations

import logging
import time
from contextlib import contextmanager


def get_logger(name: str = "credit_data") -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        fmt = logging.Formatter("%(asctime)s %(levelname)s %(name)s - %(message)s")
        handler.setFormatter(fmt)
        logger.addHandler(handler)
    return logger


@contextmanager
def timed(logger: logging.Logger, label: str):
    start = time.time()
    try:
        logger.info(f"START {label}")
        yield
    finally:
        elapsed = time.time() - start
        logger.info(f"END {label} - {elapsed:.2f}s")
