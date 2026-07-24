"""Standard structured logging with a single stdout handler (no duplicate handlers)."""

import logging
import sys


def get_logger(name: str = "aresseg", level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        h = logging.StreamHandler(sys.stdout)
        h.setFormatter(
            logging.Formatter(
                "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )
        logger.addHandler(h)
        logger.setLevel(level)
        logger.propagate = False
    return logger
