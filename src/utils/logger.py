"""
Logging configuration.
"""

from __future__ import annotations

import logging
import sys


def setup_logging(debug: bool = False) -> None:
    """
    Configure the root logger.

    Parameters
    ----------
    debug:
        If ``True``, set level to DEBUG; otherwise INFO.
    """
    level = logging.DEBUG if debug else logging.INFO
    fmt = "%(asctime)s  %(levelname)-8s  %(name)s  %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"

    logging.basicConfig(
        level=level,
        format=fmt,
        datefmt=datefmt,
        stream=sys.stdout,
    )

    # Quiet noisy third-party loggers
    for noisy in ("urllib3", "PIL"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
