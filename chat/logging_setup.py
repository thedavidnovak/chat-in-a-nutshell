#!/usr/bin/env python3
import logging

def setup_logging(name: str = None) -> logging.Logger:
    logging.basicConfig(
        encoding="utf-8",
        format="%(message)s",
        level=logging.INFO,
    )
    logger = logging.getLogger(name)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    return logger