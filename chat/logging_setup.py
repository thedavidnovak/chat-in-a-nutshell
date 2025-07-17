#!/usr/bin/env python3

import logging
import os
import sys


def setup_logging(name=None, level=None, httpx_level=logging.WARNING, mcp_level=logging.WARNING):
    """Configure application-wide logging.

    :param name: Logger name
    :param level: Logging level, defaults to INFO or value from CHAT_LOG_LEVEL env var
    :param httpx_level: Logging level for httpx
    :param mcp_level: Logging level for mcp
    :return: Configured logger instance
    """
    # Get log level from environment or use default
    env_level = os.environ.get('CHAT_LOG_LEVEL', 'INFO').upper()
    try:
        default_level = getattr(logging, env_level)
    except AttributeError:
        default_level = logging.INFO

    level = level or default_level

    # Only configure root logger once
    root = logging.getLogger()
    if not root.handlers:
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(logging.Formatter('%(message)s'))
        root.addHandler(handler)
        root.setLevel(level)

    # Get or create module-specific logger
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Configure httpx logger
    logging.getLogger('httpx').setLevel(httpx_level)

    # Configure mcp logger
    logging.getLogger('mcp').setLevel(mcp_level)

    return logger
