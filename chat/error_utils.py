#!/usr/bin/env python3
"""Utilities for consistent error formatting."""

from chat.format import COLORS
from chat.logging_setup import setup_logging

logger = setup_logging(__name__)


def format_error_message(title: str, message: str, width: int = 80) -> str:
    """Format an error message with a title and wrapped text."""
    title_text = f"\n{COLORS['ERROR']}{title}{COLORS['RESET']}"
    message_text = f"{COLORS['ERROR']}{message}{COLORS['RESET']}"

    return f'{title_text}: {message_text}'


def log_error(message: str, title: str = 'Error') -> None:
    """Log a simple error message."""
    formatted_message = format_error_message(title, message)
    logger.error(formatted_message)
