#!/usr/bin/env python3
"""Utilities for consistent formatting across the application."""
import re
import textwrap
from datetime import datetime
from chat.logging_setup import setup_logging

logger = setup_logging(__name__)

# ANSI color definitions
COLORS = {
    'BOLD': '\033[1m',
    'NORMAL_WEIGHT': '\033[22m',
    'RESET': '\033[0m',
    'STRING_TEAL': '\033[38;5;30m',
    'FUNC_CALL': '\033[38;5;23m',
    'MAGENTA': '\033[38;5;54m',
    'BLUE': '\033[38;5;24m',
    'YELLOW': '\033[38;5;136m',
    'GRAY': '\033[38;5;236m',
    'ERROR': '\033[38;5;124m',
    'WARNING': '\033[38;5;130m',
    'SUCCESS': '\033[38;5;22m',
    'GREEN': '\033[38;5;22m',
    'CODE_DARK': '\033[48;5;235m\033[38;5;75m',
    'CODE_LIGHT': '\033[48;5;255m\033[38;5;24m',
}


def format_section_header(text):
    """Format a section header consistently."""
    return f"\n{COLORS['BOLD']}{text}{COLORS['RESET']}\n"


def format_separator(length=50):
    """Create a consistent separator line."""
    return f"{COLORS['GRAY']}{'─' * length}{COLORS['RESET']}\n"


def format_bullet_item(label, value, bold_value=True, min_label_width=12):
    """Format a consistent bullet point item with aligned values."""
    formatted_value = f"{COLORS['BOLD']}{value}{COLORS['RESET']}" if bold_value else value
    padded_label = f'{label}:'.ljust(min_label_width)
    return f"  {COLORS['YELLOW']}•{COLORS['RESET']} {padded_label} {formatted_value}\n"


def wrap_text(text, width=45, max_lines=4):
    """Wrap text and truncate if necessary, returning lines and truncation info."""
    text = text.strip()
    wrapped = textwrap.wrap(text, width=width)
    truncated = len(wrapped) > max_lines
    return wrapped[:max_lines], truncated, len(wrapped) - max_lines if truncated else 0


def format_chat_config(*, provider, model, reasoning_effort=None, max_tokens=None,
                       temperature=None, use_tools=None, system_message=None,
                       user_messages=None, include_system_prompt=False, include_user_msg=False):
    """Format chatbot configuration items in a standardized way."""
    items = [
        ('Provider', provider),
        ('Model', model),
    ]

    # Add reasoning effort if using a reasoning model
    is_reasoning_model = bool(re.search(r'^(o\d|claude-(opus-4|sonnet-4|3-7-sonnet))', model))
    if is_reasoning_model:
        items.append(('Reasoning', reasoning_effort or 'Not set'))

    # Add max tokens
    max_tokens_display = str(max_tokens) if max_tokens else 'Default'
    items.append(('Max tokens', max_tokens_display))

    # Add temperature
    items.append(('Temperature', temperature))

    # Add tools status
    items.append(('Tools', 'Enabled' if use_tools else 'Disabled'))

    # Format items as bullet points
    formatted = [format_bullet_item(label, value) for label, value in items]

    # Add system prompt section if requested
    if include_system_prompt:
        formatted.append(f"\n{COLORS['BOLD']}System Prompt{COLORS['RESET']}\n")
        wrapped_system, truncated, remaining = wrap_text(
            system_message, width=68, max_lines=4
        )
        for line in wrapped_system:
            formatted.append(f'  {line}\n')
        if truncated:
            formatted.append(f"  {COLORS['GRAY']}... ({remaining} more lines){COLORS['RESET']}\n")

    # Add user message section if requested and available
    if include_user_msg and user_messages:
        formatted.append(f"\n{COLORS['BOLD']}Latest User Message{COLORS['RESET']}\n")
        wrapped_user, truncated, remaining = wrap_text(
            user_messages[-1], width=68, max_lines=4
        )
        for line in wrapped_user:
            formatted.append(f'  {line}\n')
        if truncated:
            formatted.append(f"  {COLORS['GRAY']}... ({remaining} more lines){COLORS['RESET']}\n")

    return formatted


def print_settings(*, provider, model, reasoning_effort=None, max_tokens=None,
                   temperature=None, use_tools=None, system_message=None,
                   user_messages=None, header):
    """Print chatbot settings in a clean, modern, and well-formatted style."""
    output = [format_section_header(header)]
    output.extend(format_chat_config(
        provider=provider,
        model=model,
        reasoning_effort=reasoning_effort,
        max_tokens=max_tokens,
        temperature=temperature,
        use_tools=use_tools,
        system_message=system_message,
        user_messages=user_messages,
        include_system_prompt=True
    ))
    print(''.join(output))


def log_chat_details(*, provider, model, reasoning_effort=None, max_tokens=None,
                     temperature=None, use_tools=None, system_message=None,
                     user_messages=None):
    """Log formatted chat details."""
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    color_header = f"{COLORS['BLUE']}{COLORS['BOLD']}CHAT{COLORS['RESET']}"
    header = f"\n{color_header} • {COLORS['GRAY']}{timestamp}{COLORS['RESET']}\n\n"

    sections = [header, f"{COLORS['BOLD']}Configuration{COLORS['RESET']}\n"]
    sections.extend(format_chat_config(
        provider=provider,
        model=model,
        reasoning_effort=reasoning_effort,
        max_tokens=max_tokens,
        temperature=temperature,
        use_tools=use_tools,
        system_message=system_message,
        user_messages=user_messages,
        include_system_prompt=True,
        include_user_msg=True
    ))

    logger.info(''.join(sections))


def format_header():
    """Return formatted header."""
    return f"\n{COLORS['BLUE']}{COLORS['BOLD']}MODEL RESPONSE:{COLORS['RESET']}\n"


def format_token_use(tokens):
    """Format token use."""
    return f"\n{COLORS['GRAY']}Tokens used: {COLORS['BOLD']}{tokens}{COLORS['RESET']}"

def format_tool_use(tool_name, tool_args):
    """Format tool use."""
    text = f"Executing tool {COLORS['BOLD']}{tool_name}{COLORS['NORMAL_WEIGHT']}"
    if tool_args:
        text += " with args:\n"
        for k, v in tool_args.items():
            text += f"• {k}: {v}\n"
    else:
        text += ".\n"
    return f"\n{COLORS['FUNC_CALL']}{text}{COLORS['RESET']}"