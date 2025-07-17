#!/usr/bin/env python3

import pytest
import sys
import argparse

from chat_provider import parse_arguments, str_to_float, CustomArgumentParser


def test_str_to_float_valid():
    """Test that str_to_float converts valid strings to floats."""
    assert str_to_float('0.5') == 0.5
    assert str_to_float('1.0') == 1.0
    assert str_to_float('0') == 0.0


def test_str_to_float_invalid():
    """Test that str_to_float raises an exception for invalid strings."""
    with pytest.raises(argparse.ArgumentTypeError):
        str_to_float('not_a_float')


def test_custom_argument_parser_print_usage(capsys):
    """Test CustomArgumentParser's print_usage method."""
    # Create a parser instance
    parser = CustomArgumentParser(prog='test_prog')

    # Call print_usage
    parser.print_usage()

    # Capture the output
    captured = capsys.readouterr()

    # Verify the output contains 'Usage:' and the program name
    assert 'Usage:' in captured.err
    assert 'test_prog' in captured.err


def test_parse_arguments_basic_message(monkeypatch):
    """Test parsing a basic message argument."""
    monkeypatch.setattr(sys, 'argv', ['chat', '-m', 'Test message'])
    args = parse_arguments()
    assert args.messages == ['Test message']
    assert args.conversation is False


def test_parse_arguments_conversation_mode(monkeypatch):
    """Test parsing with conversation mode enabled."""
    monkeypatch.setattr(sys, 'argv', ['chat', '-m', 'Test message', '-c'])
    args = parse_arguments()
    assert args.messages == ['Test message']
    assert args.conversation is True


def test_parse_arguments_with_temperature(monkeypatch):
    """Test parsing with temperature setting."""
    monkeypatch.setattr(sys, 'argv', ['chat', '-m', 'Test message', '-t', '0.7'])
    args = parse_arguments()
    assert args.messages == ['Test message']
    assert args.temperature == 0.7


def test_parse_arguments_provider_selection(monkeypatch):
    """Test parsing with provider selection."""
    monkeypatch.setattr(sys, 'argv', ['chat', '-p', 'anthropic'])
    args = parse_arguments()
    assert args.provider == 'anthropic'


def test_parse_arguments_tools_enabled(monkeypatch):
    """Test parsing with tools enabled."""
    monkeypatch.setattr(sys, 'argv', ['chat', '--use-tools'])
    args = parse_arguments()
    assert args.use_tools is True


def test_parse_arguments_tools_disabled(monkeypatch):
    """Test parsing with tools disabled."""
    monkeypatch.setattr(sys, 'argv', ['chat', '--no-tools'])
    args = parse_arguments()
    assert args.use_tools is False


def test_parse_arguments_reasoning_effort(monkeypatch):
    """Test parsing with reasoning effort setting."""
    monkeypatch.setattr(sys, 'argv', ['chat', '-e', 'high'])
    args = parse_arguments()
    assert args.reasoning_effort == 'high'


def test_parse_arguments_max_tokens(monkeypatch):
    """Test parsing with max tokens setting."""
    monkeypatch.setattr(sys, 'argv', ['chat', '--max-tokens', '2048'])
    args = parse_arguments()
    assert args.max_tokens == 2048

def test_parse_arguments_available_models_gpt(monkeypatch):
    """Test parsing with available GPT models flag."""
    monkeypatch.setattr(sys, 'argv', ['chat', '--available-models-gpt'])
    args = parse_arguments()
    assert args.available_models_gpt is True


def test_parse_arguments_available_tools(monkeypatch):
    """Test parsing with available tools flag."""
    monkeypatch.setattr(sys, 'argv', ['chat', '--available-tools'])
    args = parse_arguments()
    assert args.available_tools is True


def test_parse_arguments_model_specification(monkeypatch):
    """Test parsing with model specification."""
    monkeypatch.setattr(sys, 'argv', ['chat', '--model', 'gpt-4'])
    args = parse_arguments()
    assert args.model == 'gpt-4'


def test_parse_arguments_system_message(monkeypatch):
    """Test parsing with system message."""
    monkeypatch.setattr(sys, 'argv', ['chat', '-s', 'Custom system message'])
    args = parse_arguments()
    assert args.system == 'Custom system message'


def test_parse_arguments_multiple_options(monkeypatch):
    """Test parsing with multiple options combined."""
    monkeypatch.setattr(sys, 'argv', [
        'chat', '-m', 'Hello', '-p', 'anthropic', '-t', '0.8',
        '--use-tools', '-e', 'medium'
    ])
    args = parse_arguments()
    assert args.messages == ['Hello']
    assert args.provider == 'anthropic'
    assert args.temperature == 0.8
    assert args.use_tools is True
    assert args.reasoning_effort == 'medium'