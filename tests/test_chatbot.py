#!/usr/bin/env python3

import pytest
from chat.chat import Chatbot
from chat.exceptions import ChatConfigError


def test_validate_inputs_valid():
    """Test that _validate_inputs accepts valid input parameters."""
    # Valid inputs
    system = 'You are a helpful assistant'
    user_assistant = ['Hello', 'Hi there']
    max_tokens = 4096

    # Should not raise any exceptions
    Chatbot._validate_inputs(system, user_assistant, max_tokens)


def test_validate_inputs_invalid_type():
    """Test that _validate_inputs rejects invalid input types."""
    system = 'You are a helpful assistant'
    user_assistant = 'This should be a list, not a string'
    max_tokens = 4096

    with pytest.raises(ChatConfigError):
        Chatbot._validate_inputs(system, user_assistant, max_tokens)

def test_validate_inputs_invalid_max_tokens():
    """Test that _validate_inputs rejects invalid max_tokens."""
    system = 'You are a helpful assistant'
    user_assistant = ['Hello', 'Hi there']
    max_tokens = -1

    with pytest.raises(ChatConfigError):
        Chatbot._validate_inputs(system, user_assistant, max_tokens)