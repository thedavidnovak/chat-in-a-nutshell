#!/usr/bin/env python3
"""Exception hierarchy for the chat-in-a-nutshell application."""


class ChatError(Exception):
    """Base exception for all chat-related errors."""

    pass


class ChatConfigError(ChatError):
    """Error related to configuration issues."""

    pass


class ChatCompletionError(ChatError):
    """Error during chat completion request."""

    pass


class ProviderError(ChatError):
    """Base exception for provider-related errors."""

    pass


class ConnectionError(ProviderError):
    """Error when unable to connect to provider API."""

    pass


class AuthenticationError(ProviderError):
    """Error when authentication with provider fails."""

    pass


class RateLimitError(ProviderError):
    """Error when provider rate limits are exceeded."""

    pass


class ResourceNotFoundError(ProviderError):
    """Error when a requested resource is not found."""

    pass


class ResponseFormatError(ProviderError):
    """Error when provider response format is unexpected."""

    pass


class ToolError(ChatError):
    """Base exception for tool-related errors."""

    pass


class ToolLoadError(ToolError):
    """Error when loading tools definitions."""

    pass


class ToolExecutionError(ToolError):
    """Error when executing a tool."""

    pass
