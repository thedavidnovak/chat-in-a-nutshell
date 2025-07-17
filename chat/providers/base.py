#!/usr/bin/env python3

from abc import ABC, abstractmethod
from enum import Enum
from typing import Any

class Provider(Enum):
    """Enumeration of supported chat providers."""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"

class ProviderInterface(ABC):
    """Abstract base class for chat provider implementations."""

    @abstractmethod
    async def send_request(
        self,
        messages: list[dict],
        model: str,
        temperature: float,
        max_tokens: int,
        reasoning_effort: str | None = None,
        tools: list | None = None,
        system: str | None = None,
    ) -> Any:
        """Send chat completion request to provider.

        :param messages: List of message dictionaries
        :param model: Model identifier to use
        :param temperature: Sampling temperature
        :param max_tokens: Maximum tokens in response
        :param reasoning_effort: Optional reasoning effort parameter
        :param tools: Optional list of tools
        :param system: Optional system message
        :returns: Provider response
        """
        pass

    @abstractmethod
    async def list_models(self) -> Any:
        """Get available models from provider.

        :returns: API response with model data
        """
        pass


class ProviderFactory:
    """Factory for creating provider instances."""

    @staticmethod
    async def create_provider(provider: Provider) -> ProviderInterface:
        """Create and return a provider instance based on the provider name.

        :param provider: Provider enum value
        :returns: Provider instance
        :raises ProviderError: If provider is not implemented
        """
        if not isinstance(provider, Provider):
            raise TypeError(f'Expected Provider enum, got {type(provider).__name__}.')
        elif provider == Provider.OPENAI:
            from chat.providers.openai_provider import OpenAIProvider
            return OpenAIProvider()
        elif provider == Provider.ANTHROPIC:
            from chat.providers.anthropic_provider import AnthropicProvider
            return AnthropicProvider()
        else:
            from chat.exceptions import ProviderError
            raise ProviderError(f'Provider "{provider.value}" does not exist.')
