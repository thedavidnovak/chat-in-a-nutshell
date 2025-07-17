#!/usr/bin/env python3

import json
import os

import aiofiles
from typing import Any

from chat.format import (
    log_chat_details,
    format_header,
    format_token_use,
    format_tool_use,
)
from chat.logging_setup import setup_logging
from chat.error_utils import log_error
from chat.exceptions import (
    ChatConfigError,
    ToolLoadError,
    ProviderError,
)
from chat.mcp_client import MCPClient
from chat.providers.base import Provider, ProviderFactory

logger = setup_logging(__name__)

# Environment variables
MCP_CONFIG_PATH = os.getenv('MCP_CONFIG_PATH')

# Default configuration values
DEFAULT_CONFIG = {
    'system_message': 'You are a skilled Python programmer who writes tersely.',
    'model': 'gpt-3.5-turbo',
    'temperature': 0.0,
    'reasoning_effort': 'low',
    'max_tokens': 4096,
    'provider': 'openai',
    'user_messages': [''],
    'use_tools': False,
}

PROVIDER_MAP = {
    'openai': Provider.OPENAI,
    'anthropic': Provider.ANTHROPIC,
}

class Chatbot:
    """Main chatbot class for interacting with AI providers and tools."""

    def __init__(self, config_file: str, provider_name: str | None = None) -> None:
        """Initialize the Chatbot with configuration and provider.

        Note: This initializes basic properties but doesn't do any async operations.
        Use the create() class method for proper async initialization.

        :param config_file: Path to the configuration file
        :param provider_name: Optional provider name to override config
        """
        self.config_file = config_file
        self.config = DEFAULT_CONFIG.copy()
        self._provider_name = provider_name
        self.provider_instance = None
        self.tools: list[dict[str, Any]] = []
        self.mcp_client: MCPClient | None = None

    async def __aenter__(self):
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit with cleanup."""
        if self.mcp_client:
            await self.mcp_client.cleanup()

    @classmethod
    async def create(cls, config_file: str, provider_name: str | None = None) -> 'Chatbot':
        """Create and initialize a Chatbot instance without tools."""
        instance = cls(config_file, provider_name)

        # Load configuration
        await instance._load_config()

        # Set provider from parameter or config
        if provider_name:
            instance.config['provider'] = provider_name

        provider = PROVIDER_MAP.get(instance.provider)
        if not provider:
            raise ChatConfigError(f'Unknown provider: {instance.provider}')

        # Create provider instance
        instance.provider_instance = await ProviderFactory.create_provider(provider)

        return instance

    async def _load_config(self) -> None:
        """Load configuration from file, creating default if not found."""
        try:
            async with aiofiles.open(self.config_file, 'r') as f:
                file_content = await f.read()

            loaded_config = json.loads(file_content)

            # Merge with defaults to ensure all keys exist
            self.config.update(loaded_config)
        except FileNotFoundError:
            # Save default config if file doesn't exist
            await self._save_config()
        except json.JSONDecodeError as e:
            log_error(f'Invalid JSON in config file: {e}')
            await self._save_config()

    async def _save_config(self) -> None:
        """Save current configuration to the config file."""
        config_json = json.dumps(self.config, indent=4)

        async with aiofiles.open(self.config_file, 'w') as f:
            await f.write(config_json)

    async def initialize_tools(self) -> None:
        """Initialize the MCP client and connect to servers."""
        if hasattr(self, 'mcp_client') and self.mcp_client:
            await self.mcp_client.cleanup()

        if not MCP_CONFIG_PATH:
            log_error("No MCP config path provided", 'Warning')
            self.tools = []
            return

        self.mcp_client = MCPClient(config_path=MCP_CONFIG_PATH)
        failed_servers = []

        try:
            # Connect to servers from JSON configuration
            if MCP_CONFIG_PATH:
                for server_name in self.mcp_client.get_server_names():
                    success = await self.mcp_client.connect_to_server(server_name)
                    if not success:
                        failed_servers.append(server_name)
                    logger.debug(f"Server {server_name} connection: {'success' if success else 'failed'}")

            # Update tools cache
            self.tools = self.mcp_client.get_tools()
            logger.debug(f"Loaded {len(self.tools)} tools: {[tool['name'] for tool in self.tools]}")

            if failed_servers:
                log_error(f"Failed to connect to servers: {failed_servers}", 'Warning')

        except Exception as e:
            raise ToolLoadError("Failed to initialize MCP client.") from e

    async def chat_with_provider(
        self,
        *,
        system: str,
        user_assistant: list[str],
        model: str,
        temperature: float,
        max_tokens: int,
    ) -> tuple[str, int]:
        """Chat with provider using the provider interface.

        :param system: The system prompt text
        :param user_assistant: List of alternating user/assistant messages
        :param model: The model to use
        :param temperature: Temperature parameter for generation
        :param max_tokens: Maximum tokens to generate
        :return: A tuple of (content, token_count)
        :raises: Various provider-specific exceptions on failure
        """
        self._validate_inputs(system, user_assistant, max_tokens)

        # Prepare tools if enabled
        tools = self.tools if self.use_tools else None

        return await self._send_provider_request(
            messages=user_assistant,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            tools=tools,
            system=system,
        )

    async def confirm_callback(self, tool_name: str, tool_args: dict) -> bool:
        """Simple console-based confirmation callback."""
        response = input("Proceed? (y/N): ").strip().lower()
        print()
        return response in ('y', 'yes')

    async def _send_provider_request(
        self,
        messages: list[Any],
        model: str,
        temperature: float,
        max_tokens: int,
        tools: list | None = None,
        system: str | None = None,
        depth: int = 0,
        max_depth: int = 10,
    ) -> tuple[str, int]:
        """Send a request to the provider and process the response.

        :param messages: Messages list for the provider
        :param model: The model to use
        :param temperature: Temperature parameter for generation
        :param max_tokens: Maximum tokens to generate
        :param tools: Optional tools configuration
        :param system: System message
        :param depth: Current recursion depth
        :param max_depth: Maximum allowed recursion depth
        :return: A tuple of (content, token_count)
        """

        if depth >= max_depth:
            raise ProviderError(f'Maximum recursion depth ({max_depth}) exceeded')

        # Get response from provider
        response = await self.provider_instance.send_request(
            user_assistant=messages,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            reasoning_effort=self.reasoning_effort,
            tools=tools,
            system=system,
        )

        # Unpack response
        content, tokens, tool_info = response

        # Handle tool calls if present
        while tool_info:
            logger.debug(f'Iteration: {depth}')
            if not self.mcp_client:
                raise ProviderError("MCP client not available for tool execution")

            tool_calls = tool_info['tool_calls']
            thinking_blocks = tool_info['thinking_blocks']
            text_blocks = tool_info['text_blocks']

            # Execute all tool calls
            tool_results = []
            for tool_call in tool_calls:
                tool_id = tool_call.get('id', '')
                tool_name = tool_call.get('name', '')
                tool_args = tool_call.get('input', {})

                logger.info(format_tool_use(tool_name, tool_args))

                try:
                    result = await self.mcp_client.call_tool(tool_name, tool_args, self.confirm_callback)

                    if isinstance(result, dict):
                        content_result = result.get('content', str(result))
                        status = result.get('status', 'unknown')
                    else:
                        content_result = str(result)
                        status = 'success'

                    logger.debug(f"Tool {tool_name} result (status: {status}): {str(content_result)[:200]}...")

                    tool_results.append({
                        'type': 'tool_result',
                        'tool_use_id': tool_id,
                        'content': content_result
                    })

                except Exception as tool_error:
                    error_msg = f"Tool {tool_name} failed with args {tool_args}: {tool_error}"
                    log_error(error_msg, tool_error.__class__.__name__)
                    tool_results.append({
                        'type': 'tool_result',
                        'tool_use_id': tool_id,
                        'content': f'Error executing tool {tool_name}: {str(tool_error)}'
                    })

            # Create follow-up messages with tool results
            follow_up_messages = messages.copy()

            # Add the assistant's response with all original blocks
            assistant_content = []
            assistant_content.extend(thinking_blocks)
            assistant_content.extend(text_blocks)
            assistant_content.extend(tool_calls)

            logger.debug(f"Assistant content blocks for follow-up: {len(assistant_content)} "
                         f"(thinking: {len(thinking_blocks)}, text: {len(text_blocks)}, tools: {len(tool_calls)})")

            follow_up_messages.append(assistant_content)

            # Add tool results as user message
            follow_up_messages.append(tool_results)

            logger.debug(f"Sending follow-up request with {len(follow_up_messages)} messages")

            # Recursively call _send_provider_request for follow-up
            return await self._send_provider_request(
                messages=follow_up_messages,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                tools=tools,
                system=system,
                depth=depth + 1,
                max_depth=max_depth,
            )

        # No tool calls, return the response as-is
        return content, tokens

    @staticmethod
    def _validate_inputs(system: str, user_assistant: list[str], max_tokens: int) -> None:
        """Validate input types for chat requests.

        :param system: The system prompt text
        :param user_assistant: List of alternating user/assistant messages
        :param max_tokens: Maximum tokens to generate
        :raises ChatConfigError: If input types are invalid
        """
        if not isinstance(system, str):
            raise ChatConfigError('System prompt must be a string.')

        if not isinstance(user_assistant, list):
            raise ChatConfigError('User-assistant messages must be a list.')

        if any(not isinstance(msg, str) for msg in user_assistant):
            raise ChatConfigError('All messages in user_assistant must be strings.')

        if max_tokens <= 0:
            raise ChatConfigError('Max_tokens must be positive integer.')

    async def chat(
        self,
        *,
        system: str,
        user_assistant: list[str],
        model: str,
        temperature: float,
        max_tokens: int,
    ) -> str:
        """Chat with the AI provider and display formatted output.

        :param system: The system prompt text
        :param user_assistant: List of alternating user / assistant messages
        :param model: The model name to use
        :param temperature: The temperature setting for generation
        :param max_tokens: Maximum number of tokens to generate
        :return: The generated content as a string or an error message
        """
        log_chat_details(
            provider=self.provider,
            model=model,
            reasoning_effort=self.reasoning_effort,
            max_tokens=max_tokens,
            temperature=temperature,
            use_tools=self.use_tools,
            system_message=system,
            user_messages=user_assistant,
        )

        try:
            content, tokens = await self.chat_with_provider(
                system=system,
                user_assistant=user_assistant,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
            )

            logger.info(format_header())
            print(content)
            logger.info(format_token_use(tokens))
            return content

        except Exception as e:
            log_error(str(e), e.__class__.__name__)
            return str(e)

    # Model and tool management methods
    async def get_model_list(self, model_type: str | None = None) -> str:
        """Get models from the current provider with optional filtering.

        :return: Formatted string with model information
        """

        try:
            return await self.provider_instance.list_models()
        except Exception as e:
            raise ProviderError('Could not get models list.') from e

    def get_tool_list(self) -> str:
        """Get list of available tools.

        :return: Formatted string with all available tool information
        """
        if not self.tools:
            return 'No tools are currently available.'

        tool_list = []
        for tool in self.tools:
            if 'name' in tool and 'description' in tool:
                tool_info = {
                    'name': tool['name'],
                    'description': tool['description'],
                    'definition': tool['definition'],
                }
                tool_list.append(json.dumps(tool_info, indent=4))
            else:
                log_error(f'Invalid tool entry detected: {tool}', 'Warning')

        return '\n\n'.join(tool_list) if tool_list else 'No valid tools are currently available.'

    # Configuration properties with consistent patterns
    def _get_config_value(self, key: str, default_value: Any) -> Any:
        """Get a configuration value with fallback to default."""
        return self.config.get(key, default_value)

    async def _set_config_value(self, key: str, value) -> None:
        """Set a configuration value and save."""
        self.config[key] = value
        await self._save_config()

    @property
    def system_message(self) -> str:
        """Get the system message."""
        return self._get_config_value('system_message', DEFAULT_CONFIG['system_message'])

    async def set_system_message(self, message: str) -> None:
        """Set the system message."""
        await self._set_config_value('system_message', message)

    @property
    def model(self) -> str:
        """Get the current model."""
        return self._get_config_value('model', DEFAULT_CONFIG['model'])

    async def set_model(self, model: str) -> None:
        """Set the model."""
        await self._set_config_value('model', model)

    @property
    def temperature(self) -> float:
        """Get the temperature setting."""
        return self._get_config_value('temperature', DEFAULT_CONFIG['temperature'])

    async def set_temperature(self, temperature: float) -> None:
        """Set the temperature value."""
        if temperature < 0:
            raise ValueError('Negative values not allowed.')
        await self._set_config_value('temperature', temperature)

    @property
    def reasoning_effort(self) -> str:
        """Get the reasoning effort setting."""
        return self._get_config_value('reasoning_effort', DEFAULT_CONFIG['reasoning_effort'])

    async def set_reasoning_effort(self, effort: str) -> None:
        """Set the reasoning effort."""
        if effort.lower() not in ('low', 'medium', 'high'):
            raise ChatConfigError(f'Expected one of allowed values: low, medium, high. Got: {effort}')
        await self._set_config_value('reasoning_effort', effort)

    @property
    def max_tokens(self) -> int:
        """Get the maximum tokens setting."""
        return self._get_config_value('max_tokens', DEFAULT_CONFIG['max_tokens'])

    async def set_max_tokens(self, max_tokens: int) -> None:
        """Set the maximum tokens."""
        if max_tokens < 0:
            raise ValueError('Negative values not allowed.')
        await self._set_config_value('max_tokens', max_tokens)

    @property
    def provider(self) -> str:
        """Get the provider setting."""
        return self._get_config_value('provider', DEFAULT_CONFIG['provider'])

    async def set_provider(self, provider: str) -> None:
        """Set the provider."""
        await self._set_config_value('provider', provider)

    @property
    def user_messages(self) -> list[str]:
        """Get the user messages list."""
        return self._get_config_value('user_messages', DEFAULT_CONFIG['user_messages'])

    async def set_user_messages(self, messages: list[str]) -> None:
        """Set the user messages list."""
        await self._set_config_value('user_messages', messages)

    async def append_message(self, message: str) -> None:
        """Append a message to the user messages list."""
        current_messages = self.user_messages.copy()
        current_messages.append(message)
        await self.set_user_messages(current_messages)

    @property
    def use_tools(self) -> bool:
        """Get the tool usage preference."""
        return self._get_config_value('use_tools', DEFAULT_CONFIG['use_tools'])

    async def set_use_tools(self, use: bool) -> None:
        """Set the tool usage preference."""
        await self._set_config_value('use_tools', use)