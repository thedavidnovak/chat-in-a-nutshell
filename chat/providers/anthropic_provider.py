#!/usr/bin/env python3

import os

from chat.exceptions import (
    AuthenticationError,
    ProviderError,
    ResponseFormatError,
)
from chat.error_utils import log_error
from chat.logging_setup import setup_logging
from chat.providers.base import ProviderInterface

from anthropic import AsyncAnthropic


logger = setup_logging(__name__)


class AnthropicProvider(ProviderInterface):
    """Anthropic provider implementation."""

    def __init__(self):
        try:
            self.client = AsyncAnthropic(api_key=os.environ['ANTHROPIC_API_KEY'])
        except KeyError as e:
            raise AuthenticationError('Environment variable "ANTHROPIC_API_KEY" not set.') from e
        except Exception as e:
            raise ProviderError('Error occurred when initializing provider instance.') from e

    async def send_request(
        self,
        user_assistant: list[dict],
        model: str,
        temperature: float,
        max_tokens: int,
        reasoning_effort: str | None = None,
        tools: list | None = None,
        system: str | None = None,
    ) -> tuple[str, int]:
        """Send chat completion request to Anthropic.

        :param user_assistant: List of message dictionaries
        :param model: Model identifier to use
        :param temperature: Sampling temperature
        :param max_tokens: Maximum tokens in response
        :param reasoning_effort: Optional reasoning effort parameter
        :param tools: Optional list of tools
        :param system: Optional system message
        :returns: Tuple of (content_string, total_tokens, tool_info_dict_or_none)
        """
        try:
            messages = [
                {'role': 'assistant', 'content': msg} if i % 2 else {'role': 'user', 'content': msg}
                for i, msg in enumerate(user_assistant)
            ]

            # Determine if extended thinking should be enabled
            should_use_thinking = False
            thinking_config = None
            if reasoning_effort and reasoning_effort.lower() in ['medium', 'high']:
                should_use_thinking = True
                budget_map = {'medium': int(max_tokens * 0.5), 'high': int(max_tokens * 0.9)}
                thinking_config = {
                    "type": "enabled",
                    "budget_tokens": budget_map.get(reasoning_effort.lower(), 'medium')
                }

            # Prepare messages for thinking mode
            if should_use_thinking:
                prepared_messages = []
                for message in messages:
                    role = message.get('role')
                    content = message.get('content')

                    if isinstance(content, str):
                        prepared_content = [{"type": "text", "text": content}]
                    elif isinstance(content, list):
                        prepared_content = []
                        for block in content:
                            if isinstance(block, dict):
                                prepared_content.append(block)
                            else:
                                prepared_content.append({"type": "text", "text": str(block)})
                    else:
                        prepared_content = [{"type": "text", "text": str(content)}]

                    prepared_messages.append({"role": role, "content": prepared_content})
                messages = prepared_messages

            # Adjust temperature based on thinking configuration
            actual_temperature = 1.0 if should_use_thinking else temperature

            request_params = {
                'model': model,
                'messages': messages,
                'temperature': actual_temperature,
                'max_tokens': max_tokens,
            }

            # Add system message if provided
            if system:
                request_params['system'] = system

            # Add thinking configuration if needed
            if thinking_config:
                request_params['thinking'] = thinking_config

            # Add tools if provided
            if tools:
                logger.debug(f"Processing {len(tools)} tools for Anthropic")
                anthropic_tools = []
                for tool in tools:
                    logger.debug(f"Converting tool: {tool.get('name', 'Unknown')}")
                    if 'definition' in tool and 'function' in tool['definition']:
                        func_def = tool['definition']['function']
                        anthropic_tools.append({
                            'name': func_def['name'],
                            'description': func_def['description'],
                            'input_schema': func_def['parameters']
                        })

                if anthropic_tools:
                    request_params['tools'] = anthropic_tools
                    logger.debug(f"Added {len(anthropic_tools)} tools to request")

            logger.debug(f'Sending streaming request to Anthropic with params keys: {list(request_params.keys())}')

            # Use streaming with get_final_message
            try:
                async with self.client.messages.stream(**request_params) as stream:
                    async for event in stream:
                        if event.type == "text":
                            # Process text events for real-time streaming if needed
                            pass
                        elif event.type == 'content_block_stop':
                            # Handle content block completion
                            pass
            except Exception as e:
                raise ProviderError(f'Request failed: {e}') from e

            # Get the final accumulated message
            final_message = await stream.get_final_message()

            # Process the final message
            contents = final_message.content
            stop_reason = final_message.stop_reason
            input_tokens = final_message.usage.input_tokens
            output_tokens = final_message.usage.output_tokens
            total_tokens = input_tokens + output_tokens

            # Check for tool calls
            has_tool_calls = False
            tool_calls = []
            for item in contents:
                if isinstance(item, dict) and item.get('type') == 'tool_use':
                    has_tool_calls = True
                    tool_calls.append(item)
                elif hasattr(item, 'type') and getattr(item, 'type') == 'tool_use':
                    has_tool_calls = True
                    tool_calls.append({
                        'type': 'tool_use',
                        'id': getattr(item, 'id', ''),
                        'name': getattr(item, 'name', ''),
                        'input': getattr(item, 'input', {}),
                    })

            # Handle tool calls if present
            if stop_reason == 'tool_use' or has_tool_calls:
                # Extract all content blocks
                thinking_blocks = []
                text_blocks = []
                redacted_thinking_blocks = []

                for item in contents:
                    if isinstance(item, dict):
                        block_type = item.get('type')
                        if block_type == 'thinking':
                            thinking_block = {
                                'type': 'thinking',
                                'thinking': item.get('thinking', '')
                            }
                            if 'signature' in item:
                                thinking_block['signature'] = item['signature']
                            thinking_blocks.append(thinking_block)
                        elif block_type == 'redacted_thinking':
                            redacted_thinking_blocks.append({
                                'type': 'redacted_thinking',
                                'data': item.get('data', '')
                            })
                        elif block_type == 'text':
                            text_blocks.append(item)
                    elif hasattr(item, 'type'):
                        block_type = getattr(item, 'type')
                        if block_type == 'thinking':
                            thinking_block = {
                                'type': 'thinking',
                                'thinking': getattr(item, 'thinking', '')
                            }
                            if hasattr(item, 'signature'):
                                thinking_block['signature'] = getattr(item, 'signature')
                            thinking_blocks.append(thinking_block)
                        elif block_type == 'redacted_thinking':
                            redacted_thinking_blocks.append({
                                'type': 'redacted_thinking',
                                'data': getattr(item, 'data', '')
                            })
                        elif block_type == 'text':
                            text_blocks.append({
                                'type': 'text',
                                'text': getattr(item, 'text', '')
                            })

                all_thinking_blocks = thinking_blocks + redacted_thinking_blocks

                # Validate thinking blocks if present
                for block in thinking_blocks:
                    if block.get('type') == 'thinking' and 'thinking' not in block:
                        log_error(f"Thinking block missing 'thinking' field: {block.keys()}", 'Warning')

                if not tool_calls:
                    raise ResponseFormatError("No tool calls found in response, but tool call was expected")

                # Return tool information for parent to handle
                tool_info = {
                    'tool_calls': tool_calls,
                    'thinking_blocks': all_thinking_blocks,
                    'text_blocks': text_blocks,
                    'stop_reason': stop_reason
                }

                # Extract text content for consistency
                text_content = ""
                for block in text_blocks:
                    text_content += block.get('text', '')

                return text_content, total_tokens, tool_info

            elif stop_reason and stop_reason != 'end_turn':
                raise ResponseFormatError(f'The Anthropic API returned an unexpected stop reason: {stop_reason}.')

            # Extract the text content from the response
            text_content = ""
            for item in contents:
                if isinstance(item, dict) and item.get('type') == 'text':
                    text_content += item.get('text', '')
                elif hasattr(item, 'type') and getattr(item, 'type') == 'text':
                    text_content += getattr(item, 'text', '')

            if not text_content:
                raise ResponseFormatError('No text content found in the Anthropic response.')

            return text_content, total_tokens, None

        except Exception:
            raise

    async def list_models(self) -> str:
        """Get available Anthropic models and return formatted response.

        :returns: Formatted string with model information
        """
        try:
            # Get models from API
            models = await self.client.models.list()

            # Format models into readable string
            if not hasattr(models, 'data'):
                return 'Unable to retrieve Anthropic models in the expected format.'

            data = models.data

            if not data:
                return 'No models available.'

            # Calculate column widths
            id_width = max(len(model.id) for model in data) + 2
            id_width = max(id_width, len('MODEL ID') + 2)
            name_width = max(len(model.display_name) for model in data) + 2
            name_width = max(name_width, len('DISPLAY NAME') + 2)

            # Create output lines
            lines = []

            # Print header
            lines.append('\n=== AVAILABLE ANTHROPIC MODELS ===\n\n')

            # Print header row
            header = f"{'MODEL ID':<{id_width}} | {'DISPLAY NAME':<{name_width}} | {'RELEASE DATE'}"
            lines.append(header + '\n')
            lines.append('-' * len(header) + '\n')

            # Print each model's information
            for model in data:
                release_date = model.created_at.strftime('%B %d, %Y')
                lines.append(
                    f'{model.id:<{id_width}} | {model.display_name:<{name_width}} | {release_date}\n'
                )

            # Print footer
            lines.append('\n' + '-' * len(header) + '\n')
            if models.has_more:
                lines.append('Note: More models are available. These are the most recent models.\n')
            lines.append(f'Total models shown: {len(data)}\n')
            lines.append('To check all available models, refer to the Anthropic documentation.\n')

            return ''.join(lines)

        except Exception as e:
            raise ProviderError(f'Error retrieving models: {str(e)}')