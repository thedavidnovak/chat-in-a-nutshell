#!/usr/bin/env python3

import json
import os

from chat.exceptions import (
    AuthenticationError,
    ProviderError,
    ResponseFormatError,
)
from chat.logging_setup import setup_logging
from chat.providers.base import ProviderInterface

from openai import AsyncOpenAI

logger = setup_logging(__name__)


class OpenAIProvider(ProviderInterface):
    """OpenAI provider implementation."""

    def __init__(self):
        try:
            self.client = AsyncOpenAI(api_key=os.environ['OPENAI_API_KEY'])
        except KeyError as e:
            raise AuthenticationError('Environment variable "OPENAI_API_KEY" not set.') from e
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
        """Send chat completion request to OpenAI.

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
            messages = []

            # Add system message if provided
            if system:
                messages.append({'role': 'system', 'content': system})

            # Convert user_assistant list to OpenAI message format
            for i, msg in enumerate(user_assistant):
                role = 'assistant' if i % 2 else 'user'

                # Handle assistant messages with tool calls
                if role == 'assistant' and isinstance(msg, list):
                    # Extract text content and tool calls
                    text_content = ""
                    tool_calls = []

                    for item in msg:
                        if item.get('type') == 'text' and 'id' in item and 'name' in item:
                            tool_calls.append({
                                'id': item['id'],
                                'type': 'function',
                                'function': {
                                    'name': item['name'],
                                    'arguments': json.dumps(item.get('input', {}))
                                }
                            })
                        elif item.get('type') == 'text':
                            text_content += item.get('text', '')

                    assistant_msg = {'role': 'assistant', 'content': text_content}
                    if tool_calls:
                        assistant_msg['tool_calls'] = tool_calls
                    messages.append(assistant_msg)

                # Handle user messages with tool results
                elif role == 'user' and isinstance(msg, list):
                    for item in msg:
                        if 'tool_use_id' in item:
                            # This is a tool result
                            messages.append({
                                'role': 'tool',
                                'tool_call_id': item['tool_use_id'],
                                'content': item['content']
                            })
                        elif item.get('type') == 'text':
                            messages.append({'role': 'user', 'content': item.get('content', '')})
                else:
                    messages.append({'role': role, 'content': msg})

            # Handle reasoning effort for o\d models
            is_reasoning_model = model.startswith('o') and len(model) > 1 and model[1].isdigit()

            request_params = {
                'model': model,
                'messages': messages,
                'max_completion_tokens': max_tokens,
            }

            # Only add temperature for non-o\d models
            if not is_reasoning_model and 'search' not in model:
                request_params['temperature'] = temperature

            # Handle reasoning effort for o\d models
            if is_reasoning_model and reasoning_effort:
                request_params['reasoning_effort'] = reasoning_effort.lower()

            # Add tools if provided
            if tools:
                logger.debug(f"Processing {len(tools)} tools for OpenAI")
                openai_tools = []
                for tool in tools:
                    logger.debug(f"Converting tool: {tool.get('name', 'Unknown')}")

                    # Handle MCP client format with nested definition
                    if 'definition' in tool and 'function' in tool['definition']:
                        func_def = tool['definition']['function']
                        openai_tools.append({
                            'type': 'function',
                            'function': {
                                'name': func_def['name'],
                                'description': func_def['description'],
                                'parameters': func_def['parameters']
                            }
                        })
                    # Handle direct tool format (name, description, parameters at top level)
                    elif 'name' in tool and 'description' in tool:
                        openai_tools.append({
                            'type': 'function',
                            'function': {
                                'name': tool['name'],
                                'description': tool['description'],
                                'parameters': tool.get('parameters', {'type': 'object', 'properties': {}})
                            }
                        })

                if openai_tools:
                    request_params['tools'] = openai_tools
                    logger.debug(f"Added {len(openai_tools)} tools to request")

            logger.debug(f'Sending request to OpenAI with params keys: {list(request_params.keys())}')

            # Send request to OpenAI
            try:
                response = await self.client.chat.completions.create(**request_params)
            except Exception as e:
                if "'messages[0].role' does not support 'system' with this model" in str(e):
                    try:
                        logger.debug('Retrying request with developer role as system message.')
                        request_params['messages'][0]['role'] = 'developer'
                        response = await self.client.chat.completions.create(**request_params)
                    except Exception as e:
                        if "'messages[0].role' does not support 'developer' with this model" in str(e):
                            try:
                                logger.debug('Retrying request with system message prepended to first user message.')
                                request_params['messages'][0]['role'] = 'user'
                                request_params['messages'][0]['content'] += '\n\n' + messages[1]['content']
                                request_params['messages'].pop(1)
                                response = await self.client.chat.completions.create(**request_params)
                            except Exception as e:
                                raise ProviderError(f'Request failed: {e}') from e
                        else:
                            raise ProviderError(f'Request failed: {e}') from e
                else:
                    raise ProviderError(f'Request failed: {e}') from e

            # Process the response
            if not response.choices:
                raise ResponseFormatError('No choices found in OpenAI response.')

            choice = response.choices[0]
            message = choice.message
            finish_reason = choice.finish_reason

            # Calculate token usage
            usage = response.usage
            total_tokens = usage.total_tokens if usage else 0

            # Check for tool calls
            tool_calls = message.tool_calls
            if tool_calls:
                # Process tool calls
                processed_tool_calls = []
                for tool_call in tool_calls:
                    processed_tool_calls.append({
                        'type': 'text',
                        'id': tool_call.id,
                        'name': tool_call.function.name,
                        'input': json.loads(tool_call.function.arguments) if tool_call.function.arguments else {}
                    })

                # Return tool information for parent to handle
                tool_info = {
                    'tool_calls': processed_tool_calls,
                    'thinking_blocks': [],  # OpenAI doesn't expose thinking blocks like Anthropic
                    'text_blocks': [{'type': 'text', 'text': message.content or ''}],
                    'stop_reason': finish_reason
                }

                return message.content or '', total_tokens, tool_info

            # Handle unexpected finish reasons
            if finish_reason and finish_reason not in ['stop', 'length']:
                raise ResponseFormatError(f'The OpenAI API returned an unexpected finish reason: {finish_reason}.')

            # Extract text content
            text_content = message.content or ''

            if not text_content:
                raise ResponseFormatError('No text content found in the OpenAI response.')

            return text_content, total_tokens, None

        except Exception:
            raise

    async def list_models(self) -> str:
        """Get available OpenAI models and return formatted response.

        :returns: Formatted string with model information
        """
        try:
            # Get models from API
            models = await self.client.models.list()

            # Filter for chat models (exclude embeddings, audio, etc.)
            chat_models = [
                model for model in models.data
                if any(prefix in model.id for prefix in ['gpt-', 'o1-', 'o3-', 'o4-'])
                and not any(exclude in model.id for exclude in ['realtime', 'audio', 'transcribe', 'tts', 'image', 'pro', 'deep-research'])
            ]

            if not chat_models:
                return 'No chat models available.'

            # Sort models by ID for consistent output
            chat_models.sort(key=lambda x: x.id)

            # Calculate column widths
            id_width = max(len(model.id) for model in chat_models) + 2
            id_width = max(id_width, len('MODEL ID') + 2)
            owner_width = max(len(model.owned_by) for model in chat_models) + 2
            owner_width = max(owner_width, len('OWNER') + 2)

            # Create output lines
            lines = []

            # Print header
            lines.append('\n=== AVAILABLE OPENAI CHAT MODELS ===\n\n')

            # Print header row
            header = f"{'MODEL ID':<{id_width}} | {'OWNER':<{owner_width}} | {'CREATED'}"
            lines.append(header + '\n')
            lines.append('-' * len(header) + '\n')

            # Print each model's information
            for model in chat_models:
                from datetime import datetime
                created_date = datetime.fromtimestamp(model.created).strftime('%B %d, %Y')
                lines.append(
                    f'{model.id:<{id_width}} | {model.owned_by:<{owner_width}} | {created_date}\n'
                )

            # Print footer
            lines.append('\n' + '-' * len(header) + '\n')
            lines.append(f'Total chat models shown: {len(chat_models)}\n')
            lines.append('Note: Only chat-compatible models are shown (gpt-, o1-, o3-, o4- prefixes).\n')

            return ''.join(lines)

        except Exception as e:
            raise ProviderError(f'Error retrieving models: {str(e)}')