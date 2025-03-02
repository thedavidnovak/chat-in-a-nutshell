#!/usr/bin/env python3

import httpx
import importlib
import json
import logging
import os
import re
from typing import List, Optional, Tuple
from openai import OpenAI, APIConnectionError, RateLimitError, APIStatusError
from anthropic import Anthropic, NotFoundError

logger = logging.getLogger(__name__)

TOOLS_URL = os.getenv('TOOLS_URL', None)

class ChatCompletionError(Exception):
    pass

class Chatbot:
    def __init__(self, config_file: str, provider: str = None) -> None:
        """Initialize Chatbot."""
        self.config_file = config_file
        try:
            self.load_config()
        except FileNotFoundError:
            self.config = {}
            self.save_config()
        self.provider = provider or self.provider
        self.client = self.create_client()

    def get_api_key(self, provider_key):
        api_key = os.environ.get(provider_key)
        if not api_key:
            logger.error(f'{provider_key} environment variable not set.')
            sys.exit(1)
        return api_key

    def load_tools_metadata(self) -> None:
        """Load tools metadata."""
        if not TOOLS_URL:
            logger.warning("TOOLS_URL not set. Provide a URL to the raw contents, e.g., raw path to example_tools.json.")
            self.tools = []
            return
        try:
            response = httpx.get(TOOLS_URL)
            response.raise_for_status()
            self.tools = response.json()
        except (httpx.RequestError, httpx.HTTPStatusError) as e:
            logger.error(f"Failed to load tools: {e}")
            self.tools = []
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            self.tools = []

    def load_config(self) -> None:
        """Load config."""
        try:
            with open(self.config_file, "r") as f:
                self.config = json.load(f)
        except FileNotFoundError:
            self.config = {}

    def save_config(self) -> None:
        """Save config."""
        with open(self.config_file, "w") as f:
            json.dump(self.config, f, indent=4)

    def create_client(self) -> object:
        if self.provider == 'openai':
            api_key = self.get_api_key('OPENAI_API_KEY')
            return OpenAI(api_key=api_key)
        elif self.provider == 'anthropic':
            api_key = self.get_api_key('ANTHROPIC_API_KEY')
            return Anthropic(api_key=api_key)
        else:
            logger.error("Unknown provider selected: %s", self.provider)
            sys.exit(1)

    def chat_with_provider(self, system: str, user_assistant: List[str], model: str, temperature: float) -> Tuple[str, int]:
        """Chat with OpenAI."""
        self.validate_inputs(system, user_assistant)
        messages = self.construct_messages(system, user_assistant, model)

        try:
            response = self.send_request(model, messages, temperature)
            if self.provider == 'anthropic':
                return self.handle_anthropic_response(response)
            return self.handle_openai_response(response, messages, model)
        except (APIConnectionError, RateLimitError, APIStatusError, NotFoundError) as e:
            self.handle_api_exception(e)
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            raise ChatCompletionError("An unexpected error occurred.") from e

    def validate_inputs(self, system: str, user_assistant: List[str]) -> None:
        """Validate input types."""
        if not isinstance(system, str):
            raise ValueError("`system` should be a string")
        if not all(isinstance(msg, str) for msg in user_assistant):
            raise ValueError("`user_assistant` should be a list of strings")

    def construct_messages(self, system: str, user_assistant: List[str], model: str) -> List[dict]:
        """Construct the message list for OpenAI API."""
        if re.match(r'^o\d', model):
            system_msg = [{"role": "developer", "content": system}]
        else:
            system_msg = [{"role": "system", "content": system}]
        user_assistant_msgs = [
            {"role": "assistant", "content": msg} if i % 2 else {"role": "user", "content": msg}
            for i, msg in enumerate(user_assistant)
        ]

        return system_msg + user_assistant_msgs

    def send_request(self, model: str, messages: List[dict], temperature: float) -> object:
        """Send the chat completion request to OpenAI."""

        request_params = {
            "model": model,
            "messages": messages,
        }

        if self.provider == 'anthropic':
            request_params['messages'] = [msg for msg in messages if msg["role"] not in {"system", "developer"}]
            request_params['max_tokens'] = 8000
            request_params['system'] = self.system_message
            return self.client.messages.create(**request_params)

        if self.use_tools and self.tools:
            request_params["tools"] = [tool['description'] for tool in self.tools if 'description' in tool]
            request_params["parallel_tool_calls"] = False

        if re.match(r'^o\d', model):
            # For o\d models, use fixed temperature and include reasoning_effort, exclude parallel_tool_calls.
            request_params["temperature"] = 1.0
            request_params["reasoning_effort"] = self.reasoning_effort
            request_params.pop("parallel_tool_calls", None)
        else:
            request_params["temperature"] = temperature

        logger.debug(f"Sending request to OpenAI with model: {model}, temperature: {request_params['temperature']}")

        logger.debug(f"Model: {model}, Tools included: {'tools' in request_params}")
        return self.client.chat.completions.create(**request_params)

    def handle_openai_response(self, response, messages: List[dict], model: str) -> Tuple[str, int]:
        """Process the OpenAI API response."""
        response_dict = dict(response)
        choice = response_dict['choices'][0]

        if choice.finish_reason not in ["stop", "tool_calls"]:
            raise RuntimeError(f"The finish reason was {choice.finish_reason}.")

        if choice.finish_reason == 'tool_calls':
            return self.handle_tool_call(response, messages, model)

        return choice.message.content, response_dict['usage'].total_tokens

    def handle_anthropic_response(self, response) -> Tuple[str, int]:
        """Process the response from Anthropic."""
        resp = dict(response)

        # Check stop reason
        if resp.get("stop_reason") != "end_turn":
            raise RuntimeError(f"Unexpected stop reason: {resp.get('stop_reason')}")

        contents = resp.get("content", [])
        text = None
        for item in contents:
            if isinstance(item, dict) and item.get("type") == "text":
                text = item.get("text")
                break
            elif getattr(item, "type", None) == "text":
                text = getattr(item, "text", None)
                break
        if text is None:
            raise RuntimeError("No text content found in the response.")

        # Calculate total token
        usage = dict(resp.get("usage", {}))
        total_tokens = usage.get("input_tokens", 0) + usage.get("output_tokens", 0)
        return text, total_tokens

    def handle_tool_call(self, response, messages: List[dict], model: str) -> Tuple[str, int]:
        """Handle tool call responses from OpenAI."""
        tool_call = response.choices[0].message.tool_calls[0]
        tool_call_dict = dict(tool_call)

        self.log_tool_call(tool_call_dict)

        if not self.confirm_tool_execution():
            return self.abort_tool_execution(response, messages, tool_call_dict, model)

        result = self.execute_tool(tool_call_dict)
        return self.complete_tool_execution(response, messages, tool_call_dict, result, model)

    def log_tool_call(self, tool_call_dict: dict) -> None:
        """Log the details of the tool call."""
        func = tool_call_dict['function']
        func_def = next(
            (tool['definition'] for tool in self.tools if tool['name'] == func.name),
            ''
        )
        truncated_def = func_def[:50] + ('...' if len(func_def) > 50 else '')
        logger.info(
            "I would like to execute the following function call:\n"
            f"  ID        : {tool_call_dict['id']}\n"
            f"  Function  : {func.name}\n"
            f"  Arguments : {func.arguments}\n"
            f"  Type      : {tool_call_dict['type']}\n"
            f"  Definition: (see truncated definition below, use ch --available-tools to list all available tools)\n\n{truncated_def}\n"
        )

    def confirm_tool_execution(self) -> bool:
        """Ask the user to confirm tool execution."""
        response = input('Do you agree to proceed? (y/n): ').strip().lower()
        print()
        return response == 'y'

    def abort_tool_execution(self, response, messages: List[dict], tool_call_dict: dict, model: str) -> Tuple[str, int]:
        """Handle the scenario where tool execution is aborted by the user."""
        self.append_user_message(str(dict(response.choices[0].message)))
        user_response = [
            {"role": "tool", "tool_call_id": tool_call_dict['id'], "content": 'The tool call was not successful. Please try again or ask for instructions!'}
        ]
        self.append_user_message(str(user_response))
        new_messages = messages + [dict(response.choices[0].message)] + user_response
        not_ok_response = self.client.chat.completions.create(model=model, messages=new_messages)
        not_ok_dict = dict(not_ok_response)
        not_ok_choice = not_ok_dict['choices'][0]
        return not_ok_choice.message.content, not_ok_dict['usage'].total_tokens

    def execute_tool(self, tool_call_dict: dict) -> dict:
        """Execute the tool based on the tool call dictionary."""
        tool = dict(tool_call_dict['function'])
        tool_name = tool['name']
        tool_args = json.loads(tool['arguments'])
        logger.debug(f"Executing tool '{tool_name}' with arguments: {tool_args}")
        return self.call_tool(tool_name, **tool_args)

    def complete_tool_execution(self, response, messages: List[dict], tool_call_dict: dict, result: dict, model: str) -> Tuple[str, int]:
        """Handle the completion of a tool execution."""
        function_call_result_message = [
            {"role": "tool", "tool_call_id": tool_call_dict['id'], "content": json.dumps(result)}
        ]
        self.append_user_message(str(dict(response.choices[0].message)))
        self.append_user_message(str(function_call_result_message))
        new_messages = messages + [dict(response.choices[0].message)] + function_call_result_message
        complete_response = self.client.chat.completions.create(model=model, messages=new_messages)
        complete_dict = dict(complete_response)
        complete_choice = complete_dict['choices'][0]
        return complete_choice.message.content, complete_dict['usage'].total_tokens

    def handle_api_exception(self, exception: Exception) -> None:
        """Handle API exceptions by logging and raising a ChatCompletionError."""
        if isinstance(exception, APIConnectionError):
            logger.error(f'The server could not be reached: {exception.__cause__}')
            raise ChatCompletionError('The server could not be reached.') from exception
        elif isinstance(exception, RateLimitError):
            logger.error(f'A 429 status code received (RateLimitError): {exception}.')
            raise ChatCompletionError('A 429 status code received. Check your usage and limit configuration.') from exception
        elif isinstance(exception, APIStatusError):
            logger.error(f'APIStatusError: {exception}')
            raise ChatCompletionError('A non-200 status code was received from the API.') from exception
        elif isinstance(exception, NotFoundError):
            logger.error(f'NotFoundError: {exception}')
            raise ChatCompletionError('Did you set an existing model for the current provider?') from exception

    def chat(self, system: str, user_assistant: List[str], model: str, temperature: float) -> str:
        """Chat."""
        content, tokens = self.chat_with_provider(system, user_assistant, model, temperature)
        print(content)
        logger.info(f"({tokens} tokens used.)")
        return content

    @property
    def system_message(self) -> str:
        """Get system message."""
        return self.config.get("system_message", "You are a skilled Python programmer who writes tersely.")

    @system_message.setter
    def system_message(self, message: str) -> None:
        """Set system message."""
        self.config["system_message"] = message
        self.save_config()

    @property
    def model(self) -> str:
        """Get model."""
        return self.config.get("model", "gpt-3.5-turbo")

    @model.setter
    def model(self, model: str) -> None:
        """Set model."""
        self.config["model"] = model
        self.save_config()

    @property
    def temperature(self) -> float:
        """Get temperature."""
        return self.config.get("temperature", 0.0)

    @temperature.setter
    def temperature(self, temperature: float) -> None:
        """Set temperature."""
        self.config["temperature"] = temperature
        self.save_config()

    @property
    def reasoning_effort(self) -> str:
        """Get reasoning effort; default to 'low'."""
        return self.config.get("reasoning_effort", "low")

    @reasoning_effort.setter
    def reasoning_effort(self, effort: str) -> None:
        """Set reasoning effort."""
        self.config["reasoning_effort"] = effort
        self.save_config()

    @property
    def provider(self) -> str:
        """Get provider; default to 'openai'."""
        return self.config.get("provider", "openai")

    @provider.setter
    def provider(self, provider: str) -> None:
        """Set provider."""
        self.config["provider"] = provider
        self.save_config()

    @property
    def user_messages(self) -> List[str]:
        """Get user messages."""
        return self.config.get("user_messages", [""])

    @user_messages.setter
    def user_messages(self, messages: List[str]) -> None:
        """Set user messages."""
        self.config["user_messages"] = messages
        self.save_config()

    def append_user_message(self, message: str) -> None:
        """Append user message."""
        if "user_messages" not in self.config:
            self.config["user_messages"] = []
        self.config["user_messages"].append(message)
        self.save_config()

    @property
    def use_tools(self) -> bool:
        """Get tool usage preference."""
        return self.config.get("use_tools", False)

    @use_tools.setter
    def use_tools(self, use: bool) -> None:
        """Set tool usage preference."""
        self.config["use_tools"] = use
        self.save_config()

    def get_model_list(self, filter_prefix: Optional[str] = None) -> str:
        """Get models."""
        models = self.client.models.list()
        sorted_models = [d.id for d in sorted(models, key=lambda x: x.created, reverse=True)]
        if filter_prefix:
            sorted_models = [model for model in sorted_models if model.startswith(filter_prefix)]

        if not sorted_models:
            return "No models available."

        model_list = "\n".join(f"- {model}" for model in sorted_models)
        return (
            "Currently available models:\n"
            f"{model_list}\n"
            "To check all available models, refer to the OpenAI documentation."
        )

    def get_chatgpt_model_list(self) -> str:
        """Get ChatGPT models."""
        return self.get_model_list(filter_prefix="gpt")

    def get_openai_model_list(self) -> str:
        """Get OpenAI models."""
        return self.get_model_list()

    def get_anthropic_model_list(self) -> str:
        """Get OpenAI models."""
        return self.client.models.list(limit=5)

    def get_tool_list(self) -> str:
        """Get list of available tools."""
        if not self.tools:
            return "No tools are currently available."

        tool_list = []
        for tool in self.tools:
            if 'name' in tool and 'description' in tool:
                tool_info = {
                    "name": tool['name'],
                    "description": tool['description'],
                    "definition": tool['definition']
                }
                tool_list.append(json.dumps(tool_info, indent=4))
            else:
                logger.warning(f"Invalid tool entry detected: {tool}")

        if not tool_list:
            return "No valid tools are currently available."

        return "\n\n".join(tool_list)

    def call_tool(self, tool_name: str, *args, **kwargs):
        """Call a tool by name."""
        tool = next((t for t in self.tools if t['name'] == tool_name), None)
        if not tool:
            raise ValueError(f"Tool '{tool_name}' not found.")

        # Execute the function definition
        local_namespace = {}
        exec(tool['definition'], {}, local_namespace)
        func = local_namespace.get(tool_name)
        if not func:
            raise ValueError(f"Function '{tool_name}' could not be executed.")

        return func(*args, **kwargs)