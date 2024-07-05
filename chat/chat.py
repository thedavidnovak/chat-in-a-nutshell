#!/usr/bin/env python3

import json
import logging
from typing import List, Optional, Tuple
from openai import APIConnectionError, RateLimitError, APIStatusError

logging.basicConfig(encoding="utf-8", format="%(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARNING)

class ChatCompletionError(Exception):
    pass

class Chatbot:
    def __init__(self, config_file: str, client: object) -> None:
        """Initialize Chatbot."""
        self.config_file: str = config_file
        self.client: object = client
        try:
            self.load_config()
        except FileNotFoundError:
            self.config: dict = {}
            self.save_config()

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

    def chat_with_openai(self, system: str, user_assistant: List[str], model: str, temperature: float) -> Tuple[str, int]:
        """Chat with OpenAI."""
        if not isinstance(system, str):
            raise ValueError("`system` should be a string")
        if not all(isinstance(msg, str) for msg in user_assistant):
            raise ValueError("`user_assistant` should be a list of strings")

        system_msg = [{"role": "system", "content": system}]
        user_assistant_msgs = [
            {"role": "assistant", "content": user_assistant[i]}
            if i % 2
            else {"role": "user", "content": user_assistant[i]}
            for i in range(len(user_assistant))
        ]
        msgs = system_msg + user_assistant_msgs

        try:
            response = self.client.chat.completions.create(model=model, messages=msgs, temperature=temperature)
            response_dict = dict(response)
            choice = response_dict['choices'][0]
            if choice.finish_reason != "stop":
                raise RuntimeError(f"The status code was {choice.finish_reason}.")
            return choice.message.content, response_dict['usage'].total_tokens

        except APIConnectionError as e:
            logging.error(f'The server could not be reached: {e.__cause__}')
            raise ChatCompletionError('The server could not be reached.') from e
        except RateLimitError as e:
            logging.error(f'A 429 status code recieved (RateLimitError): {e}.')
            raise ChatCompletionError('A 429 status code recieved. Check your usage and limit configuration.') from e
        except APIStatusError as e:
            logging.error(f'APIStatusError: {e}')
            raise ChatCompletionError('Another non-200-range status code was received.') from e
        except Exception as e:
            logging.error(f'Error while calling OpenAI API: {e}')
            raise ChatCompletionError('Another non-200-range status code was received.') from e

    def chat(self, system: str, user_assistant: List[str], model: str, temperature: float) -> str:
        """Chat."""
        content, tokens = self.chat_with_openai(system, user_assistant, model, temperature)
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

    def get_model_list(self, filter_prefix: Optional[str] = None) -> str:
        """Get models."""
        models = self.client.models.list()
        sorted_models = [d.id for d in sorted(models, key=lambda x: x.created, reverse=True)]
        if filter_prefix:
            sorted_models = [model for model in sorted_models if model.startswith(filter_prefix)]
        return "Currently available models:\n{}.\nTo check all available models refer to the OpenAI documentation.".format(
            ", ".join(sorted_models)
        )

    def get_chatgpt_model_list(self) -> str:
        """Get ChatGPT models."""
        return self.get_model_list(filter_prefix="gpt")

    def get_openai_model_list(self) -> str:
        """Get OpenAI models."""
        return self.get_model_list()