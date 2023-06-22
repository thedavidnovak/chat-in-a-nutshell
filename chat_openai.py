#!/usr/bin/env python3
import os
import openai
import logging
import argparse
import sys
import re
import json


class Chatbot:
    def __init__(self, config_file):
        """Initializes the Chatbot object."""
        self.config_file = config_file
        try:
            self.load_config()
        except FileNotFoundError:
            self.config = {}
            self.save_config()
        self.openai_api_key = os.environ["OPENAI_API_KEY"]
        openai.api_key = self.openai_api_key

    def load_config(self):
        """Loads the configuration from the configuration file."""
        with open(self.config_file, "r") as f:
            self.config = json.load(f)

    def save_config(self):
        """Saves the configuration to the configuration file."""
        with open(self.config_file, "w") as f:
            json.dump(self.config, f)

    def chat_with_openai(self, system, user_assistant, model, temperature):
        """
        Sends a message to OpenAI chatbot API and retrieves the response.

        Args:
            system: A string representing the initial message from the system.
            user_assistant: A list of strings representing messages alternating between the user and assistant roles.
            model: A string representing the chosen model.
            temperature: A float representing the chosen temperature.

        Returns:
            A tuple containing a string representing the response message from the chatbot and an integer representing the total tokens used.

        Raises:
            ValueError: If the inputs are not of the correct type.
            RuntimeError: If there is an error while calling the OpenAI API.
        """
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
            response = openai.ChatCompletion.create(
                model=model, messages=msgs, temperature=temperature
            )
            status_code = response["choices"][0]["finish_reason"]
            if status_code != "stop":
                raise RuntimeError(f"The status code was {status_code}.")
            return (
                response["choices"][0]["message"]["content"],
                response["usage"]["total_tokens"],
            )
        except Exception as e:
            raise RuntimeError("Error while calling OpenAI API") from e

    def chat(self, system, user_assistant, model, temperature):
        """Sends a message to the chatbot and retrieves the response."""
        content, tokens = self.chat_with_openai(
            system, user_assistant, model, temperature
        )
        print(content)
        logging.info(f"({tokens} tokens used.)")
        return content

    def set_system_message(self, message):
        """Sets the system message in the configuration."""
        self.config["system_message"] = message
        self.save_config()

    def get_system_message(self):
        """Gets the system message from the configuration."""
        return self.config.get(
            "system_message", "You are a skilled Python programmer who writes tersely."
        )

    def set_model(self, model):
        """Sets the model in the configuration."""
        self.config["model"] = model
        self.save_config()

    def get_model(self):
        """Gets the model from the configuration."""
        return self.config.get("model", "gpt-3.5-turbo")

    def set_temperature(self, temperature):
        """Sets the temperature in the configuration."""
        self.config["temperature"] = temperature
        self.save_config()

    def get_temperature(self):
        """Gets the temperature from the configuration."""
        return self.config.get("temperature", 0.0)

    def set_user_messages(self, messages):
        """Sets the user messages in the configuration."""
        self.config["user_messages"] = messages
        self.save_config()

    def append_user_messages(self, messages):
        """Appends user messages to the existing user messages in the configuration."""
        self.config["user_messages"].append(messages)
        self.save_config()

    def get_user_messages(self):
        """Gets the user messages from the configuration."""
        return self.config.get("user_messages", [""])


def get_chatgpt_model_list():
    """Gets the list of available ChatGPT models."""
    models = openai.Model.list()
    available_models = []
    for model in models["data"]:
        if model["id"].startswith("gpt"):
            available_models.append(model["id"])
    return "Currently available ChatGPT models: {}. To check all available models refer to the OpenAI documentation.".format(
        ", ".join(available_models)
    )


def parse_arguments():
    parser = argparse.ArgumentParser(
        prog="chat", description="Chat from the comfort of your terminal"
    )
    parser.add_argument(
        "-s",
        "--system",
        type=str,
        default=None,
        help="Set the system message for the chatbot.\nFor example: You are a skilled Python programmer who writes tersely.",
    )
    parser.add_argument(
        "-m",
        "--messages",
        nargs="+",
        type=str,
        default=None,
        help="Write a message for the chatbot. Or write multiple messages with every other message being chatbot message, thus setting example for its future answers in conversation.",
    )
    parser.add_argument(
        "-c",
        "--conversation",
        action="store_true",
        help="On/off flag indicating whether a conversation is intended and messages will not be overwritten.",
    )
    parser.add_argument(
        "--model", type=str, default=None, help=get_chatgpt_model_list()
    )
    parser.add_argument(
        "-t",
        "--temperature",
        type=float,
        default=None,
        help="Set the temperature for the model. The default value for this script is 0.",
    )
    args = parser.parse_args()
    return args


def main():
    logging.basicConfig(encoding="utf-8", format="%(message)s", level=logging.INFO)
    args = parse_arguments()

    # Create a Chatbot instance with the configuration file
    chatbot = Chatbot(".config.json")

    # Set system message if provided
    if args.system:
        chatbot.set_system_message(args.system)

    # Set model if provided
    if args.model:
        chatbot.set_model(args.model)

    # Set temperature if provided
    if args.temperature:
        chatbot.set_temperature(args.temperature)

    # Set user messages if provided
    if args.conversation and args.messages:
        try:
            for m in args.messages:
                chatbot.append_user_messages(m)
        except KeyError:
            chatbot.set_user_messages(args.messages)
    elif args.messages:
        chatbot.set_user_messages(args.messages)
    elif args.conversation:
        chatbot.set_user_messages([""])

    # Get system message, user messages, model, and temperature
    system = chatbot.get_system_message()
    user_assistant = chatbot.get_user_messages()
    model = chatbot.get_model()
    temperature = chatbot.get_temperature()

    logging.info(
        "Chat\nModel: {model}, temperature: {temperature}\nSystem: {system}\nMessage: {user_assistant}\n".format(
            model=model,
            temperature=temperature,
            system=system,
            user_assistant=user_assistant[-1],
        )
    )
    content = chatbot.chat(system, user_assistant, model, temperature)

    # Append chat response to user messages if in conversation mode
    if args.conversation:
        chatbot.append_user_messages(content)


if __name__ == "__main__":
    main()
