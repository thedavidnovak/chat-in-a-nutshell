#!/usr/bin/env python3
"""
Chatbot script to interact with OpenAI's API from the terminal.
"""

import argparse
import os
import re
import sys

from chat import Chatbot, Speaker, ChatCompletionError, CreateAudioError, SaveAudioError
from chat.logging_setup import setup_logging


logger = setup_logging()

def str_to_float(value):
    try:
        return float(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"Invalid float value: '{value}'")

def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(prog="chat", description="Chat from the comfort of your terminal")
    parser.add_argument("-s", "--system", type=str, help="Set the system message for the chatbot.")
    parser.add_argument("-m", "--messages", nargs="+", type=str, help="Write a message for the chatbot.")
    parser.add_argument("-c", "--conversation", action="store_true", help="Flag for conversation mode.")
    parser.add_argument("--save-audio", type=str, help="Path to save the audio response.")
    parser.add_argument("--model", type=str, help="Specify the model to use.")
    parser.add_argument("--available-models-gpt", action='store_true', help="List available ChatGPT models.")
    parser.add_argument("--available-models", action='store_true', help="List available models.")
    parser.add_argument("--available-tools", action='store_true', help="List available tools.")
    parser.add_argument("-t", "--temperature", type=str_to_float, help="Set the temperature for the model.")
    parser.add_argument("-e", "--reasoning-effort", choices=["low", "medium", "high"], help="Set the reasoning effort, only applicable to reasoning models.")
    parser.add_argument("--max-tokens", type=int, help="Absolute maximum number of tokens to generate.")
    parser.add_argument("-p", "--provider", choices=["openai", "anthropic"], help="Choose the provider: OpenAI (default) or Anthropic.")
    parser.add_argument("--use-tools", action='store_true', dest='use_tools', help="Enable tool usage.")
    parser.add_argument("--no-tools", action='store_false', dest='use_tools', help="Disable tool usage.")
    parser.set_defaults(use_tools=None)
    return parser.parse_args()


def print_settings(chatbot, header):
    settings = (
        f"{header}\n"
        f"\tprovider: {chatbot.provider}\n"
        f"\tmodel: {chatbot.model:<10}\n"
        f"\ttemperature: {chatbot.temperature}\n"
        f"\tsystem message: {chatbot.system_message}\n"
        f"\treasoning effort: {chatbot.reasoning_effort}\n"
        f"\tmax tokens: {chatbot.max_tokens}\n"
        f"\tuse tools: {chatbot.use_tools}"
    )
    print(settings)

def log_chat_details(chatbot):
    is_reasoning_model = re.search(r'^o\d', chatbot.model)
    details = f"Chat\nModel: {chatbot.model}"
    if is_reasoning_model:
        details += f", Reasoning effort: {chatbot.reasoning_effort}"
    if chatbot.provider == 'openai':
        details += f", Temperature: {chatbot.temperature}\n"
    else:
        details += "\n"
    details += f"System: {chatbot.system_message}\n"
    details += f"Message: {chatbot.user_messages[-1]}\n"
    logger.info(details)

def main():
    args = parse_arguments()

    home_directory = os.path.expanduser("~")
    config_path = os.path.join(home_directory, ".chatconfig.json")
    chatbot = Chatbot(config_path, args.provider)

    if args.available_models:
        if chatbot.provider == 'anthropic':
            print(chatbot.get_anthropic_model_list())
            sys.exit(0)
        print(chatbot.get_openai_model_list())
        sys.exit(0)
    if args.available_models_gpt:
        print(chatbot.get_chatgpt_model_list())
        sys.exit(0)

    chatbot.system_message = args.system or chatbot.system_message
    chatbot.model = args.model or chatbot.model
    chatbot.reasoning_effort = args.reasoning_effort or chatbot.reasoning_effort
    chatbot.max_tokens = args.max_tokens or chatbot.max_tokens
    chatbot.temperature = args.temperature if args.temperature is not None else chatbot.temperature
    chatbot.use_tools = args.use_tools if args.use_tools is not None else chatbot.use_tools

    if chatbot.use_tools:
        chatbot.load_tools_metadata()
    else:
        chatbot.tools = []

    if args.available_tools:
        print(chatbot.get_tool_list())
        sys.exit(0)

    if args.messages:
        if args.conversation:
            for message in args.messages:
                chatbot.append_user_message(message)
        else:
            chatbot.user_messages = args.messages
    elif args.save_audio:
        print('No message specified.\n'
              'Usage: ch -m "Your message." --save-audio path_to_file.aac')
        sys.exit(1)
    elif args.conversation:
        chatbot.user_messages = []
        print("New conversation started.\n"
              "Continue chatting with: ch -c -m \"Your message.\"")
        print_settings(chatbot, "Current settings:")
        sys.exit(0)
    elif any([args.model, args.system, args.temperature is not None, args.use_tools is not None, args.reasoning_effort, args.provider, args.max_tokens]):
        print_settings(chatbot, "Settings updated.")
        sys.exit(0)
    else:
        print('Expected some arguments. Usage: ch -m "Your message"')
        sys.exit(1)

    log_chat_details(chatbot)

    try:
        content = chatbot.chat(chatbot.system_message, chatbot.user_messages, chatbot.model, chatbot.temperature, chatbot.max_tokens)
    except ChatCompletionError as e:
        logger.error(f'Could not finish chat completion:\n{e}')
        if args.conversation:
            chatbot.append_user_message('There was an error requesting the API. Please try again.')
        else:
            chatbot.user_messages = []
        sys.exit(1)
    except Exception as e:
        logger.error(f'An unexpected error occurred:\n{e}\nPlease check specified arguments and settings.')
        if args.conversation:
            chatbot.append_user_message('There was an error requesting the API. Please try again.')
        else:
            chatbot.user_messages = []
        sys.exit(1)


    if args.save_audio:
        try:
            Speaker(client).create_audio(text=content, audio_save=True, audio_file_path=args.save_audio)
        except CreateAudioError as e:
            logger.error(f'Could not create audio: {e}')
            sys.exit(1)
        except SaveAudioError as e:
            logger.error(f'Could not save audio: {e}')
            sys.exit(1)
        except Exception as e:
            logger.error(f"An unexpected error occurred: {e}")
            sys.exit(1)

    if args.conversation:
        chatbot.append_user_message(content)
    else:
        chatbot.user_messages = []

if __name__ == "__main__":
    main()