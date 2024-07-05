#!/usr/bin/env python3
"""
Chatbot script to interact with OpenAI's API from the terminal.
"""

import argparse
import logging
import os
import sys

from openai import OpenAI

from chat import Chatbot, Speaker, ChatCompletionError, CreateAudioError, SaveAudioError


def setup_logging():
    logging.basicConfig(encoding="utf-8", format="%(message)s", level=logging.INFO)
    logger = logging.getLogger(__name__)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    return logger

logger = setup_logging()

def get_api_key():
    api_key = os.environ.get('OPENAI_API_KEY')
    if not api_key:
        logger.error('OPENAI_API_KEY environment variable not set.')
        sys.exit(1)
    return api_key

client = OpenAI(api_key=get_api_key())

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
    parser.add_argument("--available-models", action='store_true', help="List available OpenAI models.")
    parser.add_argument("-t", "--temperature", type=str_to_float, help="Set the temperature for the model.")
    return parser.parse_args()


def main():
    args = parse_arguments()
    home_directory = os.path.expanduser("~")
    config_path = os.path.join(home_directory, ".chatconfig.json")
    chatbot = Chatbot(config_path, client)

    if args.available_models:
        print(chatbot.get_openai_model_list())
        sys.exit(0)
    if args.available_models_gpt:
        print(chatbot.get_chatgpt_model_list())
        sys.exit(0)

    chatbot.system_message = args.system or chatbot.system_message
    chatbot.model = args.model or chatbot.model
    chatbot.temperature = args.temperature if args.temperature is not None else chatbot.temperature

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
        print(
            "New conversation started.\n"
            "Continue chatting with: ch -c -m \"Your message.\"\n"
            "Current settings:\n"
            f"\tmodel: {chatbot.model:<10}\n"
            f"\ttemperature: {chatbot.temperature}\n"
            f"\tsystem message: {chatbot.system_message}"
        )
        sys.exit(0)
    elif any([args.model, args.system, args.temperature]):
        print(
            "Settings updated.\n"
            "Current settings:\n"
            f"\tmodel: {chatbot.model:<10}\n"
            f"\ttemperature: {chatbot.temperature}\n"
            f"\tsystem message: {chatbot.system_message}"
        )
        sys.exit(0)
    elif args.temperature is not None:
        print(
            "Settings updated.\n"
            "Current settings:\n"
            f"\tmodel: {chatbot.model:<10}\n"
            f"\ttemperature: {chatbot.temperature}\n"
            f"\tsystem message: {chatbot.system_message}"
        )
        sys.exit(0)
    else:
        print('Expected some arguments. Usage: ch -m "Your message"')
        sys.exit(1)

    logger.info(
        f"Chat\nModel: {chatbot.model}, Temperature: {chatbot.temperature}\n"
        f"System: {chatbot.system_message}\nMessage: {chatbot.user_messages[-1]}\n"
    )
    try:
        content = chatbot.chat(chatbot.system_message, chatbot.user_messages, chatbot.model, chatbot.temperature)
    except ChatCompletionError as e:
        logging.error(f'Could not finish chat completion: {e}')
        if args.conversation:
            chatbot.append_user_message('There was an error requesting the API. Please try again.')
        else:
            chatbot.user_messages = []
        sys.exit(1)
    except Exception as e:
        logging.error(f'An unexpected error occurred: {e}. Please check specified arguments and settings.')
        if args.conversation:
            chatbot.append_user_message('There was an error requesting the API. Please try again.')
        else:
            chatbot.user_messages = []
        sys.exit(1)


    if args.save_audio:
        try:
            Speaker(client).create_audio(text=content, audio_save=True, audio_file_path=args.save_audio)
        except CreateAudioError as e:
            logging.error(f'Could not create audio: {e}')
            sys.exit(1)
        except SaveAudioError as e:
            logging.error(f'Could not save audio: {e}')
            sys.exit(1)
        except Exception as e:
            logging.error(f"An unexpected error occurred: {e}")
            sys.exit(1)

    if args.conversation:
        chatbot.append_user_message(content)
    else:
        chatbot.user_messages = []

if __name__ == "__main__":
    main()