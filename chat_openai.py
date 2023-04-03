#!/usr/bin/env python3
import os
import openai
import logging
import argparse
import sys
import re

def parse_arguments():
    parser = argparse.ArgumentParser()
    system='You are a skilled Python programmer who writes tersely.'
    user_assistant = ['Write a Python function `hello_world`.']
    parser.add_argument('-s','--system', type=str, default=system)
    parser.add_argument('-m','--messages', nargs='+', type=str, default=user_assistant)
    args = parser.parse_args()
    return args

def chat_with_openai(system, user_assistant):
    """
    Sends a message to OpenAI's chatbot API and retrieves the response.

    Args:
        system: A string representing the initial message from the system.
        user_assistant: A list of strings representing messages alternating between the user and assistant roles.

    Returns:
        A string representing the response message from the chatbot.

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
        {"role": "assistant", "content": user_assistant[i]} if i % 2 else {"role": "user", "content": user_assistant[i]}
        for i in range(len(user_assistant))
    ]
    msgs = system_msg + user_assistant_msgs

    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=msgs
        )
        status_code = response["choices"][0]["finish_reason"]
        if status_code != "stop":
            raise RuntimeError(f"The status code was {status_code}.")
        return response['choices'][0]['message']['content'], response["usage"]["total_tokens"]
    except Exception as e:
        raise RuntimeError("Error while calling OpenAI API") from e

def main():
    logging.basicConfig(encoding='utf-8', format='%(message)s', level=logging.INFO)
    openai.api_key = os.environ["OPENAI_API_KEY"]
    args = parse_arguments()
    logging.info(f"Making a request to OPENAI with arguments:\n{str(args)}")
    content, tokens = chat_with_openai(args.system, args.messages)
    print(content)
    logging.info(f"({tokens} tokens used.)")

if __name__ == "__main__":
    main()