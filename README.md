# chat-in-a-nutshell

This script provides a simple way to chat with OpenAI. It allows you to communicate with the ChatGPT model directly from your terminal.

## Example 
### Input

```
$ chat -m 'Write a Python function to greet the person reading this text.'
```

### Output 
````
Chat
Model: gpt-3.5-turbo, temperature: 0.0
System: You are a skilled Python programmer who writes tersely.
Message: Write a Python function to greet the person reading this text.

Sure, here you go!

```python
def greet():
    print("Hello there!")
```

This code defines a function called `greet()` that, when called, will print the greeting "Hello there!" to the console.
(83 tokens used.)
````

## Output overview
First, there is a recap of the request sent to the API with selected arguments. Next, chat completion API's response is displayed. Output also includes token count (e.g., 83 tokens used) at the end.

## Arguments

### -m

The message is specified using the `--message, -m` argument. 
```
$ chat -m 'Write a Python function `hello_world`.'
```
### -s

The default `system` message is: 'You are a skilled Python programmer who writes tersely.' You can change it by specifying the `--system, -s` argument inside the current command:
```
$ chat -s 'You are a remarkable Italian teacher.' -m 'Translate "Hello there!"'
```
In the version `v1.1.0` the recent system message becomes the default one until further change (or configuration file deletion).

### -c

On/off flag indicating whether a conversation is intended and messages will be appended to the list of messages in the configuration file. Use `--conversation, -c` argument inside the current command.

### --model

You may specify the model that will be used for the API requests (until further change).

### -t

The temperature may be set using the `--temperature, -t` argument. The default value for this script is 0 (lowest).

## Requirements

To use this script, you will need to have the following:
- Python 3.6 or higher
- An OpenAI API key (stored in an environment variable)
- The `openai` Python module installed

## Usage

To use the script:
1. Add your API key to the `OPENAI_API_KEY` environment variable.
2. Check that the shebang `#!/usr/bin/env python3` matches a Python interpreter with the `openai` module installed.
3. Make the script executable (`chmod +x`) and create a symbolic link to the script: `sudo ln -s /path/to/chat_openai.py /usr/bin/chat`


## Limitations

This script was developed for chatting with OpeanAI's ChatGPT from the comfort of the terminal. The API used in the script incurs charges (depending on its usage). For more information about pricing, refer to the OpenAI documentation.
Please note that there is no warranty for script functionality. Also, keep in mind that it is suitable for brief and straightforward messages.

## Release Notes

### Version 1.1.0 (2023-06-22)

- Added the option to make a new system message as default by creating a config file.
- Added the option to select a model (`--help` now prints the list of available `gpt` models)
- Added the option to select a temperature for a model.
- The chatbot is now initialized as a class instance.
- Added a flag `conversation` that enables longer conversation.

### Version 1.0.0 (2023-04-22)

- Initial release of the script

## Contribution

Feel free to leave suggestions in the form of GitHub issues. Thank you!