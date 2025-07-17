# chat-in-a-nutshell

This script provides a simple way to chat with AI models from OpenAI and Anthropic directly from your terminal. It lets you send messages and receive responses without leaving the command line interface.

## Example 
### Input

```
$ ch -m 'Write a Python function to greet the person reading this text.'
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
$ ch -m 'Write a Python function `hello_world`.'
```
### -s

The default `system` message is: 'You are a skilled Python programmer who writes tersely.' You can change it by specifying the `--system, -s` argument inside the current command:
```
$ ch -s 'You are a remarkable Italian teacher.' -m 'Translate "Hello there!"'
```
From the version `v1.1.0` the recent system message becomes the default one until further change (or configuration file deletion).

### -c

On/off flag indicating whether a conversation is intended and messages will be appended to the list of messages in the configuration file. Use `--conversation, -c` argument inside the current command.

### --model

You may specify the model that will be used for the API requests (until further change).

### -t

The temperature may be set using the `--temperature, -t` argument. The default value for this script is 0 (lowest).

### --save-audio

Specify the path to save the audio response using the `--save-audio` argument.

### --available-models

List available provider models using the `--available-models` argument.
Not all models are suitable to use with this script.

### --available-models-gpt

List available ChatGPT models using the `--available-models-gpt` argument.
Not all models are suitable to use with this script.

### --available-tools

List available tools for function calling using the `--available-tools` argument.

### --use-tools, --no-tools

Specify `--use-tools` to enable the model the access to user-defined tools. Specify `--no-tools` to disable it. The option is saved for future requests. Tools are disabled by default. **Note:** To use tools, you must set the `TOOLS_URL` environment variable to the URL path of your tool definitions. You can refer to `example_tools.json` in the repository to see the example tool definition file structure.

### -e, --reasoning-effort
Set the reasoning effort by specifying `--reasoning-effort, -e`. Choose one of the available options: low, medium, high. The option is saved for future requests. The default value for this script is low reasoning effort. This configuration is only applicable to reasoning models.

### -p, --provider
Set the provider by specifying `--provider, -p`. Choose one of the available options: `openai`, `anthropic`. The option is saved for future requests. The default value for this script is `openai`.
Example usage:
```
$ ch -p anthropic --model claude-3-7-sonnet-latest
```
```
$ ch -p openai --model gpt-4o
```

## Requirements

To use this script, you will need to have the following:
- Python 3.8 or higher
- A provider API key (stored in an environment variable `OPENAI_API_KEY` for provider `openai`, `ANTHROPIC_API_KEY` for `anthropic`)


## Usage

To use the script:
1. Add your API key to the `OPENAI_API_KEY` environment variable (`ANTHROPIC_API_KEY` for Anthropic).
2. Install the package using: `pip install chat-in-a-nutshell`

### Debug Mode

To see detailed error logs and tracebacks, set the `CHAT_LOG_LEVEL` environment variable to `DEBUG`:

```bash
export CHAT_LOG_LEVEL=DEBUG
ch -m "Your message"
```


## Limitations

This script was developed for chatting with OpenAI's ChatGPT from the comfort of the terminal. The API used in the script incurs charges (depending on its usage). For more information about pricing, refer to the provider documentation.
Please note that there is no warranty for script functionality. Also, keep in mind that it is suitable for brief and straightforward messages.

## Release Notes

### Version 1.6.0 (2025-07-17)
- Documentation for this release will be updated in the next version.

### Version 1.5.1 (2025-03-02)
- Added `--max-tokens` parameter to limit token generation (default: 4096)
- Enhanced error handling with formatted error messages
- Improved provider model listings with tabular display
- Renamed main script to `chat_provider.py` for clarity

### Version 1.5.0 (2025-03-02)
- **Anthropic Provider Integration**:
  - Added support for Anthropic as a provider.
  - Select provider with `-p` or `--provider` flag.
  - Requires `ANTHROPIC_API_KEY` environment variable for authentication.
  - Required parameter `max_tokens` is set to 8,000 tokens (the absolute maximum number of tokens to generate)
- **Dependencies**: Added `anthropic~=0.49.0`.

### Version 1.4.0 (2025-02-24)
- **OpenAI requests refactoring**:
  - Added back system messages for `o1` and `o3` models, now supported as developer messages.
  - Introduced an option to specify reasoning effort for reasoning models.
  - Improved readability of `--available-models` and `--available-models-gpt` outputs.

- **Tool calling**:
  - The feature can be toggled with `--use-tools` to enable and `--no-tools` to disable.
  - To use your own tools, specify the URL path using the `TOOLS_URL` environment variable. Check `example_tools.json` in the repo for the example tool definition file structure.
  - Review available tools with `--available-tools`.

### Version 1.3.0 (2024-11-17)
- **o1 model support**:
  - Removed system message from requests to o1 models because they are currently not supported.
  - Set temperature to 1 for o1 models; other values are not currently supported.

- **Added CI/CD Pipeline**:
  - Added GitHub Actions pipeline for deploying to PyPI.

### Version 1.2.0 (2024-07-05)
- Migrated to OpenAI v1
- Published as PyPi package (https://pypi.org/project/chat-in-a-nutshell/)
- Updated executable name from `chat` to `ch` to avoid conflicts with other programs.
- Configuration file containing system message, temperature, model and conversation if in conversation mode is now saved in the home directory (`~/.chatconfig.json`).
- Added the option to save the response as audio file to a specified path.
- Added the option to list available OpenAI and ChatGPT models.

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