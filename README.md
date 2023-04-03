# chat-in-a-nutshell

This script provides a simple way to chat with OpenAI. It allows you to communicate with the `gpt-3.5-turbo` model directly from your terminal.

## Example 
### Input

```
$ chat -m 'Write a Python function to greet the person reading this text.'
```

### Output 
```
Making a request to OPENAI with arguments:
Namespace(system='You are a skilled Python programmer who writes tersely.', messages=['Write a Python function to greet the person reading this text.'])
Sure, here you go!
```
```python
def greet():
    print("Hello there!")
```
```
This code defines a function called `greet()` that, when called, will print the greeting "Hello there!" to the console.
(83 tokens used.)
```

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

## Contribution

Feel free to leave suggestions in the form of GitHub issues. Thank you!