#!/usr/bin/env python3
"""
Chatbot script to interact with provider's API from the terminal.
"""

import argparse
import os
import readline
import sys
import asyncio
import signal

from chat.chat import Chatbot
from chat.logging_setup import setup_logging
from chat.format import COLORS, print_settings
from chat.error_utils import log_error

logger = setup_logging(__name__)


class CustomArgumentParser(argparse.ArgumentParser):
    """Custom ArgumentParser with better error formatting."""

    def print_usage(self):
        """Print formatted usage information."""
        usage = self.format_usage().strip()
        formatted_usage = f"{COLORS['BOLD']}{COLORS['BLUE']}Usage:{COLORS['RESET']} {usage[7:]}"
        print(formatted_usage, file=sys.stderr)

    def error(self, message):
        """Print a clean error message and exit."""
        self.print_usage()
        log_error(message)
        raise ValueError(message)


def str_to_float(value: str) -> float:
    """Convert string to float with proper error handling."""
    try:
        return float(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"Invalid float value: '{value}'")


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = CustomArgumentParser(
        prog='chat',
        description='Chat from the comfort of your terminal',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Add arguments
    parser.add_argument('-s', '--system', type=str,
                        help='Set the system message for the chatbot.')
    parser.add_argument('-m', '--messages', nargs='+', type=str,
                        help='Write a message for the chatbot.')
    parser.add_argument('-c', '--conversation', action='store_true',
                        help='Flag for conversation mode.')
    parser.add_argument('--model', type=str,
                        help='Specify the model to use.')
    parser.add_argument('--available-models-gpt', action='store_true',
                        help='List available ChatGPT models.')
    parser.add_argument('--available-models', action='store_true',
                        help='List available models.')
    parser.add_argument('--available-tools', action='store_true',
                        help='List available tools.')
    parser.add_argument('-t', '--temperature', type=str_to_float,
                        help='Set the temperature for the model.')

    # Reasoning effort choices
    reasoning_choices = ['low', 'medium', 'high']
    parser.add_argument('-e', '--reasoning-effort', choices=reasoning_choices,
                        help=f'Set the reasoning effort: {", ".join(reasoning_choices)}. '
                        'Only applicable to reasoning models.')

    parser.add_argument('--max-tokens', type=int,
                        help='Absolute maximum number of tokens to generate.')

    # Provider choices
    provider_choices = ['openai', 'anthropic']
    parser.add_argument('-p', '--provider', choices=provider_choices,
                        help=f"Choose the provider: {', '.join(provider_choices)} (default: openai).")

    # Tool usage flags
    parser.add_argument('--use-tools', action='store_true', dest='use_tools',
                        help='Enable tool usage.')
    parser.add_argument('--no-tools', action='store_false', dest='use_tools',
                        help='Disable tool usage.')
    parser.set_defaults(use_tools=None)

    try:
        return parser.parse_args()
    except Exception as e:
        log_error(f"{COLORS['ERROR']}Error parsing arguments: {e}{COLORS['RESET']}")
        raise


def setup_signal_handlers() -> None:
    """Setup signal handlers for graceful shutdown."""

    def sigint_handler(signum, frame):
        logger.info('\nShutdown signal received, exiting.')
        raise KeyboardInterrupt

    def sigterm_handler(signum, frame):
        logger.info('\nSIGTERM signal received, exiting.')
        raise SystemExit

    signal.signal(signal.SIGINT, sigint_handler)
    signal.signal(signal.SIGTERM, sigterm_handler)


async def initialize_chatbot(args: argparse.Namespace) -> Chatbot:
    """Initialize the chatbot with configuration."""
    try:
        home_directory = os.path.expanduser('~')
        config_path = os.path.join(home_directory, '.chatconfig.json')
        provider_name = args.provider if args.provider else None

        # Create chatbot but don't initialize tools yet
        chatbot = await Chatbot.create(config_path, provider_name)

        return chatbot

    except Exception as e:
        log_error(f'Failed to initialize chatbot: {str(e)}', e.__class__.__name__)
        raise


async def list_models(args: argparse.Namespace, chatbot: Chatbot) -> bool:
    """Handle listing commands (models, tools). Returns True if a list command was executed."""

    if args.available_models:
        try:
            print(await chatbot.get_model_list())
            return True
        except Exception as e:
            log_error(f'Failed to retrieve available models: {str(e)}', e.__class__.__name__)
            raise

    if args.available_models_gpt:
        try:
            print(await chatbot.get_model_list('gpt'))  # Pass 'gpt' as model_type filter
            return True
        except Exception as e:
            log_error(f'Failed to retrieve available ChatGPT models: {str(e)}',
                      e.__class__.__name__)
            raise

    return False


async def list_available_tools(args: argparse.Namespace, chatbot: Chatbot) -> bool:
    if args.available_tools:
        try:
            print(chatbot.get_tool_list())
            return True
        except Exception as e:
            log_error(f'Failed to retrieve available tools: {str(e)}', e.__class__.__name__)
            raise

    return False


async def update_chatbot_config(args: argparse.Namespace, chatbot: Chatbot) -> None:
    """Update chatbot configuration based on arguments."""

    if args.provider:
        await chatbot.set_provider(args.provider)
    if args.system:
        await chatbot.set_system_message(args.system)
    if args.model:
        await chatbot.set_model(args.model)
    if args.reasoning_effort:
        await chatbot.set_reasoning_effort(args.reasoning_effort)
    if args.max_tokens:
        await chatbot.set_max_tokens(args.max_tokens)
    if args.temperature is not None:
        await chatbot.set_temperature(args.temperature)
    if args.use_tools is not None:
        await chatbot.set_use_tools(args.use_tools)


async def enter_interactive_mode(args: argparse.Namespace, chatbot: Chatbot) -> None:
    """Enter interactive mode if no arguments are passed."""

    if not any([
        args.messages,
        args.conversation,
        args.model,
        args.system,
        args.temperature is not None,
        args.use_tools is not None,
        args.reasoning_effort,
        args.provider,
        args.max_tokens,
    ]):
        await interactive_mode(chatbot)
        return True
    return False


async def start_new_conversation(args: argparse.Namespace, chatbot: Chatbot) -> bool:
    """Handle conversation mode. Returns True if conversation mode was handled."""

    if args.conversation:
        if args.messages:
            for message in args.messages:
                await chatbot.append_message(message)
        else:
            await chatbot.set_user_messages([])
            print(f"{COLORS['BOLD']}New conversation started.{COLORS['RESET']}\n"
                  f"Continue chatting with: ch -c -m 'Your message.'")
            print_settings(
                provider=chatbot.provider,
                model=chatbot.model,
                reasoning_effort=chatbot.reasoning_effort,
                max_tokens=chatbot.max_tokens,
                temperature=chatbot.temperature,
                use_tools=chatbot.use_tools,
                system_message=chatbot.system_message,
                user_messages=chatbot.user_messages,
                header='Current settings:',
            )
            return True

    return False


async def update_settings(args: argparse.Namespace, chatbot: Chatbot) -> bool:
    """Handle settings update display. Returns True if settings were displayed."""

    if any([
        args.model,
        args.system,
        args.temperature is not None,
        args.use_tools is not None,
        args.reasoning_effort,
        args.provider,
        args.max_tokens,
    ]) and not args.messages:
        print_settings(
            provider=chatbot.provider,
            model=chatbot.model,
            reasoning_effort=chatbot.reasoning_effort,
            max_tokens=chatbot.max_tokens,
            temperature=chatbot.temperature,
            use_tools=chatbot.use_tools,
            system_message=chatbot.system_message,
            user_messages=chatbot.user_messages,
            header='Settings updated.',
        )
        return True

    return False


async def process_chat_request(args: argparse.Namespace, chatbot: Chatbot) -> str:
    """Process the chat request and return the response."""

    if args.messages:
        if not args.conversation:
            await chatbot.set_user_messages(args.messages)

    try:
        logger.debug("Starting chat request")
        content = await chatbot.chat(
            system=chatbot.system_message,
            user_assistant=chatbot.user_messages,
            model=chatbot.model,
            temperature=chatbot.temperature,
            max_tokens=chatbot.max_tokens,
        )
        logger.debug("Chat request completed successfully")
        return content

    except Exception as e:
        log_error(str(e), e.__class__.__name__)
        if args.conversation:
            await chatbot.append_message('There was an error. Please try again.')
        else:
            await chatbot.set_user_messages([])
        raise

async def initialize_tools(chatbot: Chatbot) -> None:
    """Initialize tools for the chatbot if needed."""
    if chatbot.use_tools:
        try:
            await chatbot.initialize_tools()
            if not chatbot.tools:
                log_error("No tools available despite use_tools being enabled", 'Warning')
        except Exception as e:
            log_error(f'Failed to setup tools: {str(e)}', e.__class__.__name__)
            chatbot.tools = []
    else:
        chatbot.tools = []


async def interactive_mode(chatbot: Chatbot) -> None:
    """Run the chatbot in interactive mode."""
    print(f"{COLORS['BOLD']}{COLORS['BLUE']}Interactive Chat Mode{COLORS['RESET']}")
    print("Type 'exit', 'quit', or press Ctrl+C to end the session.\n")

    # Show current settings
    print_settings(
        provider=chatbot.provider,
        model=chatbot.model,
        reasoning_effort=chatbot.reasoning_effort,
        max_tokens=chatbot.max_tokens,
        temperature=chatbot.temperature,
        use_tools=chatbot.use_tools,
        system_message=chatbot.system_message,
        user_messages=[],  # Don't show conversation history in settings
        header='Current settings:',
    )
    print()

    # Clear any existing conversation for fresh start
    await chatbot.set_user_messages([])

    while True:
        try:
            # Get user input
            user_input = input(f"{COLORS['BOLD']}{COLORS['BLUE']}You:{COLORS['RESET']} ").strip()

            # Skip empty inputs
            if not user_input:
                continue

            # Commands inputs
            if user_input.startswith('/'):
                command_parts = user_input[1:].split()
                command = command_parts[0].lower()

                if command in ['exit', 'quit', 'q']:
                    print(f"{COLORS['BOLD']}Goodbye!{COLORS['RESET']}")
                    break
                elif command == 'help':
                    print("Available commands:")
                    print("  /settings - Show current settings")
                    print("  /model <name> - Change model")
                    print("  /temperature <value> - Change temperature")
                    print("  /effort <value> - Change reasoning effort")
                    print("  /tokens <value> - Change max tokens")
                    print("  /tools - List tools if enabled")
                    print("  /clear - Clear conversation history")
                    print("  /exit - Exit interactive mode")
                elif command == 'settings':
                    print_settings(
                        provider=chatbot.provider,
                        model=chatbot.model,
                        reasoning_effort=chatbot.reasoning_effort,
                        max_tokens=chatbot.max_tokens,
                        temperature=chatbot.temperature,
                        use_tools=chatbot.use_tools,
                        system_message=chatbot.system_message,
                        user_messages=chatbot.user_messages,
                        header='Current settings:',
                    )
                elif command == 'clear':
                    await chatbot.set_user_messages([])
                    print("Conversation history cleared.")
                elif command == 'model' and len(command_parts) > 1:
                    await chatbot.set_model(command_parts[1])
                    print(f"Model changed to: {command_parts[1]}")
                elif command == 'model' and len(command_parts) == 1:
                    print(await chatbot.get_model_list())
                elif command == 'tools':
                    if chatbot.use_tools:
                        print(chatbot.get_tool_list())
                    else:
                        print('Tools are disabled.')
                elif command == 'temperature' and len(command_parts) > 1:
                    try:
                        temp = float(command_parts[1])
                        await chatbot.set_temperature(temp)
                        print(f"Temperature changed to: {temp}")
                    except ValueError:
                        print("Invalid temperature value. Use a number between 0 and 2.")
                elif command == 'effort' and len(command_parts) > 1:
                    try:
                        effort = command_parts[1]
                        await chatbot.set_reasoning_effort(effort)
                        print(f"Effort changed to: {effort}")
                    except ValueError as e:
                        print(f'Invalid reasoning effort value. {e}')
                elif command == 'tokens' and len(command_parts) > 1:
                    try:
                        tokens = int(command_parts[1])
                        await chatbot.set_max_tokens(tokens)
                        print(f"Max tokens changed to: {tokens}")
                    except ValueError as e:
                        print(f'Invalid max_tokens value. {e}')
                else:
                    print("Unknown command. Type /help for available commands.")
                continue

            # Check for exit commands
            if user_input.lower() in ['exit', 'quit', 'q']:
                print(f"{COLORS['BOLD']}Goodbye!{COLORS['RESET']}")
                break

            # Add messages to conversation
            await chatbot.append_message(user_input)

            # Get LLM response
            print(f"{COLORS['BOLD']}{COLORS['GREEN']}Assistant:{COLORS['RESET']}")
            response, _ = await chatbot.chat_with_provider(
                system=chatbot.system_message,
                user_assistant=chatbot.user_messages,
                model=chatbot.model,
                temperature=chatbot.temperature,
                max_tokens=chatbot.max_tokens,
            )
            print(response)

            await chatbot.append_message(response)

            print()

        except EOFError:
            # Handle Ctrl+D
            print(f"\n{COLORS['BOLD']}Goodbye!{COLORS['RESET']}")
            break
        except Exception as e:
            log_error(e)
            await chatbot.append_message('There was an error. Please try again.')
            continue

async def async_main() -> None:
    """Main application entry point."""

    try:
        # Setup signal handlers
        setup_signal_handlers()

        # Parse arguments
        args = parse_arguments()

        # Initialize chatbot (without tools)
        chatbot = await initialize_chatbot(args)

        # Update chatbot configuration
        await update_chatbot_config(args, chatbot)

        # Handle list commands (exit early if executed)
        if await list_models(args, chatbot):
            return

        await initialize_tools(chatbot)

        # List tools if requested and exit
        if await list_available_tools(args, chatbot):
            return

        # Enter Interactive mode
        if await enter_interactive_mode(args, chatbot):
            return

        # Handle conversation mode (exit early if new conversation)
        if await start_new_conversation(args, chatbot):
            return

        # Handle settings update display (exit early if just updating settings)
        if await update_settings(args, chatbot):
            return

        # Process chat request
        content = await process_chat_request(args, chatbot)

        if args.conversation:
            await chatbot.append_message(content)

    except Exception:
        raise

    finally:
        if chatbot.mcp_client:
            await chatbot.mcp_client.cleanup()

def main() -> None:
    """Synchronous entry point for console scripts."""
    try:
        asyncio.run(async_main())
        exit_code = os.EX_OK
    except KeyboardInterrupt:
        exit_code = os.EX_OK
    except SystemExit:
        exit_code = 128 + signal.SIGTERM
    except Exception as e:
        log_error(f"Application error: {e}")
        exit_code = 1
    finally:
        logger.debug('Chat completed')
        sys.exit(exit_code)


if __name__ == '__main__':
    main()