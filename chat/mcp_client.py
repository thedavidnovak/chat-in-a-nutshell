#!/usr/bin/env python3

import json
from typing import Any
from contextlib import AsyncExitStack

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from chat.logging_setup import setup_logging
from chat.error_utils import log_error

logger = setup_logging(__name__)


class MCPClient:
    """MCP client for Chatbot."""

    def __init__(self, config_path: str | None = None):
        """Initialize the MCP client with optional configuration path.

        :param config_path: Path to the MCP server configuration JSON file
        """
        self.config_path = config_path
        self.servers: dict[str, dict[str, Any]] = {}
        self.exit_stack = AsyncExitStack()
        self.active_servers: dict[str, dict[str, Any]] = {}
        self._tools_cache: list[dict[str, Any]] = []

        # Load server configuration if path provided
        if self.config_path:
            self._load_server_config()

    def _load_server_config(self) -> None:
        """Load MCP server configuration from JSON file."""
        try:
            with open(self.config_path, 'r') as f:
                config = json.load(f)
                self.servers = config.get('mcpServers', {})

            if not self.servers:
                log_error("No MCP servers defined in configuration file.", 'Warning')
        except FileNotFoundError as e:
            log_error(f"MCP config file not found: {self.config_path}", e.__class__.__name__)
            self.servers = {}
        except json.JSONDecodeError as e:
            log_error(f"Invalid JSON in MCP config file: {e}", e.__class__.__name__)
            self.servers = {}
        except Exception as e:
            log_error(f"Failed to load MCP server configuration: {e}", e.__class__.__name__)
            self.servers = {}

    def get_server_names(self) -> list[str]:
        """Get list of configured server names.

        :return: List of server names
        """
        return list(self.servers.keys())

    async def connect_to_server(self, server_identifier: str) -> bool:
        """Connect to MCP server.

        :param server_identifier: server name from config
        :return: True if connection successful, False otherwise
        """
        if server_identifier in self.active_servers:
            logger.debug(f"Already connected to server '{server_identifier}'")
            return True

        if server_identifier not in self.servers:
            log_error(f"Server '{server_identifier}' not found in configuration", 'Warning')
            return False

        server_config = self.servers[server_identifier]
        server_params = StdioServerParameters(
            command=server_config.get('command', ''),
            args=server_config.get('args', []),
            env=server_config.get('env')
        )
        if not server_params:
            return False

        try:
            stdio_transport = await self.exit_stack.enter_async_context(
                stdio_client(server_params)
            )
            stdio, write = stdio_transport
            session = await self.exit_stack.enter_async_context(
                ClientSession(stdio, write)
            )

            await session.initialize()

            # Get available tools
            response = await session.list_tools()

            tools = [
                {
                    "name": tool.name,
                    "description": tool.description,
                    "definition": {
                        "type": "function",
                        "function": {
                            "name": tool.name,
                            "description": tool.description,
                            "parameters": tool.inputSchema
                        }
                    }
                }
                for tool in response.tools
            ]

            # Store the session and tools
            self.active_servers[server_identifier] = {
                "session": session,
                "tools": tools
            }

            self._tools_cache = []
            for server_info in self.active_servers.values():
                self._tools_cache.extend(server_info["tools"])

            logger.debug(f"Connected to MCP server '{server_identifier}' "
                         f"with tools: {[tool['name'] for tool in tools]}")
            return True

        except Exception as e:
            log_error(f"Failed to connect to MCP server '{server_identifier}': {e}")
            return False

    async def call_tool(self, tool_name: str, tool_args: dict[str, Any], confirm_callback=None) -> dict[str, Any]:
        """Call a tool by name with the provided arguments.

        :param tool_name: Name of the tool to call
        :param tool_args: Arguments to pass to the tool
        :param confirm_callback: Optional callback for user confirmation
        :return: Dictionary containing the tool result and status
        """
        for server_name, server_info in self.active_servers.items():
            session = server_info["session"]
            tools = server_info["tools"]

            if any(tool["name"] == tool_name for tool in tools):
                try:
                    # User confirmation if callback provided
                    if confirm_callback:
                        if not await confirm_callback(tool_name, tool_args):
                            return {"content": "Tool execution cancelled", "status": "cancelled"}

                    # Call tool
                    result = await session.call_tool(tool_name, tool_args)

                    logger.debug(f"Raw MCP result for {tool_name}: {result}")

                    # Process different possible result structures
                    if hasattr(result, 'content'):
                        content = result.content
                        if isinstance(content, list):
                            text_content = []
                            for item in content:
                                if hasattr(item, 'text'):
                                    text_content.append(item.text)
                                elif isinstance(item, dict) and 'text' in item:
                                    text_content.append(item['text'])
                                else:
                                    text_content.append(str(item))
                            content = '\n'.join(text_content)
                        elif hasattr(content, 'text'):
                            content = content.text

                        return {
                            "content": content,
                            "status": "success"
                        }
                    else:
                        return {
                            "content": str(result),
                            "status": "success"
                        }

                except Exception as e:
                    log_error(f"Error calling tool {tool_name} on server {server_name}: {e}")
                    return {
                        "content": f"Error: {str(e)}",
                        "status": "error"
                    }

        return {
            "content": f"Error: Tool '{tool_name}' not found in any active server",
            "status": "error"
        }

    def get_tools(self) -> list[dict[str, Any]]:
        """Get the list of available tools.

        :return: List of tool definitions
        """
        return self._tools_cache.copy()

    async def cleanup(self) -> None:
        """Clean up resources."""
        try:
            # Clear references to prevent further operations
            self.active_servers.clear()
            self._tools_cache.clear()

            # Close all connections
            await self.exit_stack.aclose()
            logger.debug("MCPClient cleanup completed")
        except Exception as e:
            logger.debug(f"Error during MCP client cleanup: {e}")
            self.exit_stack = AsyncExitStack()
