# -*- coding: utf-8 -*-
"""MCP (Model Context Protocol) STDIO client."""

import os
import sys
from contextlib import AsyncExitStack
from pathlib import Path
from typing import Any, Dict, List

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from ..config import SERVER_SCRIPT
from ..utils.logging_config import setup_logging


class MCPClient:
    """
    MCP client using STDIO transport.

    Manages connection to an MCP server via STDIO and provides tool execution.

    Attributes:
        session: MCP ClientSession instance
        exit_stack: AsyncExitStack for resource management
        tools: List of available tools
        _running: Whether the client is running
    """

    def __init__(self) -> None:
        """Initialize the MCP client."""
        self.session: ClientSession | None = None
        self.exit_stack = AsyncExitStack()
        self.tools: List[Dict] = []
        self._running = False
        self.log = setup_logging("mcp-client")

    async def connect_to_server(self, server_script_path: str) -> None:
        """
        Connect to an MCP server.

        Args:
            server_script_path: Path to the MCP server script

        Raises:
            FileNotFoundError: If the server script doesn't exist
        """
        self.log.info(f"Connecting to server: {server_script_path}")

        if not Path(server_script_path).exists():
            raise FileNotFoundError(f"Server not found: {server_script_path}")

        # Server parameters
        server_params = StdioServerParameters(
            command=sys.executable,
            args=[server_script_path],
            env={**os.environ},
        )

        # Create transport and session via context manager
        stdio_transport = await self.exit_stack.enter_async_context(
            stdio_client(server_params)
        )

        self.session = await self.exit_stack.enter_async_context(
            ClientSession(
                stdio_transport[0],
                stdio_transport[1],
                client_info={"name": "mcp-client", "version": "1.0.0"},
            )
        )

        # Initialize session
        await self.session.initialize()

        # Get tool list
        tools_result = await self.session.list_tools()
        self.tools = [
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description or "",
                    "parameters": tool.inputSchema,
                },
            }
            for tool in (tools_result.tools if tools_result else [])
        ]

        self._running = True
        self.log.info(f"Connected to server. Available tools: {len(self.tools)}")

        if self.tools:
            for tool in self.tools:
                self.log.info(
                    f"  - {tool['function']['name']}: {tool['function']['description']}"
                )

    async def call_tool(self, name: str, arguments: Dict) -> str:
        """
        Call an MCP server tool.

        Args:
            name: Tool name
            arguments: Tool arguments

        Returns:
            Tool result as string
        """
        if not self._running or not self.session:
            return "[Error] Server not connected"

        try:
            result = await self.session.call_tool(name, arguments)

            # Combine all text blocks into one response
            text_parts = []
            for block in result.content or []:
                if hasattr(block, "text"):
                    text_parts.append(block.text)

            return "\n".join(text_parts) if text_parts else "Tool executed without result"

        except Exception as exc:
            error_msg = f"Error calling tool {name}: {exc}"
            self.log.error(error_msg)
            return f"[Error] {error_msg}"

    async def cleanup(self) -> None:
        """Free resources."""
        if self._running:
            self._running = False
            await self.exit_stack.aclose()
            self.log.info("MCP client resources freed")

    @property
    def available_tools(self) -> List[Dict]:
        """Return list of available tools."""
        return self.tools
