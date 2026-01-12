#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OpenAI-compatible chat client with MCP tools integration.
Refactored to use modular architecture with separate services.
"""
from __future__ import annotations

import asyncio
import json
import os
import re
import sys
import typing as tp
from datetime import datetime

import httpx
import openai
from dotenv import load_dotenv

# Add src to path for imports
sys.path.insert(0, str(os.path.join(os.path.dirname(__file__), "src")))

from src.clients import DockerMCPClient, MCPClient
from src.config import (
    DEFAULT_MODEL,
    DEFAULT_TEMPERATURE,
    DOCKER_MCP_HOST,
    DOCKER_MCP_PORT,
    SAVE_DIR,
    SERVER_SCRIPT,
    SYSTEM_PROMPT,
)
from src.services import RAGService, HelpService
from src.utils import setup_logging

load_dotenv()

log = setup_logging("mcp-client")


# -------------------- UTILITIES --------------------
def build_openai_client() -> openai.AsyncOpenAI:
    """Build OpenAI client with environment settings."""
    key = os.getenv("OPENAI_API_KEY")
    base = os.getenv("OPENAI_BASE_URL")
    verify = os.getenv("OPENAI_VERIFY_SSL", "true").lower() != "false"
    http = httpx.AsyncClient(verify=verify)
    return openai.AsyncOpenAI(api_key=key, base_url=base, http_client=http)


def save_conversation(history: tp.List[dict], name: str | None = None) -> str:
    """Save conversation history to JSON file."""
    name = (
        f"conversation_{datetime.now():%Y%m%d_%H%M%S}.json"
        if name is None
        else name
    )
    if not name.endswith(".json"):
        name += ".json"

    path = SAVE_DIR / name
    try:
        path.write_text(json.dumps(history, ensure_ascii=False, indent=2))
        log.info(f"Conversation saved to {path}")
        return str(path)
    except Exception as exc:
        error_msg = f"Error saving: {exc}"
        log.error(error_msg)
        return error_msg


def load_conversation(name: str) -> tp.Tuple[tp.List[dict] | None, str]:
    """Load conversation history from file."""
    try:
        if name.isdigit():
            # Load by number
            files = sorted(SAVE_DIR.glob("conversation_*.json"), reverse=True)
            idx = int(name) - 1
            if 0 <= idx < len(files):
                path = files[idx]
                return json.loads(path.read_text()), str(path)
            return None, "Invalid save number."

        # Load by name
        path = SAVE_DIR / (
            name if name.endswith(".json") else f"{name}.json"
        )
        if path.exists():
            return json.loads(path.read_text()), str(path)
        return None, f"File {path} not found."

    except Exception as exc:
        error_msg = f"Error loading: {exc}"
        log.error(error_msg)
        return None, error_msg


def list_saved_conversations() -> str:
    """Return list of saved conversations."""
    files = sorted(SAVE_DIR.glob("conversation_*.json"), reverse=True)
    if not files:
        return "No saved conversations."

    lines = ["Saved conversations:", "=" * 40]
    for idx, fp in enumerate(files, 1):
        # Extract date from filename
        ts_match = fp.stem.replace("conversation_", "")
        try:
            dt = datetime.strptime(ts_match, "%Y%m%d_%H%M%S")
            nice_date = dt.strftime("%Y-%m-%d %H:%M:%S")
        except Exception:
            nice_date = fp.stem

        lines.append(f"{idx}. {nice_date} – {fp.name}")

    return "\n".join(lines)


async def create_summary(
    cli: openai.AsyncOpenAI, model: str, history: tp.List[dict]
) -> str:
    """Create a brief summary of the conversation."""
    msgs = [m for m in history if m["role"] in ("user", "assistant")]
    if not msgs:
        return "No history to summarize."

    text = "Please create a brief summary of the following dialogue. Highlight main topics and details:\n\n"
    for msg in msgs:
        role = "User" if msg["role"] == "user" else "AI"
        text += f"{role}: {msg['content']}\n\n"

    try:
        resp = await cli.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": text}],
            temperature=0.3,
            max_tokens=512,
        )
        return resp.choices[0].message.content or "Failed to create summary."
    except Exception as exc:
        error_msg = f"Error creating summary: {exc}"
        log.error(error_msg)
        return error_msg


# -------------------- CHAT CLIENT --------------------
class ChatClient:
    """
    Main chat client with MCP, RAG, and Help integration.

    Attributes:
        model_name: Name of the LLM model
        openai_client: OpenAI client instance
        mcp_client: MCP STDIO client
        docker_mcp_client: Docker MCP client
        conversation: Conversation history
        temperature: LLM temperature setting
        rag_service: RAG service instance
        help_service: Help service instance
        use_rag: Whether RAG mode is enabled
    """

    def __init__(self, model_name: str = DEFAULT_MODEL) -> None:
        """
        Initialize the chat client.

        Args:
            model_name: Name of the LLM model to use
        """
        self.model_name = model_name
        self.openai_client = build_openai_client()
        self.mcp_client = MCPClient()
        self.docker_mcp_client = DockerMCPClient()
        self.conversation: tp.List[dict] = [
            {"role": "system", "content": SYSTEM_PROMPT}
        ]
        self.temperature = DEFAULT_TEMPERATURE
        self.rag_service = RAGService()
        self.help_service = HelpService()
        self.use_rag = False

    async def process_query(self, query: str) -> str:
        """
        Process user query using available tools.

        Args:
            query: User query text

        Returns:
            Assistant response
        """
        log.info(f"Processing query: {query[:50]}...")

        # Check for help commands
        if query.lower().startswith("/help"):
            return await self._handle_help_command(query)

        # Check for RAG control commands
        if query.lower().startswith("/rag"):
            return await self._handle_rag_command(query)

        # Use RAG for regular queries if enabled
        if self.use_rag:
            return await self._process_with_rag(query)

        # Process normally with tools
        return await self._process_with_tools(query)

    async def _handle_help_command(self, query: str) -> str:
        """Handle /help commands with RAG documentation and git context."""
        parts = query.split(" ", 1)
        subcommand = parts[1].strip() if len(parts) > 1 else ""

        # Handle subcommands
        if subcommand.lower() in ("style", "guidelines", "conventions"):
            # Style guide help
            return await self._help_with_style_guide()
        elif subcommand.lower().startswith("api"):
            # API reference help
            component = subcommand[3:].strip() if len(subcommand) > 3 else ""
            return await self._help_with_api(component)
        elif subcommand.lower() in ("structure", "architecture", "modules"):
            # Project structure help
            return await self._help_with_structure()
        elif subcommand.lower() == "git":
            # Git status help
            return await self._help_with_git_context()
        else:
            # General help with documentation search
            return await self._help_with_docs(subcommand or "project overview getting started")

    async def _help_with_docs(self, question: str) -> str:
        """Get help with documentation search and git context."""
        try:
            # Gather git context first
            git_context = await self._gather_git_context()

            # Get help response from service
            help_prompt = self.help_service.get_help_response(
                question, git_context=git_context, top_k=5
            )

            messages = self.conversation + [{"role": "user", "content": help_prompt}]

            response = await self.openai_client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=self.temperature,
                max_tokens=2048,
            )
            content = response.choices[0].message.content or ""

            # Don't add help queries to history
            return content

        except Exception as exc:
            error_msg = f"Error getting help: {exc}"
            log.error(error_msg)
            return (
                f"Sorry, I couldn't find help information. "
                f"Make sure to run 'python index_docs.py' to index project documentation. "
                f"Error: {exc}"
            )

    async def _help_with_style_guide(self) -> str:
        """Get coding style guidelines."""
        style_guide = self.help_service.get_style_guide_help()
        messages = self.conversation + [{"role": "user", "content": style_guide}]

        response = await self.openai_client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=self.temperature,
            max_tokens=2048,
        )
        return response.choices[0].message.content or "No style guide available."

    async def _help_with_api(self, component: str) -> str:
        """Get API reference for a component."""
        api_ref = self.help_service.get_api_reference(component)
        messages = self.conversation + [{"role": "user", "content": api_ref}]

        response = await self.openai_client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=self.temperature,
            max_tokens=2048,
        )
        return response.choices[0].message.content or "No API reference available."

    async def _help_with_structure(self) -> str:
        """Get project structure information."""
        structure_info = self.help_service.get_project_structure()
        messages = self.conversation + [{"role": "user", "content": structure_info}]

        response = await self.openai_client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=self.temperature,
            max_tokens=2048,
        )
        return response.choices[0].message.content or "No structure information available."

    async def _help_with_git_context(self) -> str:
        """Get help with current git context."""
        log.info("=" * 50)
        log.info("[/help git] Starting git context request")
        start_time = datetime.now()

        try:
            # Get git info directly (NOT through MCP to avoid blocking)
            import subprocess
            log.info("[/help git] Running git commands directly...")

            git_start = datetime.now()
            info = []

            # Branch
            try:
                result = subprocess.run(
                    ["git", "rev-parse", "--abbrev-ref", "HEAD"],
                    capture_output=True, text=True, timeout=5, cwd="."
                )
                if result.stdout and not result.stderr:
                    info.append(f"Branch: {result.stdout.strip()}")
            except Exception as e:
                log.warning(f"git branch failed: {e}")

            # Remote
            try:
                result = subprocess.run(
                    ["git", "config", "--get", "remote.origin.url"],
                    capture_output=True, text=True, timeout=5, cwd="."
                )
                if result.stdout and not result.stderr:
                    info.append(f"Remote: {result.stdout.strip()}")
            except Exception as e:
                log.warning(f"git remote failed: {e}")

            # Commits
            try:
                result = subprocess.run(
                    ["git", "rev-list", "--count", "HEAD"],
                    capture_output=True, text=True, timeout=5, cwd="."
                )
                if result.stdout and not result.stderr:
                    info.append(f"Commits: {result.stdout.strip()}")
            except Exception as e:
                log.warning(f"git count failed: {e}")

            # Latest
            try:
                result = subprocess.run(
                    ["git", "log", "-1", "--format=%h %s (%cr)"],
                    capture_output=True, text=True, timeout=5, cwd="."
                )
                if result.stdout and not result.stderr:
                    info.append(f"Latest: {result.stdout.strip()}")
            except Exception as e:
                log.warning(f"git log failed: {e}")

            # Status
            try:
                result = subprocess.run(
                    ["git", "status", "--short"],
                    capture_output=True, text=True, timeout=5, cwd="."
                )
                if result.stderr and "not a git repository" in result.stderr.lower():
                    info.append("Not a git repository")
                elif result.stdout.strip():
                    info.append(f"Status: {result.stdout.strip()[:100]}")
                else:
                    info.append("Clean (no changes)")
            except Exception as e:
                log.warning(f"git status failed: {e}")

            git_elapsed = (datetime.now() - git_start).total_seconds()
            repo_info = "\n".join(info)
            log.info(f"[/help git] Git commands completed in {git_elapsed:.2f}s")
            log.info(f"[/help git] Result:\n{repo_info}")

            git_summary = f"""Current git repository state:

{repo_info}

Provide a brief summary of the project state and any recommendations."""

            log.info("[/help git] Sending to LLM for analysis...")
            llm_start = datetime.now()
            messages = self.conversation + [{"role": "user", "content": git_summary}]

            response = await self.openai_client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=self.temperature,
                max_tokens=2048,
            )
            llm_elapsed = (datetime.now() - llm_start).total_seconds()
            log.info(f"[/help git] LLM response completed in {llm_elapsed:.2f}s")

            total_elapsed = (datetime.now() - start_time).total_seconds()
            log.info(f"[/help git] Total time: {total_elapsed:.2f}s")
            log.info("=" * 50)

            return response.choices[0].message.content or "Could not retrieve git context."

        except Exception as exc:
            total_elapsed = (datetime.now() - start_time).total_seconds()
            log.error(f"[/help git] Error after {total_elapsed:.2f}s: {exc}")
            log.info("=" * 50)
            return f"Error getting git context: {exc}"

    async def _gather_git_context(self) -> Dict[str, str]:
        """Gather git context for help queries using subprocess directly (avoids MCP blocking)."""
        context = {}

        try:
            import subprocess
            info = []

            # Branch
            try:
                result = subprocess.run(
                    ["git", "rev-parse", "--abbrev-ref", "HEAD"],
                    capture_output=True, text=True, timeout=5, cwd="."
                )
                if result.stdout and not result.stderr:
                    context["current_branch"] = result.stdout.strip()
            except Exception:
                pass

            # Remote
            try:
                result = subprocess.run(
                    ["git", "config", "--get", "remote.origin.url"],
                    capture_output=True, text=True, timeout=5, cwd="."
                )
                if result.stdout and not result.stderr:
                    info.append(f"Remote: {result.stdout.strip()}")
            except Exception:
                pass

            # Commits
            try:
                result = subprocess.run(
                    ["git", "rev-list", "--count", "HEAD"],
                    capture_output=True, text=True, timeout=5, cwd="."
                )
                if result.stdout and not result.stderr:
                    info.append(f"Commits: {result.stdout.strip()}")
            except Exception:
                pass

            # Latest
            try:
                result = subprocess.run(
                    ["git", "log", "-1", "--format=%h %s (%cr)"],
                    capture_output=True, text=True, timeout=5, cwd="."
                )
                if result.stdout and not result.stderr:
                    info.append(f"Latest: {result.stdout.strip()}")
            except Exception:
                pass

            if info:
                context["repo_info"] = "\n".join(info)

        except Exception as e:
            log.warning(f"Could not gather git context: {e}")

        return context

    async def _handle_rag_command(self, query: str) -> str:
        """Handle RAG-specific commands."""
        parts = query.split(" ", 2)
        if len(parts) == 1:
            # Just /rag - toggle RAG mode
            self.use_rag = not self.use_rag
            status = "enabled" if self.use_rag else "disabled"
            return f"RAG mode {status}."
        elif len(parts) == 2:
            if parts[1].lower() == "rerank":
                # Toggle reranker
                rerank_status = self.rag_service.toggle_reranker()
                status = "enabled" if rerank_status else "disabled"
                return f"Reranker {status}."
            elif parts[1].lower().startswith("threshold:"):
                # Set threshold
                try:
                    threshold_value = float(parts[1].split(":", 1)[1])
                    new_threshold = self.rag_service.set_relevance_threshold(
                        threshold_value
                    )
                    return f"Relevance threshold set to: {new_threshold}."
                except (ValueError, IndexError):
                    return "Invalid threshold format. Use: /rag threshold:0.5"
            else:
                # RAG query
                actual_query = parts[1]
                return await self._process_with_rag(actual_query)
        else:  # len(parts) == 3
            subcommand = parts[1].lower()
            value = parts[2]

            if subcommand == "threshold":
                try:
                    threshold_value = float(value)
                    new_threshold = self.rag_service.set_relevance_threshold(
                        threshold_value
                    )
                    return f"Relevance threshold set to: {new_threshold}."
                except ValueError:
                    return "Invalid threshold value. Use a number from 0 to 1."
            else:
                # RAG query
                actual_query = f"{parts[1]} {parts[2]}"
                return await self._process_with_rag(actual_query)

    async def _process_with_rag(self, query: str) -> str:
        """Process query using RAG."""
        try:
            rag_prompt = self.rag_service.get_rag_response(query)
            messages = self.conversation + [{"role": "user", "content": rag_prompt}]

            response = await self.openai_client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=self.temperature,
                max_tokens=2048,
            )
            content = response.choices[0].message.content or ""

            # Add original query and response to history
            self.conversation.append({"role": "user", "content": query})
            self.conversation.append({"role": "assistant", "content": content})

            return content
        except Exception as exc:
            error_msg = f"Error processing RAG query: {exc}"
            log.error(error_msg)
            return f"Sorry, an error occurred while processing the RAG query: {exc}"

    async def _process_with_tools(self, query: str) -> str:
        """Process query using available tools."""
        try:
            # Combine tools from both clients
            all_tools = []
            all_tools.extend(self.mcp_client.available_tools or [])
            all_tools.extend(self.docker_mcp_client.available_tools or [])

            messages = self.conversation + [{"role": "user", "content": query}]

            while True:  # Loop for multiple rounds of tool calls
                response = await self.openai_client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    tools=all_tools or None,
                    tool_choice="auto" if all_tools else None,
                    temperature=self.temperature,
                    max_tokens=2048,
                )

                assistant_message = response.choices[0].message
                content = assistant_message.content or ""

                # Check for tool calls
                if assistant_message.tool_calls:
                    log.info(
                        f"Detected tool calls: {[tc.function.name for tc in assistant_message.tool_calls]}"
                    )

                    # Add assistant message
                    messages.append(
                        {
                            "role": "assistant",
                            "content": content,
                            "tool_calls": [
                                {
                                    "id": tc.id,
                                    "type": "function",
                                    "function": {
                                        "name": tc.function.name,
                                        "arguments": tc.function.arguments,
                                    },
                                }
                                for tc in assistant_message.tool_calls
                            ],
                        }
                    )

                    # Execute all tool calls
                    for tool_call in assistant_message.tool_calls:
                        tool_name = tool_call.function.name

                        try:
                            tool_args = json.loads(tool_call.function.arguments)

                            # Handle XML-like arguments format
                            if isinstance(tool_args, str):
                                if "<tool_name>" in tool_args and "</tool_name>" in tool_args:
                                    tool_args = self._parse_xml_like_args(tool_args)
                            elif isinstance(tool_args, dict):
                                for key, value in tool_args.items():
                                    if isinstance(value, str) and "<tool_name>" in value and "</tool_name>" in value:
                                        tool_args[key] = self._parse_xml_like_args(value)

                            log.info(f"Calling tool: {tool_name} with arguments: {tool_args}")

                            tool_result = await self._call_appropriate_tool(
                                tool_name, tool_args
                            )

                            log.info(f"Tool {tool_name} result: {tool_result[:100]}...")

                            messages.append(
                                {
                                    "role": "tool",
                                    "tool_call_id": tool_call.id,
                                    "content": tool_result,
                                }
                            )
                        except json.JSONDecodeError as e:
                            log.error(f"Error decoding tool arguments {tool_name}: {e}")
                            try:
                                corrected_args = self._parse_xml_like_args(
                                    tool_call.function.arguments
                                )
                                tool_result = await self._call_appropriate_tool(
                                    tool_name, corrected_args
                                )

                                messages.append(
                                    {
                                        "role": "tool",
                                        "tool_call_id": tool_call.id,
                                        "content": tool_result,
                                    }
                                )
                            except Exception as parse_error:
                                log.error(f"Error parsing tool arguments {tool_name}: {parse_error}")
                                messages.append(
                                    {
                                        "role": "tool",
                                        "tool_call_id": tool_call.id,
                                        "content": f"[Error] Failed to parse tool arguments: {parse_error}",
                                    }
                                )

                    # Continue loop for potential more tool calls
                    continue
                else:
                    # No tool calls, final response
                    self.conversation.extend(messages[1:])
                    return content

        except Exception as exc:
            error_msg = f"Error processing query: {exc}"
            log.error(error_msg)
            return f"Sorry, an error occurred: {exc}"

    def _parse_xml_like_args(self, xml_string: str) -> dict:
        """Parse XML-like argument string into dictionary."""
        args_dict = {}

        # Regex for finding <tool_name> and <tool_description> pairs
        pattern = r"<tool_name>(.*?)</tool_name>\s*<tool_description>(.*?)</tool_description>"
        matches = re.findall(pattern, xml_string, re.DOTALL)

        for key, value in matches:
            key = key.strip()
            value = value.strip()

            # Type inference
            if value.lower() in ("true", "false"):
                value = value.lower() == "true"
            elif value.isdigit():
                value = int(value)
            elif value.replace(".", "").isdigit():
                value = float(value)
            elif value.startswith('"') and value.endswith('"'):
                value = value[1:-1]

            args_dict[key] = value

        return args_dict

    async def _call_appropriate_tool(self, name: str, arguments: dict) -> str:
        """Determine which client to use for tool call."""
        start = datetime.now()
        stdio_tool_names = [
            tool["function"]["name"] for tool in self.mcp_client.available_tools
        ]
        docker_tool_names = [
            tool["function"]["name"]
            for tool in self.docker_mcp_client.available_tools
        ]

        log.info(f"[MCP] → {name}({arguments})")

        if name in stdio_tool_names:
            result = await self.mcp_client.call_tool(name, arguments)
            elapsed = (datetime.now() - start).total_seconds()
            log.info(f"[MCP] ← {name} completed in {elapsed:.3f}s | {len(result)} chars")
            return result
        elif name in docker_tool_names:
            result = await self.docker_mcp_client.call_tool(name, arguments)
            elapsed = (datetime.now() - start).total_seconds()
            log.info(f"[MCP] ← {name} completed in {elapsed:.3f}s | {len(result)} chars")
            return result
        else:
            # Try both clients
            if self.mcp_client._running:
                result = await self.mcp_client.call_tool(name, arguments)
                if result and not (
                    result.startswith("[Error]") or result.startswith("[Warning]")
                ):
                    elapsed = (datetime.now() - start).total_seconds()
                    log.info(f"[MCP] ← {name} completed in {elapsed:.3f}s (fallback)")
                    return result

            if self.docker_mcp_client.connected:
                result = await self.docker_mcp_client.call_tool(name, arguments)
                if result and not (
                    result.startswith("[Error]") or result.startswith("[Warning]")
                ):
                    elapsed = (datetime.now() - start).total_seconds()
                    log.info(f"[MCP] ← {name} completed in {elapsed:.3f}s (fallback)")
                    return result

            elapsed = (datetime.now() - start).total_seconds()
            log.error(f"[MCP] ✗ {name} not found (took {elapsed:.3f}s)")
            return f"[Error] Tool '{name}' not found in any active MCP servers"

    async def start(self, server_script: str) -> None:
        """Start client and connect to MCP servers."""
        log.info("Starting chat client...")

        # Connect to STDIO MCP server
        await self.mcp_client.connect_to_server(server_script)

        # Connect to Docker MCP server (don't raise exception if unavailable)
        await self.docker_mcp_client.connect_to_server()

        log.info("Chat client ready!")

    async def cleanup(self) -> None:
        """Free resources."""
        await self.mcp_client.cleanup()
        await self.openai_client.close()
        log.info("Client stopped")

    def add_message(self, role: str, content: str) -> None:
        """Add message to conversation history."""
        self.conversation.append({"role": role, "content": content})


# -------------------- INTERACTIVE INTERFACE --------------------
async def interactive_chat(client: ChatClient) -> None:
    """Interactive chat loop."""
    print("=" * 60)
    print("ROBOT Chat Client with MCP Tools")
    print("Commands: quit/exit, save <name>, load <name>, temp <0-2>, clear, print")
    print(
        "RAG commands: /rag (toggle), /rag <question> (RAG query), /rag rerank (toggle reranker), /rag threshold <value>"
    )
    print(
        "Help commands: /help, /help <question>, /help style, /help api, /help structure, /help git"
    )
    print("=" * 60)

    while True:
        try:
            user_input = input("\nUSER You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n\nGOODBYE Goodbye!")
            break

        if not user_input:
            continue

        # Handle commands
        if user_input.lower() in ("quit", "exit"):
            print("GOODBYE Goodbye!")
            break

        if user_input.lower() == "clear":
            client.conversation = [{"role": "system", "content": SYSTEM_PROMPT}]
            print("CLEARED History cleared")
            continue

        if user_input.lower() == "print":
            print("=" * 50, "HISTORY Conversation History:", "=" * 50, sep="\n")
            for i, msg in enumerate(client.conversation[1:], 1):
                print(
                    f"{i}. {msg['role'].upper()}: {msg['content'][:100]}{'...' if len(msg['content']) > 100 else ''}"
                )
            print("=" * 50)
            continue

        if user_input.lower().startswith("save "):
            name = user_input[5:].strip()
            path = save_conversation(client.conversation, name)
            print(f"SAVED Saved: {path}")
            continue

        if user_input.lower().startswith("load "):
            name = user_input[5:].strip()
            loaded, msg = load_conversation(name)
            if loaded:
                client.conversation = loaded
                print(f"LOADED Loaded: {msg}")
            else:
                print(f"ERROR {msg}")
            continue

        if user_input.lower().startswith("temp "):
            try:
                temp = float(user_input[5:].strip())
                if 0.0 <= temp <= 2.0:
                    client.temperature = temp
                    print(f"THERMOMETER Temperature set to: {temp}")
                else:
                    print("WARNING Temperature must be from 0 to 2")
            except ValueError:
                print("WARNING Example: temp 0.7")
            continue

        # Process query
        try:
            start_time = datetime.now()
            response = await client.process_query(user_input)
            print(f"\nASSISTANT Assistant: {response}")

            elapsed = (datetime.now() - start_time).total_seconds()
            total_tools = len(client.mcp_client.available_tools) + len(
                client.docker_mcp_client.available_tools
            )
            print(
                f"TIME Time: {elapsed:.2f}s | Tools: {total_tools} (STDIO: {len(client.mcp_client.available_tools)}, Docker: {len(client.docker_mcp_client.available_tools)})"
            )

        except Exception as exc:
            error_msg = f"ERROR Error: {exc}"
            log.error(error_msg)
            print(error_msg)


# -------------------- MAIN FUNCTION --------------------
async def main() -> None:
    """Main entry point."""
    client = ChatClient(model_name=DEFAULT_MODEL)

    try:
        await client.start(SERVER_SCRIPT)
        await interactive_chat(client)

    except KeyboardInterrupt:
        print("\n\nSTOP Interrupted by user")
    except Exception as exc:
        error_msg = f"CRITICAL Critical error: {exc}"
        log.exception(error_msg)
        print(error_msg)
    finally:
        await client.cleanup()


if __name__ == "__main__":
    asyncio.run(main())
