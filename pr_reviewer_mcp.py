#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AI PR Reviewer using MCP for GitHub API integration
Uses GitHub MCP server to fetch PR data and RAG for context
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import httpx
import openai
from dotenv import load_dotenv

# Add src to path for imports
sys.path.insert(0, str(os.path.join(os.path.dirname(__file__), "src")))

from src.clients import MCPClient
from src.config import DEFAULT_MODEL
from src.services.rag_service import RAGService
from src.utils.logging_config import setup_logging

load_dotenv()

log = setup_logging("pr-reviewer-mcp")


class PRReviewerMCP:
    """
    AI-powered PR reviewer using MCP and RAG.

    Uses GitHub MCP server to fetch PR data and RAG for documentation/code context.
    """

    def __init__(
        self,
        github_mcp_script: str = "github_mcp_server.py",
        openai_api_key: str | None = None,
        openai_base_url: str | None = None,
        model_name: str = DEFAULT_MODEL,
    ) -> None:
        """
        Initialize the PR reviewer with MCP integration.

        Args:
            github_mcp_script: Path to GitHub MCP server script
            openai_api_key: OpenAI API key (reads from env if None)
            openai_base_url: OpenAI base URL (reads from env if None)
            model_name: LLM model name
        """
        self.model_name = model_name

        # Initialize OpenAI client
        openai_api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        openai_base_url = openai_base_url or os.getenv("OPENAI_BASE_URL")

        if not openai_api_key:
            raise ValueError("OPENAI_API_KEY is required")

        http_client = httpx.AsyncClient(verify=False)
        self.openai_client = openai.AsyncOpenAI(
            api_key=openai_api_key,
            base_url=openai_base_url,
            http_client=http_client,
        )

        # Initialize MCP client for GitHub
        self.github_mcp = MCPClient()

        # Initialize RAG services
        self.doc_rag = RAGService(
            collection_name="project_docs",
            search_docs=False,
        )
        self.code_rag = RAGService(
            collection_name="pdf_chunks",
            search_docs=False,
        )

        self.github_mcp_script = github_mcp_script
        log.info("PR Reviewer MCP initialized")

    async def start(self) -> None:
        """Start MCP client and connect to GitHub server."""
        log.info("Connecting to GitHub MCP server...")
        await self.github_mcp.connect_to_server(self.github_mcp_script)
        log.info("Connected to GitHub MCP server")

    async def review_pr(self, repo_name: str, pr_number: int) -> str:
        """
        Review a pull request using MCP and RAG.

        Args:
            repo_name: Repository name (format: "owner/repo")
            pr_number: Pull request number

        Returns:
            Review text in markdown format
        """
        log.info(f"Reviewing PR #{pr_number} in {repo_name}")

        try:
            # Get PR info via MCP
            pr_info_result = await self.github_mcp.call_tool(
                "get_pull_request",
                {"repo_name": repo_name, "pr_number": pr_number},
            )
            pr_info = json.loads(pr_info_result)

            if "Error" in pr_info_result:
                return f"## Error\n\nFailed to get PR info: {pr_info_result}"

            # Get diff via MCP
            diff_result = await self.github_mcp.call_tool(
                "get_pr_diff",
                {"repo_name": repo_name, "pr_number": pr_number},
            )

            if "Error" in diff_result:
                diff = ""
            else:
                diff = diff_result

            # Get files via MCP
            files_result = await self.github_mcp.call_tool(
                "get_pr_files",
                {"repo_name": repo_name, "pr_number": pr_number, "max_files": 50},
            )

            # Parse files if successful
            files_data = []
            if "Error" not in files_result:
                try:
                    files_json = json.loads(files_result)
                    files_data = files_json.get("files", [])
                except json.JSONDecodeError:
                    pass

            # Get documentation context via RAG
            doc_context = await self._get_documentation_context(pr_info, diff)

            # Get code context via RAG
            code_context = await self._get_code_context(
                repo_name, pr_number, files_data
            )

            # Generate review
            review = await self._generate_review(
                pr_info=pr_info,
                diff=diff,
                files=files_data,
                doc_context=doc_context,
                code_context=code_context,
            )

            # Post comment via MCP if enabled
            if os.getenv("POST_TO_GITHUB", "true").lower() == "true":
                await self._post_review_comment(
                    repo_name, pr_number, review
                )

            return review

        except Exception as exc:
            error_msg = f"Review failed: {exc}"
            log.exception(error_msg)
            return f"## Error\n\n{error_msg}"

    async def _get_documentation_context(
        self, pr_info: Dict[str, Any], diff: str
    ) -> List[str]:
        """Get relevant documentation using RAG.

        Args:
            pr_info: PR information
            diff: PR diff

        Returns:
            List of relevant documentation snippets
        """
        try:
            # Create search query
            query = f"{pr_info.get('title', '')} {pr_info.get('description', '')}"

            # Search documentation
            results = self.doc_rag.search(query, top_k=5)

            # Extract content
            context = []
            for result in results:
                if result.score > 0.5:
                    source = result.metadata.get("source", "unknown")
                    context.append(
                        f"[{source}] (relevance: {result.score:.2f})\n{result.content[:400]}"
                    )

            log.info(f"Found {len(context)} documentation snippets")
            return context

        except Exception as exc:
            log.error(f"Error getting documentation context: {exc}")
            return []

    async def _get_code_context(
        self, repo_name: str, pr_number: int, files_data: List[Dict]
    ) -> List[str]:
        """Get relevant code context using RAG and MCP.

        Args:
            repo_name: Repository name
            pr_number: PR number
            files_data: List of changed files

        Returns:
            List of relevant code snippets
        """
        try:
            # Extract file paths to search
            query_parts = []
            for file_info in files_data[:10]:
                filename = file_info.get("filename", "")
                # Extract function/class names from filename
                parts = filename.split("/")
                query_parts.extend(parts[-2:])

            query = " ".join(query_parts)

            if not query:
                return []

            # Search code
            results = self.code_rag.search(query, top_k=5)

            # Extract content
            context = []
            for result in results:
                if result.score > 0.5:
                    file_path = result.metadata.get("file", "unknown")
                    line = result.metadata.get("start_line", "?")
                    context.append(
                        f"[{file_path}:{line}] (relevance: {result.score:.2f})\n{result.content[:400]}"
                    )

            log.info(f"Found {len(context)} code snippets")
            return context

        except Exception as exc:
            log.error(f"Error getting code context: {exc}")
            return []

    async def _generate_review(
        self,
        pr_info: Dict[str, Any],
        diff: str,
        files: List[Dict],
        doc_context: List[str],
        code_context: List[str],
    ) -> str:
        """Generate PR review using LLM.

        Args:
            pr_info: PR information
            diff: PR diff
            files: List of changed files
            doc_context: Relevant documentation
            code_context: Relevant code

        Returns:
            Review text in markdown format
        """
        system_prompt = """You are an expert code reviewer. Analyze pull requests and provide constructive feedback.

Your review should include:
1. **Summary**: Brief overview of changes
2. **Strengths**: What was done well
3. **Concerns**: Potential issues, bugs, or problems
4. **Suggestions**: Specific improvements
5. **Documentation**: Check if docs need updates
6. **Testing**: Verify test coverage

Be specific and actionable. Reference exact code when providing feedback."""

        # Build context
        context_parts = [
            f"**PR #{pr_info.get('number')}**: {pr_info.get('title')}",
            f"**Author**: {pr_info.get('author')}",
            f"**Branch**: {pr_info.get('head_branch')} → {pr_info.get('base_branch')}",
            f"**Changes**: +{pr_info.get('additions')} -{pr_info.get('deletions')} files",
        ]

        if pr_info.get("description"):
            context_parts.append(f"\n### Description\n{pr_info['description']}")

        # Changed files summary
        if files:
            context_parts.append("\n### Changed Files")
            for file_info in files[:10]:
                status_emoji = {
                    "added": "✅",
                    "modified": "📝",
                    "deleted": "❌",
                }.get(file_info.get("status", "modified"), "📝")
                context_parts.append(
                    f"- {status_emoji} `{file_info['filename']}` "
                    f"({file_info.get('status', 'modified')})"
                )

        # Documentation context
        if doc_context:
            context_parts.append("\n### Relevant Documentation")
            context_parts.extend(doc_context[:3])

        # Code context
        if code_context:
            context_parts.append("\n### Related Code")
            context_parts.extend(code_context[:3])

        # Diff (truncated)
        max_diff_chars = 15000
        truncated_diff = diff[:max_diff_chars]
        if len(diff) > max_diff_chars:
            truncated_diff += "\n\n... (diff truncated)"

        context_parts.append(f"\n### Diff\n```\n{truncated_diff}\n```")

        user_message = "\n".join(context_parts)

        try:
            response = await self.openai_client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message},
                ],
                temperature=0.3,
                max_tokens=4096,
            )

            review = response.choices[0].message.content or "No review generated."

            # Add header
            header = f"# 🤖 AI PR Review: #{pr_info.get('number')} - {pr_info.get('title')}\n\n"
            header += f"**Generated**: {datetime.now():%Y-%m-%d %H:%M:%S} UTC\n\n"
            header += "---\n\n"

            return header + review

        except Exception as exc:
            log.error(f"Error generating review: {exc}")
            return f"## Error\n\nFailed to generate review: {exc}"

    async def _post_review_comment(
        self, repo_name: str, pr_number: int, review: str
    ) -> None:
        """Post review as a comment on the PR using MCP.

        Args:
            repo_name: Repository name
            pr_number: PR number
            review: Review text
        """
        try:
            comment_body = (
                f"{review}\n\n---\n"
                f"*This review was generated automatically by AI PR Reviewer*"
            )

            result = await self.github_mcp.call_tool(
                "create_pr_comment",
                {
                    "repo_name": repo_name,
                    "pr_number": pr_number,
                    "comment_body": comment_body,
                },
            )

            if "Error" in result:
                log.error(f"Failed to post comment: {result}")
            else:
                log.info(f"Comment posted to PR #{pr_number}")

        except Exception as exc:
            log.error(f"Failed to post GitHub comment: {exc}")

    async def cleanup(self) -> None:
        """Clean up resources."""
        await self.github_mcp.cleanup()
        await self.openai_client.close()
        log.info("PR Reviewer MCP cleaned up")


async def main() -> None:
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="AI PR Reviewer with MCP")
    parser.add_argument("repo", help="Repository name (owner/repo)")
    parser.add_argument("pr", type=int, help="Pull request number")
    parser.add_argument(
        "--output",
        "-o",
        help="Output file (default: pr_review_output.md)",
        default="pr_review_output.md",
    )
    parser.add_argument(
        "--no-post",
        action="store_true",
        help="Don't post comment to GitHub",
    )

    args = parser.parse_args()

    # Set environment for posting
    if args.no_post:
        os.environ["POST_TO_GITHUB"] = "false"

    # Run review
    reviewer = PRReviewerMCP()

    try:
        await reviewer.start()
        review = await reviewer.review_pr(args.repo, args.pr)

        # Save review
        output_path = Path(args.output)
        output_path.write_text(review, encoding="utf-8")
        log.info(f"Review saved to {output_path}")

        # Print review
        print("\n" + "=" * 60)
        print("PR REVIEW")
        print("=" * 60 + "\n")
        print(review)

    finally:
        await reviewer.cleanup()


if __name__ == "__main__":
    asyncio.run(main())
