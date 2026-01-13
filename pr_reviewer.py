#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AI PR Reviewer - Analyzes Pull Requests using RAG and MCP
"""
from __future__ import annotations

import asyncio
import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import httpx
import openai
from dotenv import load_dotenv
from github import Github, GithubException
from github.PullRequest import PullRequest

# Add src to path for imports
sys.path.insert(0, str(os.path.join(os.path.dirname(__file__), "src")))

from src.config import (
    DEFAULT_COLLECTION_NAME,
    DEFAULT_DOCS_COLLECTION,
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_MODEL,
    DEFAULT_OLLAMA_HOST,
    DEFAULT_QDRANT_HOST,
    DEFAULT_QDRANT_PORT,
)
from src.services.rag_service import RAGService
from src.utils.logging_config import setup_logging

load_dotenv()

log = setup_logging("pr-reviewer")


class PRReviewer:
    """
    AI-powered PR reviewer using RAG for code analysis.

    Attributes:
        github_client: GitHub API client
        openai_client: OpenAI client for LLM
        rag_service: RAG service for documentation search
        code_rag_service: RAG service for code search
        model_name: LLM model to use
    """

    def __init__(
        self,
        github_token: str,
        openai_api_key: str,
        openai_base_url: str | None = None,
        model_name: str = DEFAULT_MODEL,
    ) -> None:
        """Initialize the PR reviewer.

        Args:
            github_token: GitHub API token
            openai_api_key: OpenAI API key
            openai_base_url: Custom OpenAI base URL
            model_name: LLM model name
        """
        self.github_client = Github(github_token)
        self.model_name = model_name

        # Initialize OpenAI client
        http_client = httpx.AsyncClient(verify=False)
        self.openai_client = openai.AsyncOpenAI(
            api_key=openai_api_key,
            base_url=openai_base_url,
            http_client=http_client,
        )

        # Initialize RAG services
        self.rag_service = RAGService(
            collection_name=DEFAULT_DOCS_COLLECTION,
            search_docs=False,  # Only search in docs collection
        )
        self.code_rag_service = RAGService(
            collection_name=DEFAULT_COLLECTION_NAME,
            search_docs=False,  # Only search in code collection
        )

        log.info("PR Reviewer initialized")

    async def review_pr(self, repo_name: str, pr_number: int) -> str:
        """
        Review a pull request and generate feedback.

        Args:
            repo_name: Repository name (format: "owner/repo")
            pr_number: Pull request number

        Returns:
            Review text in markdown format
        """
        log.info(f"Reviewing PR #{pr_number} in {repo_name}")

        try:
            # Get PR from GitHub
            repo = self.github_client.get_repo(repo_name)
            pr = repo.get_pull(pr_number)

            # Gather PR information
            pr_info = await self._gather_pr_info(pr)

            # Get diff
            diff = await self._get_pr_diff(pr)

            # Get relevant documentation via RAG
            doc_context = await self._get_documentation_context(pr_info, diff)

            # Get relevant code context via RAG
            code_context = await self._get_code_context(pr_info, diff)

            # Generate review
            review = await self._generate_review(
                pr_info=pr_info,
                diff=diff,
                doc_context=doc_context,
                code_context=code_context,
            )

            return review

        except GithubException as exc:
            error_msg = f"GitHub API error: {exc}"
            log.error(error_msg)
            return f"## Error\n\n{error_msg}"

        except Exception as exc:
            error_msg = f"Review failed: {exc}"
            log.exception(error_msg)
            return f"## Error\n\n{error_msg}"

    async def _gather_pr_info(self, pr: PullRequest) -> Dict[str, Any]:
        """Gather basic PR information.

        Args:
            pr: PullRequest object

        Returns:
            Dictionary with PR information
        """
        return {
            "number": pr.number,
            "title": pr.title,
            "description": pr.body or "",
            "author": pr.user.login,
            "base_branch": pr.base.ref,
            "head_branch": pr.head.ref,
            "changed_files": pr.changed_files,
            "additions": pr.additions,
            "deletions": pr.deletions,
            "url": pr.html_url,
        }

    async def _get_pr_diff(self, pr: PullRequest) -> str:
        """Get the diff for the PR.

        Args:
            pr: PullRequest object

        Returns:
            Diff string
        """
        try:
            # Get diff from GitHub API
            diff_url = pr.diff_url
            async with httpx.AsyncClient() as client:
                response = await client.get(diff_url, timeout=30)
                response.raise_for_status()
                diff = response.text

            # Truncate if too large
            max_chars = 50000
            if len(diff) > max_chars:
                diff = diff[:max_chars] + "\n\n... (truncated)"

            log.info(f"Retrieved diff: {len(diff)} characters")
            return diff

        except Exception as exc:
            log.error(f"Error getting diff: {exc}")
            return ""

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
            # Create search query from PR title and description
            query = f"{pr_info['title']} {pr_info['description']}"

            # Also include changed file paths
            file_patterns = re.findall(r"^\+\+\+ b/(.+)$", diff, re.MULTILINE)
            if file_patterns:
                query += " " + " ".join(file_patterns[:10])

            # Search documentation
            results = self.rag_service.search(query, top_k=5)

            # Extract content from results
            context = []
            for result in results:
                if result.score > 0.5:  # Only use relevant results
                    context.append(
                        f"[{result.metadata.get('source', 'unknown')}] "
                        f"(relevance: {result.score:.2f})\n{result.content[:500]}"
                    )

            log.info(f"Found {len(context)} relevant documentation snippets")
            return context

        except Exception as exc:
            log.error(f"Error getting documentation context: {exc}")
            return []

    async def _get_code_context(
        self, pr_info: Dict[str, Any], diff: str
    ) -> List[str]:
        """Get relevant code context using RAG.

        Args:
            pr_info: PR information
            diff: PR diff

        Returns:
            List of relevant code snippets
        """
        try:
            # Extract function/class names from diff
            functions = re.findall(r"^\+.*def\s+(\w+)", diff, re.MULTILINE)
            classes = re.findall(r"^\+.*class\s+(\w+)", diff, re.MULTILINE)

            query = " ".join(functions[:5] + classes[:5])

            if not query:
                # Use file paths as fallback
                file_patterns = re.findall(r"^\+\+\+ b/(.+)$", diff, re.MULTILINE)
                query = " ".join(file_patterns[:5])

            # Search code
            results = self.code_rag_service.search(query, top_k=5)

            # Extract content from results
            context = []
            for result in results:
                if result.score > 0.5:
                    context.append(
                        f"[{result.metadata.get('file', 'unknown')}:{result.metadata.get('line', '?')}] "
                        f"(relevance: {result.score:.2f})\n{result.content[:500]}"
                    )

            log.info(f"Found {len(context)} relevant code snippets")
            return context

        except Exception as exc:
            log.error(f"Error getting code context: {exc}")
            return []

    async def _generate_review(
        self,
        pr_info: Dict[str, Any],
        diff: str,
        doc_context: List[str],
        code_context: List[str],
    ) -> str:
        """Generate PR review using LLM.

        Args:
            pr_info: PR information
            diff: PR diff
            doc_context: Relevant documentation
            code_context: Relevant code

        Returns:
            Review text in markdown format
        """
        system_prompt = """You are an expert code reviewer. Analyze pull requests and provide constructive feedback.

Your review should:
1. **Summary**: Brief overview of what the PR changes
2. **Strengths**: What was done well
3. **Concerns**: Potential issues, bugs, or problems
4. **Suggestions**: Specific improvements
5. **Documentation**: Check if documentation needs updating
6. **Testing**: Verify if tests are adequate

Be specific and actionable. Reference exact code when providing feedback.
Use markdown formatting for readability."""

        # Build context
        context_parts = [
            f"PR Title: {pr_info['title']}",
            f"Author: {pr_info['author']}",
            f"Branch: {pr_info['head_branch']} -> {pr_info['base_branch']}",
            f"Changes: +{pr_info['additions']} -{pr_info['deletions']} files",
            f"URL: {pr_info['url']}",
        ]

        if pr_info["description"]:
            context_parts.append(f"\nDescription:\n{pr_info['description']}")

        if doc_context:
            context_parts.append("\n## Relevant Documentation\n")
            context_parts.extend(doc_context[:3])

        if code_context:
            context_parts.append("\n## Related Code\n")
            context_parts.extend(code_context[:3])

        context_parts.append(f"\n## Diff\n```\n{diff[:10000]}\n```")

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
            header = f"# PR Review: #{pr_info['number']} - {pr_info['title']}\n\n"
            header += f"**Generated:** {datetime.now():%Y-%m-%d %H:%M:%S} UTC\n\n"

            return header + review

        except Exception as exc:
            log.error(f"Error generating review: {exc}")
            return f"## Error\n\nFailed to generate review: {exc}"

    async def cleanup(self) -> None:
        """Clean up resources."""
        await self.openai_client.close()
        log.info("PR Reviewer cleaned up")


async def main() -> None:
    """Main entry point."""
    # Get environment variables
    github_token = os.getenv("GITHUB_TOKEN")
    openai_key = os.getenv("OPENAI_API_KEY")
    openai_base = os.getenv("OPENAI_BASE_URL")
    pr_number = os.getenv("PR_NUMBER")
    repo_name = os.getenv("REPO_NAME")

    # Validate
    if not all([github_token, openai_key, pr_number, repo_name]):
        missing = [
            name
            for name, value in [
                ("GITHUB_TOKEN", github_token),
                ("OPENAI_API_KEY", openai_key),
                ("PR_NUMBER", pr_number),
                ("REPO_NAME", repo_name),
            ]
            if not value
        ]
        log.error(f"Missing environment variables: {', '.join(missing)}")
        sys.exit(1)

    try:
        pr_number_int = int(pr_number)
    except ValueError:
        log.error(f"Invalid PR_NUMBER: {pr_number}")
        sys.exit(1)

    # Run review
    reviewer = PRReviewer(
        github_token=github_token,
        openai_api_key=openai_key,
        openai_base_url=openai_base,
    )

    try:
        review = await reviewer.review_pr(repo_name, pr_number_int)

        # Save review
        output_path = Path("pr_review_output.md")
        output_path.write_text(review, encoding="utf-8")
        log.info(f"Review saved to {output_path}")

        # Print review
        print("\n" + "=" * 60)
        print("PR REVIEW")
        print("=" * 60 + "\n")
        print(review)

        # Post comment to GitHub if needed
        if os.getenv("POST_TO_GITHUB", "true").lower() == "true":
            await _post_github_comment(github_token, repo_name, pr_number_int, review)

    finally:
        await reviewer.cleanup()


async def _post_github_comment(
    github_token: str, repo_name: str, pr_number: int, review: str
) -> None:
    """Post review as a comment on the PR.

    Args:
        github_token: GitHub API token
        repo_name: Repository name
        pr_number: PR number
        review: Review text
    """
    try:
        github_client = Github(github_token)
        repo = github_client.get_repo(repo_name)
        pr = repo.get_pull(pr_number)

        # Create issue comment (visible to all)
        comment_body = (
            f"## 🤖 AI PR Review\n\n{review}\n\n---\n"
            f"*This review was generated automatically by [AI PR Reviewer](https://github.com/)*"
        )

        pr.create_issue_comment(comment_body)
        log.info(f"Comment posted to PR #{pr_number}")

    except Exception as exc:
        log.error(f"Failed to post GitHub comment: {exc}")


if __name__ == "__main__":
    asyncio.run(main())
