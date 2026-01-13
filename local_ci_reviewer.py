#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Local CI Reviewer - Automated code review using RAG and LLM.

This script performs automated code review by:
1. Getting git diff against origin/master or origin/main
2. Using RAG to find relevant documentation and rules
3. Generating comprehensive review with LLM
4. Outputting results to console and markdown file
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import httpx
from dotenv import load_dotenv

# Add src to path for imports
sys.path.insert(0, str(os.path.join(os.path.dirname(__file__), "src")))

from src.config import (
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_OLLAMA_HOST,
    DEFAULT_QDRANT_HOST,
    DEFAULT_QDRANT_PORT,
)
from src.services import RAGService
from src.utils import setup_logging

load_dotenv()

log = setup_logging("local-ci", output_stream="stdout")


class LocalCIReviewer:
    """
    Local CI reviewer for automated code review.

    Performs comprehensive code review using git diff, RAG for documentation
    lookup, and LLM for generating review comments.
    """

    def __init__(
        self,
        base_branch: str = "origin/main",
        relevance_threshold: float = 0.3,
        top_k: int = 5,
    ):
        """
        Initialize the local CI reviewer.

        Args:
            base_branch: Base branch to compare against (origin/main or origin/master)
            relevance_threshold: RAG relevance threshold (default: 0.3)
            top_k: Number of RAG results to retrieve
        """
        self.base_branch = base_branch
        self.relevance_threshold = relevance_threshold
        self.top_k = top_k

        # Initialize RAG services for both documentation and code
        # Documentation service (README, docs, etc.)
        self.docs_rag_service = RAGService(
            collection_name="project_docs",  # Documentation
        )
        self.docs_rag_service.set_relevance_threshold(relevance_threshold)

        # Code service (indexed by functions/classes)
        self.code_rag_service = RAGService(
            collection_name="code_chunks",  # Code chunks
        )
        self.code_rag_service.set_relevance_threshold(relevance_threshold)

        log.info(f"Local CI Reviewer initialized with base_branch={base_branch}, threshold={relevance_threshold}")

    def run_git_command(self, command: List[str]) -> str:
        """
        Run a git command and return output.

        Args:
            command: Git command as list of strings

        Returns:
            Command output as string
        """
        try:
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                check=True,
                timeout=30,
                encoding='utf-8',
                errors='replace',
            )
            return result.stdout
        except subprocess.CalledProcessError as e:
            log.error(f"Git command failed: {' '.join(command)}")
            log.error(f"Error: {e.stderr}")
            return ""
        except subprocess.TimeoutExpired:
            log.error(f"Git command timed out: {' '.join(command)}")
            return ""

    def get_current_branch(self) -> str:
        """Get current git branch name."""
        return self.run_git_command(["git", "rev-parse", "--abbrev-ref", "HEAD"]).strip()

    def get_changed_files(self) -> List[str]:
        """
        Get list of changed files compared to base branch.

        Returns:
            List of changed file paths
        """
        log.info(f"Getting changed files compared to {self.base_branch}")

        # Check if base branch exists
        remote_exists = self.run_git_command(
            ["git", "rev-parse", "--verify", self.base_branch]
        ).strip()

        if not remote_exists:
            log.warning(f"Base branch {self.base_branch} not found, trying fallback options...")
            # Try alternative branches
            for fallback in ["origin/master", "origin/main", "main", "master"]:
                if self.run_git_command(["git", "rev-parse", "--verify", fallback]).strip():
                    self.base_branch = fallback
                    log.info(f"Using fallback base branch: {self.base_branch}")
                    break

        # Get changed files
        output = self.run_git_command(
            ["git", "diff", "--name-only", f"{self.base_branch}...HEAD"]
        )

        if not output:
            # Alternative: try with different syntax
            output = self.run_git_command(
                ["git", "diff", "--name-only", self.base_branch]
            )

        files = [f.strip() for f in output.split("\n") if f.strip() and not f.startswith(".venv")]

        log.info(f"Found {len(files)} changed files")
        return files

    def get_file_diff(self, file_path: str) -> str:
        """
        Get git diff for a specific file.

        Args:
            file_path: Path to the file

        Returns:
            Diff content as string
        """
        diff = self.run_git_command(
            ["git", "diff", f"{self.base_branch}...HEAD", "--", file_path]
        )

        if not diff:
            diff = self.run_git_command(
                ["git", "diff", self.base_branch, "--", file_path]
            )

        return diff

    def search_documentation(self, query: str) -> List[Dict[str, Any]]:
        """
        Search for relevant documentation using RAG.

        Searches in both project_docs (documentation) and code_chunks (code).

        Args:
            query: Search query

        Returns:
            List of relevant documentation chunks from both sources
        """
        all_results = []

        # Search in documentation (README, docs, etc.)
        try:
            log.info("Searching in project_docs...")
            docs_results = self.docs_rag_service.search_similar(query, top_k=self.top_k)
            all_results.extend(docs_results)
            log.info(f"Found {len(docs_results)} results from project_docs")
        except Exception as e:
            log.warning(f"Could not search project_docs: {e}")

        # Search in code chunks (functions, classes)
        try:
            log.info("Searching in code_chunks...")
            code_results = self.code_rag_service.search_similar(query, top_k=self.top_k)
            all_results.extend(code_results)
            log.info(f"Found {len(code_results)} results from code_chunks")
        except Exception as e:
            log.warning(f"Could not search code_chunks: {e}")

        # Sort by score and return top results
        all_results.sort(key=lambda x: x.get("score", 0), reverse=True)
        return all_results[:self.top_k * 2]  # Return more results for better context

    def generate_review_prompt(
        self,
        diff_content: str,
        docs_results: List[Dict[str, Any]],
        changed_files: List[str],
    ) -> str:
        """
        Generate LLM prompt for code review.

        Args:
            diff_content: Git diff content
            docs_results: Relevant documentation from RAG
            changed_files: List of changed files

        Returns:
            LLM prompt for review
        """
        # Build documentation context
        docs_context = []
        for i, doc in enumerate(docs_results):
            text = doc.get("text") or doc.get("full_text", "")
            source = doc.get("source_document", "Unknown")
            score = doc.get("score", 0.0)

            docs_context.append(
                f"### Documentation #{i + 1} (relevance: {score:.3f})\n"
                f"Source: {source}\n"
                f"{text[:1000]}..."
            )

        docs_section = "\n\n".join(docs_context) if docs_context else "No relevant documentation found."

        # Build prompt
        prompt = f"""Ты эксперт по коду, проводящий комплексную ревью изменений. ВСЁ РЕВЬЮ ДОЛЖНО БЫТЬ НА РУССКОМ ЯЗЫКЕ.

## Измененные файлы
{chr(10).join(f'- {f}' for f in changed_files)}

## Git Diff
```
{diff_content[:5000]}
```

## Релевантная документация и правила
{docs_section}

## Инструкции
Предоставь комплексное ревью кода на русском языке со следующими секциями:

### 1. Обзор (Summary)
Краткий обзор того, какие изменения были сделаны и их цель.

### 2. Сильные стороны (Strengths)
Что сделано хорошо в этих изменениях (качество кода, структура и т.д.)

### 3. Проблемы и опасения (Concerns & Issues)
Потенциальные проблемы, баги или области, требующие внимания. Будь конкретен:
- Уязвимости безопасности
- Проблемы производительности
- Логические ошибки
- Плохой код (code smell)

### 4. Рекомендации (Suggestions)
Конкретные рекомендации по улучшению:
- Структура кода
- Лучшие практики
- Возможности рефакторинга
- Обновление документации

### 5. Проверка документации (Documentation Check)
- Зафиксированы ли изменения в документации?
- Обновлен ли README если нужно?
- Присутствуют ли docstrings и понятны ли они?

### 6. Рекомендации по тестированию (Testing Considerations)
- Какие тесты следует добавить?
- Граничные случаи для покрытия
- Точки интеграции для проверки

Будь тщательным, но лаконичным. Указывай конкретные файлы и номера строк когда возможно.
"""

        return prompt

    def call_llm(self, prompt: str) -> str:
        """
        Call LLM API to generate review.

        Args:
            prompt: LLM prompt

        Returns:
            LLM response
        """
        api_key = os.getenv("OPENAI_API_KEY")
        base_url = os.getenv("OPENAI_BASE_URL")
        model = os.getenv("OPENAI_MODEL", "glm-4.5-air")

        if not api_key:
            return "Error: OPENAI_API_KEY not found in environment"

        try:
            client = httpx.Client(timeout=120.0)

            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            }

            payload = {
                "model": model,
                "messages": [
                    {
                        "role": "system",
                        "content": "Ты эксперт по ревью кода, предоставляющий тщательную и конструктивную обратную связь на русском языке.",
                    },
                    {"role": "user", "content": prompt},
                ],
                "temperature": 0.3,
                "max_tokens": 4096,
            }

            response = client.post(
                f"{base_url}/chat/completions",
                headers=headers,
                json=payload,
            )
            response.raise_for_status()

            result = response.json()
            return result["choices"][0]["message"]["content"]

        except Exception as e:
            log.error(f"Error calling LLM: {e}")
            return f"Error generating review: {e}"

    def format_review_markdown(
        self,
        review_content: str,
        changed_files: List[str],
        current_branch: str,
    ) -> str:
        """
        Format review as markdown.

        Args:
            review_content: LLM generated review
            changed_files: List of changed files
            current_branch: Current git branch

        Returns:
            Formatted markdown content
        """
        timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")

        md = f"""# Local CI Code Review

**Generated:** {timestamp}  \
**Branch:** `{current_branch}`  \
**Base:** `{self.base_branch}`  \
**Changed Files:** {len(changed_files)}

## Changed Files

"""
        for f in changed_files:
            md += f"- `{f}`\n"

        md += f"\n---\n\n{review_content}\n\n---\n\n"
        md += f"\n*Generated by Local CI Reviewer with RAG assistance*"

        return md

    def save_review(self, content: str, output_path: str = None) -> str:
        """
        Save review to markdown file.

        Args:
            content: Review content
            output_path: Output file path (optional)

        Returns:
            Path to saved file
        """
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"ci_review_{timestamp}.md"

        output_file = Path(output_path)
        output_file.write_text(content, encoding="utf-8")

        log.info(f"Review saved to: {output_file}")
        return str(output_file)

    def run_review(self, output_file: str = None) -> Dict[str, Any]:
        """
        Run complete CI review process.

        Args:
            output_file: Optional output file path

        Returns:
            Dictionary with review results
        """
        log.info("=" * 60)
        log.info("Starting Local CI Review")
        log.info("=" * 60)

        # Check if documentation is indexed
        docs_indexed = False
        code_indexed = False

        try:
            test_results = self.docs_rag_service.search_similar("test", top_k=1)
            if test_results:
                docs_indexed = True
        except:
            pass

        try:
            test_results = self.code_rag_service.search_similar("test", top_k=1)
            if test_results:
                code_indexed = True
        except:
            pass

        if not docs_indexed and not code_indexed:
            log.warning("⚠️  Warning: No documentation or code indexed.")
            log.warning("   Run 'python index_docs.py' to index documentation.")
            log.warning("   Run 'python index_code.py' to index code by functions/classes.")
            log.warning("   Continuing with review anyway...\n")
        elif not docs_indexed:
            log.warning("⚠️  Warning: Documentation not indexed.")
            log.warning("   Run 'python index_docs.py' for better results.")
            log.warning("   Code indexing is active.\n")
        elif not code_indexed:
            log.warning("⚠️  Warning: Code not indexed.")
            log.warning("   Run 'python index_code.py' for smarter code review.")
            log.warning("   Documentation indexing is active.\n")
        else:
            log.info("✓ Both documentation and code are indexed.\n")

        # Get changed files
        current_branch = self.get_current_branch()
        log.info(f"Current branch: {current_branch}")

        changed_files = self.get_changed_files()

        if not changed_files:
            log.warning("No changed files found compared to base branch")
            return {"success": False, "error": "No changes to review"}

        # Collect diffs
        log.info("Collecting diffs...")
        all_diffs = []
        for file_path in changed_files:
            if file_path.endswith(".py") or file_path.endswith(".md") or file_path.endswith(".txt"):
                diff = self.get_file_diff(file_path)
                if diff:
                    all_diffs.append(f"### File: {file_path}\n{diff}\n")

        if not all_diffs:
            log.warning("No code changes found in diff")
            return {"success": False, "error": "No code changes to review"}

        diff_content = "\n".join(all_diffs)

        # Search documentation
        log.info("Searching relevant documentation...")
        docs_results = self.search_documentation(diff_content[:2000])
        log.info(f"Found {len(docs_results)} relevant documentation chunks")

        # Generate review
        log.info("Generating review with LLM...")
        review_prompt = self.generate_review_prompt(
            diff_content,
            docs_results,
            changed_files,
        )

        review_content = self.call_llm(review_prompt)

        # Format and output
        formatted_review = self.format_review_markdown(
            review_content,
            changed_files,
            current_branch,
        )

        # Print to console
        print("\n" + "=" * 60)
        print("CODE REVIEW RESULTS")
        print("=" * 60)
        print(formatted_review)
        print("=" * 60)

        # Save to file
        saved_path = self.save_review(formatted_review, output_file)

        return {
            "success": True,
            "review_path": saved_path,
            "changed_files": changed_files,
            "branch": current_branch,
        }


def main():
    """Main entry point for local CI reviewer."""
    parser = argparse.ArgumentParser(
        description="Local CI Reviewer - Automated code review with RAG"
    )
    parser.add_argument(
        "--base",
        default="origin/main",
        help="Base branch to compare against (default: origin/main)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.3,
        help="RAG relevance threshold (default: 0.3)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of RAG results to retrieve (default: 5)",
    )
    parser.add_argument(
        "--output",
        help="Output markdown file path (default: auto-generated)",
    )

    args = parser.parse_args()

    # Run review
    reviewer = LocalCIReviewer(
        base_branch=args.base,
        relevance_threshold=args.threshold,
        top_k=args.top_k,
    )

    result = reviewer.run_review(output_file=args.output)

    if result["success"]:
        log.info(f"✅ Review complete: {result['review_path']}")
        sys.exit(0)
    else:
        log.error(f"❌ Review failed: {result.get('error', 'Unknown error')}")
        sys.exit(1)


if __name__ == "__main__":
    main()
