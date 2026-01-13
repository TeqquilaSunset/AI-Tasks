#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MCP Server for GitHub API - Provides tools for PR analysis via MCP
"""
from __future__ import annotations

import os
import sys
from typing import Any

from dotenv import load_dotenv
from github import Github, GithubException
from github.PullRequest import PullRequest
from mcp.server.fastmcp import FastMCP

# Add src to path for imports
sys.path.insert(0, str(os.path.join(os.path.dirname(__file__), "src")))

from src.utils.logging_config import setup_logging

# -------------------- LOGGING --------------------
log = setup_logging("github-mcp", output_stream="stderr")

load_dotenv()

# Initialize GitHub client
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
if not GITHUB_TOKEN:
    log.error("GITHUB_TOKEN не найден в окружении!")
    sys.exit(1)

github_client = Github(GITHUB_TOKEN)

# -------------------- FASTMCP SERVER --------------------
mcp = FastMCP("github")

# -------------------- GITHUB TOOLS --------------------


@mcp.tool()
async def get_pull_request(
    repo_name: str, pr_number: int
) -> str:
    """Получить информацию о Pull Request.

    Args:
        repo_name: Имя репозитория в формате "owner/repo"
        pr_number: Номер Pull Request

    Returns:
        Информация о PR в формате JSON
    """
    log.info(f"Getting PR #{pr_number} from {repo_name}")

    try:
        repo = github_client.get_repo(repo_name)
        pr = repo.get_pull(pr_number)

        data = {
            "number": pr.number,
            "title": pr.title,
            "description": pr.body or "",
            "author": pr.user.login,
            "state": pr.state,
            "base_branch": pr.base.ref,
            "head_branch": pr.head.ref,
            "changed_files": pr.changed_files,
            "additions": pr.additions,
            "deletions": pr.deletions,
            "commits": pr.commits,
            "url": pr.html_url,
            "created_at": pr.created_at.isoformat(),
            "updated_at": pr.updated_at.isoformat(),
        }

        import json

        return json.dumps(data, indent=2, ensure_ascii=False)

    except GithubException as e:
        error_msg = f"GitHub API Error: {e}"
        log.error(error_msg)
        return error_msg
    except Exception as e:
        error_msg = f"Error: {e}"
        log.exception(error_msg)
        return error_msg


@mcp.tool()
async def get_pr_diff(repo_name: str, pr_number: int) -> str:
    """Получить diff для Pull Request.

    Args:
        repo_name: Имя репозитория в формате "owner/repo"
        pr_number: Номер Pull Request

    Returns:
        Diff в unified format
    """
    log.info(f"Getting diff for PR #{pr_number} from {repo_name}")

    try:
        import httpx

        repo = github_client.get_repo(repo_name)
        pr = repo.get_pull(pr_number)

        diff_url = pr.diff_url
        async with httpx.AsyncClient() as client:
            response = await client.get(diff_url, timeout=30)
            response.raise_for_status()
            diff = response.text

        # Truncate if too large
        max_chars = 100000
        if len(diff) > max_chars:
            diff = diff[:max_chars] + "\n\n... (truncated)"

        log.info(f"Retrieved diff: {len(diff)} characters")
        return diff

    except GithubException as e:
        error_msg = f"GitHub API Error: {e}"
        log.error(error_msg)
        return error_msg
    except Exception as e:
        error_msg = f"Error: {e}"
        log.exception(error_msg)
        return error_msg


@mcp.tool()
async def get_pr_files(
    repo_name: str, pr_number: int, max_files: int = 50
) -> str:
    """Получить список измененных файлов в Pull Request.

    Args:
        repo_name: Имя репозитория в формате "owner/repo"
        pr_number: Номер Pull Request
        max_files: Максимальное количество файлов для возврата (по умолчанию 50)

    Returns:
        Список файлов с информацией об изменениях
    """
    log.info(
        f"Getting files for PR #{pr_number} from {repo_name} (max: {max_files})"
    )

    try:
        import json

        repo = github_client.get_repo(repo_name)
        pr = repo.get_pull(pr_number)

        files = pr.get_files()
        files_data = []

        for idx, file in enumerate(files):
            if idx >= max_files:
                break

            files_data.append(
                {
                    "filename": file.filename,
                    "status": file.status,  # added, modified, deleted
                    "additions": file.additions,
                    "deletions": file.deletions,
                    "changes": file.changes,
                    "patch": file.patch[:1000] if file.patch else "",  # Truncated patch
                }
            )

        result = {
            "total_files": pr.changed_files,
            "shown_files": len(files_data),
            "files": files_data,
        }

        return json.dumps(result, indent=2, ensure_ascii=False)

    except GithubException as e:
        error_msg = f"GitHub API Error: {e}"
        log.error(error_msg)
        return error_msg
    except Exception as e:
        error_msg = f"Error: {e}"
        log.exception(error_msg)
        return error_msg


@mcp.tool()
async def get_file_content(
    repo_name: str, file_path: str, ref: str | None = None
) -> str:
    """Получить содержимое файла из репозитория.

    Args:
        repo_name: Имя репозитория в формате "owner/repo"
        file_path: Путь к файлу в репозитории
        ref: Git ref (branch, tag, commit). По умолчанию - ветка по умолчанию

    Returns:
        Содержимое файла
    """
    log.info(f"Getting file {file_path} from {repo_name} @ {ref or 'default'}")

    try:
        repo = github_client.get_repo(repo_name)

        # Try to get file from default branch or specified ref
        if ref:
            content_file = repo.get_contents(file_path, ref=ref)
        else:
            content_file = repo.get_contents(file_path)

        if isinstance(content_file, list):
            return f"Error: {file_path} is a directory, not a file"

        # Decode content
        content = content_file.decoded_content.decode("utf-8")

        # Truncate if too large
        max_chars = 50000
        if len(content) > max_chars:
            content = content[:max_chars] + "\n\n... (truncated)"

        log.info(f"Retrieved file: {len(content)} characters")
        return content

    except GithubException as e:
        error_msg = f"GitHub API Error: {e}"
        log.error(error_msg)
        return error_msg
    except Exception as e:
        error_msg = f"Error: {e}"
        log.exception(error_msg)
        return error_msg


@mcp.tool()
async def get_pr_commits(
    repo_name: str, pr_number: int, max_commits: int = 20
) -> str:
    """Получить коммиты из Pull Request.

    Args:
        repo_name: Имя репозитория в формате "owner/repo"
        pr_number: Номер Pull Request
        max_commits: Максимальное количество коммитов (по умолчанию 20)

    Returns:
        Список коммитов с сообщениями
    """
    log.info(
        f"Getting commits for PR #{pr_number} from {repo_name} (max: {max_commits})"
    )

    try:
        import json

        repo = github_client.get_repo(repo_name)
        pr = repo.get_pull(pr_number)

        commits = pr.get_commits()
        commits_data = []

        for idx, commit in enumerate(commits):
            if idx >= max_commits:
                break

            commits_data.append(
                {
                    "sha": commit.sha[:7],
                    "message": commit.commit.message.split("\n")[0],
                    "author": commit.author.login if commit.author else "Unknown",
                    "date": commit.commit.author.date.isoformat(),
                }
            )

        result = {
            "total_commits": pr.commits,
            "shown_commits": len(commits_data),
            "commits": commits_data,
        }

        return json.dumps(result, indent=2, ensure_ascii=False)

    except GithubException as e:
        error_msg = f"GitHub API Error: {e}"
        log.error(error_msg)
        return error_msg
    except Exception as e:
        error_msg = f"Error: {e}"
        log.exception(error_msg)
        return error_msg


@mcp.tool()
async def create_pr_comment(
    repo_name: str, pr_number: int, comment_body: str
) -> str:
    """Создать комментарий к Pull Request.

    Args:
        repo_name: Имя репозитория в формате "owner/repo"
        pr_number: Номер Pull Request
        comment_body: Текст комментария

    Returns:
        Результат операции
    """
    log.info(f"Posting comment to PR #{pr_number} in {repo_name}")

    try:
        repo = github_client.get_repo(repo_name)
        pr = repo.get_pull(pr_number)

        comment = pr.create_issue_comment(comment_body)

        result = {
            "success": True,
            "comment_id": comment.id,
            "comment_url": comment.html_url,
        }

        import json

        return json.dumps(result, indent=2, ensure_ascii=False)

    except GithubException as e:
        error_msg = f"GitHub API Error: {e}"
        log.error(error_msg)
        return error_msg
    except Exception as e:
        error_msg = f"Error: {e}"
        log.exception(error_msg)
        return error_msg


@mcp.tool()
async def get_repo_info(repo_name: str) -> str:
    """Получить информацию о репозитории.

    Args:
        repo_name: Имя репозитория в формате "owner/repo"

    Returns:
        Информация о репозитории
    """
    log.info(f"Getting repo info for {repo_name}")

    try:
        import json

        repo = github_client.get_repo(repo_name)

        data = {
            "name": repo.name,
            "full_name": repo.full_name,
            "description": repo.description or "",
            "owner": repo.owner.login,
            "language": repo.language,
            "stars": repo.stargazers_count,
            "forks": repo.forks_count,
            "open_issues": repo.open_issues_count,
            "default_branch": repo.default_branch,
            "url": repo.html_url,
            "created_at": repo.created_at.isoformat(),
            "updated_at": repo.updated_at.isoformat(),
        }

        return json.dumps(data, indent=2, ensure_ascii=False)

    except GithubException as e:
        error_msg = f"GitHub API Error: {e}"
        log.error(error_msg)
        return error_msg
    except Exception as e:
        error_msg = f"Error: {e}"
        log.exception(error_msg)
        return error_msg


# -------------------- MAIN --------------------
if __name__ == "__main__":
    # Run MCP server
    import mcp.server.stdio

    log.info("Starting GitHub MCP server...")
    mcp.run(transport="stdio")
