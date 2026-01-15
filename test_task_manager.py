#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test script for Task Manager functionality.
"""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(os.path.join(os.path.dirname(__file__), "src")))

from src.services.task_service import TaskService

def test_task_service():
    """Test TaskService basic operations."""
    print("=" * 60)
    print("Task Manager Test")
    print("=" * 60)

    # Initialize service
    task_service = TaskService()
    print(f"\n1. Loaded {len(task_service.data['tasks'])} tasks")

    # Test getting all tasks
    tasks = task_service.get_all_tasks()
    print(f"2. Retrieved {len(tasks)} tasks")

    # Test filtering by priority
    high_priority = task_service.get_all_tasks(priority="high")
    print(f"3. High priority tasks: {len(high_priority)}")

    # Test filtering by status
    todo_tasks = task_service.get_all_tasks(status="todo")
    print(f"4. Todo tasks: {len(todo_tasks)}")

    # Test recommendations
    recommendations = task_service.get_priority_recommendations(limit=3)
    print(f"5. Top 3 recommendations:")
    for i, rec in enumerate(recommendations, 1):
        print(f"   {i}. {rec.task.title} [{rec.task.priority}] - {rec.reason[:50]}...")

    # Test project status
    summary = task_service.get_project_status_summary()
    print(f"\n6. Project Status:")
    print(f"   Total: {summary['total_tasks']}")
    print(f"   Done: {summary['by_status'].get('done', 0)} ({summary['completion_rate']}%)")
    print(f"   In Progress: {summary['by_status'].get('in_progress', 0)}")
    print(f"   Overdue: {summary['overdue']}")

    # Test search
    search_results = task_service.search_tasks("RAG")
    print(f"\n7. Search 'RAG': {len(search_results)} results")
    for task in search_results:
        print(f"   - {task.id}: {task.title}")

    # Test getting specific task
    if tasks:
        first_task = tasks[0]
        task_context = task_service.get_task_context_for_rag(first_task.id)
        print(f"\n8. Task context for {first_task.id}:")
        print(f"   {task_context[:100]}...")

    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)

if __name__ == "__main__":
    test_task_service()
