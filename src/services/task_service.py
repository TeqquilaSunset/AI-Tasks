# -*- coding: utf-8 -*-
"""
Task Service - Jira-like task management system.

Provides comprehensive task management with:
- CRUD operations for tasks, projects, sprints
- Task filtering and search
- Priority-based recommendations
- Statistics and reporting
- RAG integration for intelligent recommendations
"""

import json
from dataclasses import dataclass, asdict, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, List, Dict, Any
from enum import Enum

from ..utils import setup_logging

log = setup_logging("task-service")


class TaskStatus(Enum):
    """Task status enumeration."""
    TODO = "todo"
    IN_PROGRESS = "in_progress"
    IN_REVIEW = "in_review"
    DONE = "done"
    CANCELLED = "cancelled"
    OPEN = "open"


class TaskPriority(Enum):
    """Task priority enumeration."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class TaskType(Enum):
    """Task type enumeration."""
    FEATURE = "feature"
    BUG = "bug"
    ENHANCEMENT = "enhancement"
    DOCUMENTATION = "documentation"
    OPTIMIZATION = "optimization"
    REFACTORING = "refactoring"


@dataclass
class Comment:
    """Comment data model."""
    id: str
    author: str
    content: str
    created_at: str


@dataclass
class Subtask:
    """Subtask data model."""
    id: str
    title: str
    status: str


@dataclass
class Task:
    """Task data model."""
    id: str
    project_id: str
    title: str
    description: str
    status: str
    priority: str
    type: str
    assignee: Optional[str]
    labels: List[str]
    created_at: str
    updated_at: str
    due_date: Optional[str]
    estimated_hours: Optional[int]
    story_points: Optional[int]
    subtasks: List[Dict[str, Any]]
    comments: List[Dict[str, Any]]


@dataclass
class Project:
    """Project data model."""
    id: str
    name: str
    description: str
    status: str


@dataclass
class Sprint:
    """Sprint data model."""
    id: str
    name: str
    status: str
    start_date: str
    end_date: str
    task_ids: List[str]


@dataclass
class TaskRecommendation:
    """Task recommendation data model."""
    task: Task
    reason: str
    priority_score: float
    estimated_impact: str


class TaskService:
    """
    Service for managing tasks with intelligent recommendations.

    Features:
    - Full CRUD for tasks, projects, sprints
    - Task filtering and search
    - Priority-based recommendations
    - Statistics and reporting
    - RAG-ready context generation
    """

    def __init__(self, data_path: str = "data/tasks.json"):
        """
        Initialize task service.

        Args:
            data_path: Path to JSON file with tasks data
        """
        self.data_path = Path(data_path)
        self.data: Dict[str, Any] = {
            "metadata": {},
            "projects": [],
            "tasks": [],
            "sprints": [],
            "users": []
        }
        self._load_data()

    def _load_data(self):
        """Load tasks data from JSON file."""
        try:
            if self.data_path.exists():
                with open(self.data_path, 'r', encoding='utf-8') as f:
                    self.data = json.load(f)
                log.info(f"Loaded {len(self.data['tasks'])} tasks from {self.data_path}")
            else:
                log.warning(f"Data file not found: {self.data_path}, creating new")
                self._save_data()
        except Exception as e:
            log.error(f"Error loading data: {e}")
            self.data = {
                "metadata": {
                    "version": "1.0",
                    "created_at": datetime.now().isoformat() + "Z"
                },
                "projects": [],
                "tasks": [],
                "sprints": [],
                "users": []
            }

    def _save_data(self):
        """Save tasks data to JSON file."""
        try:
            self.data_path.parent.mkdir(parents=True, exist_ok=True)
            self.data["metadata"]["updated_at"] = datetime.now().isoformat() + "Z"
            with open(self.data_path, 'w', encoding='utf-8') as f:
                json.dump(self.data, f, ensure_ascii=False, indent=2)
            log.debug(f"Saved data to {self.data_path}")
        except Exception as e:
            log.error(f"Error saving data: {e}")

    # -------------------- PROJECT OPERATIONS --------------------
    def get_all_projects(self) -> List[Project]:
        """Get all projects."""
        return [Project(**p) for p in self.data.get("projects", [])]

    def get_project(self, project_id: str) -> Optional[Project]:
        """Get project by ID."""
        for p in self.data.get("projects", []):
            if p["id"] == project_id:
                return Project(**p)
        return None

    def create_project(self, name: str, description: str = "") -> Project:
        """Create a new project."""
        project_id = f"proj_{len(self.data['projects']) + 1:03d}"
        project = {
            "id": project_id,
            "name": name,
            "description": description,
            "status": "active"
        }
        self.data["projects"].append(project)
        self._save_data()
        log.info(f"Created project {project_id}: {name}")
        return Project(**project)

    # -------------------- TASK CRUD OPERATIONS --------------------
    def get_all_tasks(
        self,
        status: Optional[str] = None,
        priority: Optional[str] = None,
        assignee: Optional[str] = None,
        project_id: Optional[str] = None
    ) -> List[Task]:
        """
        Get all tasks with optional filtering.

        Args:
            status: Filter by status
            priority: Filter by priority
            assignee: Filter by assignee
            project_id: Filter by project

        Returns:
            Filtered list of tasks
        """
        tasks = [Task(**t) for t in self.data.get("tasks", [])]

        # Apply filters
        if status:
            tasks = [t for t in tasks if t.status == status]
        if priority:
            tasks = [t for t in tasks if t.priority == priority]
        if assignee:
            tasks = [t for t in tasks if t.assignee == assignee]
        if project_id:
            tasks = [t for t in tasks if t.project_id == project_id]

        # Sort by priority and updated date
        priority_order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
        return sorted(
            tasks,
            key=lambda t: (priority_order.get(t.priority, 99), t.updated_at),
            reverse=True
        )

    def get_task(self, task_id: str) -> Optional[Task]:
        """Get task by ID."""
        for t in self.data.get("tasks", []):
            if t["id"] == task_id:
                return Task(**t)
        return None

    def create_task(
        self,
        title: str,
        description: str = "",
        priority: str = "medium",
        type: str = "feature",
        project_id: str = "proj_001",
        assignee: Optional[str] = None,
        labels: Optional[List[str]] = None,
        due_date: Optional[str] = None,
        estimated_hours: Optional[int] = None,
        story_points: Optional[int] = None
    ) -> Task:
        """
        Create a new task.

        Args:
            title: Task title
            description: Task description
            priority: Task priority (critical, high, medium, low)
            type: Task type
            project_id: Project ID
            assignee: Assignee username
            labels: List of labels
            due_date: Due date in ISO format
            estimated_hours: Estimated hours to complete
            story_points: Story points for agile planning

        Returns:
            Created task
        """
        task_num = len(self.data["tasks"]) + 1
        task_id = f"task_{task_num:03d}"
        now = datetime.now().isoformat() + "Z"

        task = {
            "id": task_id,
            "project_id": project_id,
            "title": title,
            "description": description,
            "status": "todo",
            "priority": priority,
            "type": type,
            "assignee": assignee,
            "labels": labels or [],
            "created_at": now,
            "updated_at": now,
            "due_date": due_date,
            "estimated_hours": estimated_hours,
            "story_points": story_points,
            "subtasks": [],
            "comments": []
        }

        self.data["tasks"].append(task)
        self._save_data()
        log.info(f"Created task {task_id}: {title}")
        return Task(**task)

    def update_task_status(self, task_id: str, status: str) -> Task:
        """Update task status."""
        for t in self.data.get("tasks", []):
            if t["id"] == task_id:
                t["status"] = status
                t["updated_at"] = datetime.now().isoformat() + "Z"
                self._save_data()
                log.info(f"Updated task {task_id} status to {status}")
                return Task(**t)
        raise ValueError(f"Task not found: {task_id}")

    def update_task_priority(self, task_id: str, priority: str) -> Task:
        """Update task priority."""
        for t in self.data.get("tasks", []):
            if t["id"] == task_id:
                t["priority"] = priority
                t["updated_at"] = datetime.now().isoformat() + "Z"
                self._save_data()
                log.info(f"Updated task {task_id} priority to {priority}")
                return Task(**t)
        raise ValueError(f"Task not found: {task_id}")

    def assign_task(self, task_id: str, assignee: str) -> Task:
        """Assign task to user."""
        for t in self.data.get("tasks", []):
            if t["id"] == task_id:
                t["assignee"] = assignee
                t["updated_at"] = datetime.now().isoformat() + "Z"
                self._save_data()
                log.info(f"Assigned task {task_id} to {assignee}")
                return Task(**t)
        raise ValueError(f"Task not found: {task_id}")

    def add_comment(self, task_id: str, author: str, content: str) -> Comment:
        """Add comment to task."""
        for t in self.data.get("tasks", []):
            if t["id"] == task_id:
                comment_num = len(t["comments"]) + 1
                comment = {
                    "id": f"comment_{task_id.split('_')[1]}{comment_num:03d}",
                    "author": author,
                    "content": content,
                    "created_at": datetime.now().isoformat() + "Z"
                }
                t["comments"].append(comment)
                t["updated_at"] = comment["created_at"]
                self._save_data()
                log.info(f"Added comment to task {task_id}")
                return Comment(**comment)
        raise ValueError(f"Task not found: {task_id}")

    def add_subtask(self, task_id: str, title: str) -> Subtask:
        """Add subtask to task."""
        for t in self.data.get("tasks", []):
            if t["id"] == task_id:
                subtask_num = len(t["subtasks"]) + 1
                subtask = {
                    "id": f"sub_{task_id.split('_')[1]}{subtask_num:03d}",
                    "title": title,
                    "status": "pending"
                }
                t["subtasks"].append(subtask)
                t["updated_at"] = datetime.now().isoformat() + "Z"
                self._save_data()
                log.info(f"Added subtask to task {task_id}")
                return Subtask(**subtask)
        raise ValueError(f"Task not found: {task_id}")

    def search_tasks(self, query: str) -> List[Task]:
        """Search tasks by title, description, or labels."""
        query_lower = query.lower()
        results = []

        for task_data in self.data.get("tasks", []):
            # Search in title
            if query_lower in task_data["title"].lower():
                results.append(Task(**task_data))
                continue

            # Search in description
            if query_lower in task_data.get("description", "").lower():
                results.append(Task(**task_data))
                continue

            # Search in labels
            if any(query_lower in label.lower() for label in task_data.get("labels", [])):
                results.append(Task(**task_data))
                continue

        return results

    # -------------------- INTELLIGENT RECOMMENDATIONS --------------------
    def get_priority_recommendations(
        self,
        limit: int = 5,
        assignee: Optional[str] = None
    ) -> List[TaskRecommendation]:
        """
        Get AI-powered task priority recommendations.

        Args:
            limit: Maximum number of recommendations
            assignee: Filter by assignee

        Returns:
            List of task recommendations with scores and reasons
        """
        # Get ALL active tasks (not done/cancelled)
        all_tasks = self.get_all_tasks(assignee=assignee)
        active_statuses = ["todo", "open", "in_progress", "in_review"]
        tasks = [t for t in all_tasks if t.status in active_statuses]

        recommendations = []
        now = datetime.now()

        for task in tasks:
            score = 0.0
            reasons = []

            # Base priority score
            priority_scores = {
                "critical": 100.0,
                "high": 75.0,
                "medium": 50.0,
                "low": 25.0
            }
            score += priority_scores.get(task.priority, 50.0)

            # Status bonus (already in progress = higher priority to finish)
            status_bonus = {
                "in_progress": 40.0,  # Continue what you started
                "open": 30.0,         # Open issues need attention
                "in_review": 20.0,    # Almost done
                "todo": 0.0
            }
            score += status_bonus.get(task.status, 0.0)
            if task.status == "in_progress":
                reasons.append("Уже в работе - важно завершить")
            elif task.status == "open":
                reasons.append("Открытая задача - требует внимания")

            # Type bonus (bugs > features > enhancements)
            if task.type == "bug" and task.priority in ["high", "critical"]:
                score += 50.0
                reasons.append("Критический баг")
            elif task.type == "bug":
                score += 30.0
                reasons.append("Баг")

            # Due date urgency (more nuanced)
            if task.due_date:
                try:
                    due_date = datetime.fromisoformat(task.due_date.replace('Z', '+00:00'))
                    days_until_due = (due_date - now.replace(tzinfo=due_date.tzinfo)).days

                    if days_until_due < 0:
                        score += 100.0  # OVERDUE - top priority
                        reasons.append(f"ПРОСРОЧЕНА на {abs(days_until_due)} дн.")
                    elif days_until_due == 0:
                        score += 70.0
                        reasons.append("Срок сегодня!")
                    elif days_until_due == 1:
                        score += 50.0
                        reasons.append("Срок завтра")
                    elif days_until_due <= 3:
                        score += 30.0
                        reasons.append(f"Срок {days_until_due} дня")
                    elif days_until_due <= 7:
                        score += 15.0
                        reasons.append("Срок на этой неделе")
                except Exception:
                    pass  # Invalid date format

            # Small tasks bonus (quick wins)
            if task.story_points and task.story_points <= 2:
                score += 10.0
                reasons.append("Быстрая победа")

            # Penalty for too many in-progress tasks
            if assignee and task.status == "in_progress":
                in_progress_count = len([t for t in tasks if t.assignee == assignee and t.status == "in_progress"])
                if in_progress_count >= 5:
                    score -= 20.0
                    reasons.append("Много задач в работе")

            # Format recommendations
            impact = "Критичный" if score >= 150 else "Высокий" if score >= 100 else "Средний" if score >= 60 else "Низкий"

            recommendations.append(TaskRecommendation(
                task=task,
                reason=f"Приоритет: {task.priority.upper()}" + (f", {', '.join(reasons)}" if reasons else ""),
                priority_score=score,
                estimated_impact=impact
            ))

        # Sort by score and limit
        recommendations.sort(key=lambda r: r.priority_score, reverse=True)
        return recommendations[:limit]

    def get_project_status_summary(self) -> Dict[str, Any]:
        """Get comprehensive project status summary."""
        tasks = self.data.get("tasks", [])

        total = len(tasks)
        by_status = {}
        by_priority = {}
        by_type = {}
        overdue = 0
        unassigned = 0

        now = datetime.now()

        for task in tasks:
            # By status
            status = task.get("status", "unknown")
            by_status[status] = by_status.get(status, 0) + 1

            # By priority
            priority = task.get("priority", "unknown")
            by_priority[priority] = by_priority.get(priority, 0) + 1

            # By type
            type_ = task.get("type", "unknown")
            by_type[type_] = by_type.get(type_, 0) + 1

            # Check overdue
            if task.get("due_date"):
                try:
                    due_date = datetime.fromisoformat(task["due_date"].replace('Z', '+00:00'))
                    if due_date < now.replace(tzinfo=due_date.tzinfo) and task.get("status") not in ["done", "cancelled"]:
                        overdue += 1
                except:
                    pass

            # Check unassigned
            if not task.get("assignee"):
                unassigned += 1

        return {
            "total_tasks": total,
            "by_status": by_status,
            "by_priority": by_priority,
            "by_type": by_type,
            "overdue": overdue,
            "unassigned": unassigned,
            "completion_rate": round((by_status.get("done", 0) / total * 100) if total > 0 else 0, 1)
        }

    def get_task_context_for_rag(self, task_id: str) -> str:
        """
        Get detailed task context for RAG processing.

        Returns formatted context about task for LLM analysis.
        """
        task = self.get_task(task_id)
        if not task:
            return ""

        project = self.get_project(task.project_id)
        project_name = project.name if project else "Unknown Project"

        # Format subtasks
        subtasks_text = "\n".join([
            f"  - [{st['status'].upper()}] {st['title']}"
            for st in task.subtasks
        ]) if task.subtasks else "Нет подзадач"

        # Format comments
        comments_text = "\n".join([
            f"  {c['author']} ({c['created_at']}): {c['content']}"
            for c in task.comments[-5:]  # Last 5 comments
        ]) if task.comments else "Нет комментариев"

        context = f"""
## Задача: {task.title}

**ID**: {task.id}
**Проект**: {project_name}
**Статус**: {task.status.upper()}
**Приоритет**: {task.priority.upper()}
**Тип**: {task.type.upper()}
**Исполнитель**: {task.assignee or 'Не назначен'}
**Оценка**: {task.story_points or 'N/A'} story points, {task.estimated_hours or 'N/A'} часов
**Срок**: {task.due_date or 'Не установлен'}

### Описание
{task.description}

### Метки
{', '.join(task.labels) if task.labels else 'Нет меток'}

### Подзадачи
{subtasks_text}

### Последние комментарии
{comments_text}

### Метаданные
**Создана**: {task.created_at}
**Обновлена**: {task.updated_at}
"""
        return context.strip()

    def get_all_tasks_context(self) -> str:
        """Get context about all tasks for RAG analysis."""
        summary = self.get_project_status_summary()
        tasks = self.get_all_tasks()

        context = f"""
## Статус проекта

**Всего задач**: {summary['total_tasks']}
**Выполнено**: {summary['by_status'].get('done', 0)} ({summary['completion_rate']}%)
**В работе**: {summary['by_status'].get('in_progress', 0)}
**К выполнению**: {summary['by_status'].get('todo', 0)}
**Просрочено**: {summary['overdue']}
**Без исполнителя**: {summary['unassigned']}

### По приоритетам
"""

        for priority, count in sorted(summary['by_priority'].items(), key=lambda x: x[1], reverse=True):
            context += f"- **{priority.upper()}**: {count}\n"

        context += "\n### Текущие задачи\n"

        for task in tasks[:10]:  # Top 10 tasks
            context += f"- **{task.id}**: {task.title} [{task.status}] [{task.priority}]\n"

        return context.strip()

    # -------------------- SPRINT OPERATIONS --------------------
    def get_all_sprints(self) -> List[Sprint]:
        """Get all sprints."""
        return [Sprint(**s) for s in self.data.get("sprints", [])]

    def create_sprint(
        self,
        name: str,
        start_date: str,
        end_date: str,
        task_ids: Optional[List[str]] = None
    ) -> Sprint:
        """Create a new sprint."""
        sprint_id = f"sprint_{len(self.data['sprints']) + 1:03d}"

        sprint = {
            "id": sprint_id,
            "name": name,
            "status": "planned",
            "start_date": start_date,
            "end_date": end_date,
            "task_ids": task_ids or []
        }

        self.data["sprints"].append(sprint)
        self._save_data()
        log.info(f"Created sprint {sprint_id}: {name}")
        return Sprint(**sprint)

    def get_sprint_tasks(self, sprint_id: str) -> List[Task]:
        """Get all tasks in a sprint."""
        sprint = next((s for s in self.data.get("sprints", []) if s["id"] == sprint_id), None)
        if not sprint:
            return []

        return [self.get_task(tid) for tid in sprint.get("task_ids", []) if self.get_task(tid)]
