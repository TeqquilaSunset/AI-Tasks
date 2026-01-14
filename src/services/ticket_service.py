# -*- coding: utf-8 -*-
"""
Ticket Service for Support System

Provides CRUD operations for support tickets and users stored in JSON format.
"""

import json
import asyncio
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, asdict
from ..utils import setup_logging

log = setup_logging("ticket-service")


@dataclass
class User:
    """User data model."""
    id: str
    name: str
    email: str
    company: str
    tier: str
    created_at: str


@dataclass
class Message:
    """Message data model."""
    id: str
    role: str  # 'user' or 'agent'
    content: str
    timestamp: str


@dataclass
class Ticket:
    """Ticket data model."""
    id: str
    user_id: str
    subject: str
    status: str  # 'open', 'in_progress', 'closed'
    priority: str  # 'low', 'medium', 'high'
    category: str
    created_at: str
    updated_at: str
    messages: List[Dict[str, Any]]


class TicketService:
    """
    Service for managing support tickets and users.

    Provides:
    - Load/save tickets from JSON
    - CRUD operations for tickets
    - User management
    - Ticket search and filtering
    - Context retrieval for LLM
    """

    def __init__(self, data_path: str = "data/tickets.json"):
        """
        Initialize ticket service.

        Args:
            data_path: Path to JSON file with tickets data
        """
        self.data_path = Path(data_path)
        self.data: Dict[str, Any] = {"users": [], "tickets": []}
        self._load_data()

    def _load_data(self):
        """Load tickets data from JSON file."""
        try:
            if self.data_path.exists():
                with open(self.data_path, 'r', encoding='utf-8') as f:
                    self.data = json.load(f)
                log.info(f"Loaded {len(self.data['tickets'])} tickets for {len(self.data['users'])} users")
            else:
                log.warning(f"Data file not found: {self.data_path}, creating new")
                self.data = {"users": [], "tickets": []}
                self._save_data()
        except Exception as e:
            log.error(f"Error loading data: {e}")
            self.data = {"users": [], "tickets": []}

    def _save_data(self):
        """Save tickets data to JSON file."""
        try:
            self.data_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.data_path, 'w', encoding='utf-8') as f:
                json.dump(self.data, f, ensure_ascii=False, indent=2)
            log.debug(f"Saved data to {self.data_path}")
        except Exception as e:
            log.error(f"Error saving data: {e}")

    def get_user(self, user_id: str) -> Optional[User]:
        """
        Get user by ID.

        Args:
            user_id: User ID

        Returns:
            User object or None if not found
        """
        for user_data in self.data["users"]:
            if user_data["id"] == user_id:
                return User(**user_data)
        return None

    def get_all_users(self) -> List[User]:
        """Get all users."""
        return [User(**u) for u in self.data["users"]]

    def get_ticket(self, ticket_id: str) -> Optional[Ticket]:
        """
        Get ticket by ID.

        Args:
            ticket_id: Ticket ID

        Returns:
            Ticket object or None if not found
        """
        for ticket_data in self.data["tickets"]:
            if ticket_data["id"] == ticket_id:
                return Ticket(**ticket_data)
        return None

    def get_all_tickets(self, status: Optional[str] = None) -> List[Ticket]:
        """
        Get all tickets, optionally filtered by status.

        Args:
            status: Filter by status ('open', 'in_progress', 'closed')

        Returns:
            List of tickets
        """
        tickets = [Ticket(**t) for t in self.data["tickets"]]
        if status:
            tickets = [t for t in tickets if t.status == status]
        return sorted(tickets, key=lambda x: x.updated_at, reverse=True)

    def get_user_tickets(self, user_id: str) -> List[Ticket]:
        """
        Get all tickets for a specific user.

        Args:
            user_id: User ID

        Returns:
            List of user's tickets
        """
        tickets = [
            Ticket(**t) for t in self.data["tickets"]
            if t["user_id"] == user_id
        ]
        return sorted(tickets, key=lambda x: x.updated_at, reverse=True)

    def search_tickets(self, query: str) -> List[Ticket]:
        """
        Search tickets by subject or message content.

        Args:
            query: Search query

        Returns:
            List of matching tickets
        """
        query_lower = query.lower()
        results = []

        for ticket_data in self.data["tickets"]:
            # Search in subject
            if query_lower in ticket_data["subject"].lower():
                results.append(Ticket(**ticket_data))
                continue

            # Search in messages
            for msg in ticket_data.get("messages", []):
                if query_lower in msg["content"].lower():
                    results.append(Ticket(**ticket_data))
                    break

        return results

    def create_ticket(
        self,
        user_id: str,
        subject: str,
        content: str,
        priority: str = "medium",
        category: str = "general"
    ) -> Ticket:
        """
        Create a new ticket.

        Args:
            user_id: User ID
            subject: Ticket subject
            content: Initial message content
            priority: Priority level ('low', 'medium', 'high')
            category: Ticket category

        Returns:
            Created ticket
        """
        # Verify user exists
        user = self.get_user(user_id)
        if not user:
            raise ValueError(f"User not found: {user_id}")

        # Generate ticket ID
        ticket_num = len(self.data["tickets"]) + 1
        ticket_id = f"ticket_{ticket_num:03d}"

        # Create ticket
        now = datetime.utcnow().isoformat() + "Z"
        ticket = {
            "id": ticket_id,
            "user_id": user_id,
            "subject": subject,
            "status": "open",
            "priority": priority,
            "category": category,
            "created_at": now,
            "updated_at": now,
            "messages": [
                {
                    "id": f"msg_{len(self.data['tickets']) * 10 + 1:03d}",
                    "role": "user",
                    "content": content,
                    "timestamp": now
                }
            ]
        }

        self.data["tickets"].append(ticket)
        self._save_data()

        log.info(f"Created ticket {ticket_id} for user {user_id}")
        return Ticket(**ticket)

    def add_message(
        self,
        ticket_id: str,
        content: str,
        role: str = "agent"
    ) -> Message:
        """
        Add a message to a ticket.

        Args:
            ticket_id: Ticket ID
            content: Message content
            role: Message role ('user' or 'agent')

        Returns:
            Created message
        """
        ticket = self.get_ticket(ticket_id)
        if not ticket:
            raise ValueError(f"Ticket not found: {ticket_id}")

        # Generate message ID
        msg_num = len(ticket.messages) + 1
        message = {
            "id": f"msg_{ticket_id.split('_')[1]}{msg_num:02d}",
            "role": role,
            "content": content,
            "timestamp": datetime.utcnow().isoformat() + "Z"
        }

        # Find and update ticket
        for t in self.data["tickets"]:
            if t["id"] == ticket_id:
                t["messages"].append(message)
                t["updated_at"] = message["timestamp"]
                break

        self._save_data()
        log.info(f"Added {role} message to ticket {ticket_id}")
        return Message(**message)

    def update_ticket_status(
        self,
        ticket_id: str,
        status: str
    ) -> Ticket:
        """
        Update ticket status.

        Args:
            ticket_id: Ticket ID
            status: New status ('open', 'in_progress', 'closed')

        Returns:
            Updated ticket
        """
        for t in self.data["tickets"]:
            if t["id"] == ticket_id:
                t["status"] = status
                t["updated_at"] = datetime.utcnow().isoformat() + "Z"
                self._save_data()
                log.info(f"Updated ticket {ticket_id} status to {status}")
                return Ticket(**t)

        raise ValueError(f"Ticket not found: {ticket_id}")

    def get_ticket_context(self, ticket_id: str) -> Dict[str, Any]:
        """
        Get ticket context for LLM (user info + ticket history).

        Args:
            ticket_id: Ticket ID

        Returns:
            Dictionary with ticket context
        """
        ticket = self.get_ticket(ticket_id)
        if not ticket:
            return {}

        user = self.get_user(ticket.user_id)
        if not user:
            return {}

        # Format message history
        messages_text = "\n".join([
            f"{msg['role'].upper()}: {msg['content']}"
            for msg in ticket.messages
        ])

        return {
            "ticket_id": ticket.id,
            "subject": ticket.subject,
            "status": ticket.status,
            "priority": ticket.priority,
            "category": ticket.category,
            "created_at": ticket.created_at,
            "updated_at": ticket.updated_at,
            "user": {
                "id": user.id,
                "name": user.name,
                "email": user.email,
                "company": user.company,
                "tier": user.tier
            },
            "message_history": messages_text,
            "message_count": len(ticket.messages)
        }

    def detect_ticket_from_message(
        self,
        message: str,
        user_id: Optional[str] = None
    ) -> Optional[Ticket]:
        """
        Auto-detect relevant ticket from message content.

        Args:
            message: User message
            user_id: Optional user ID to filter tickets

        Returns:
            Most relevant ticket or None
        """
        # If user_id provided, search only user's tickets
        if user_id:
            user_tickets = self.get_user_tickets(user_id)
            # Prioritize open tickets
            open_tickets = [t for t in user_tickets if t.status == "open"]
            if open_tickets:
                # Return most recently updated open ticket
                return open_tickets[0]

        # Search by keywords in message
        keywords = ["т не работает", "проблема с", "ошибка", "не могу",
                   "problem", "error", "issue", "not working"]

        message_lower = message.lower()

        # Check if message mentions a specific problem
        for keyword in keywords:
            if keyword in message_lower:
                # Search tickets with similar issues
                similar_tickets = self.search_tickets(keyword.split()[0])
                if similar_tickets:
                    return similar_tickets[0]

        return None

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get support statistics.

        Returns:
            Dictionary with statistics
        """
        total_tickets = len(self.data["tickets"])
        open_tickets = len([t for t in self.data["tickets"] if t["status"] == "open"])
        in_progress = len([t for t in self.data["tickets"] if t["status"] == "in_progress"])
        closed = len([t for t in self.data["tickets"] if t["status"] == "closed"])

        # Category breakdown
        categories = {}
        for t in self.data["tickets"]:
            cat = t.get("category", "unknown")
            categories[cat] = categories.get(cat, 0) + 1

        return {
            "total_users": len(self.data["users"]),
            "total_tickets": total_tickets,
            "open_tickets": open_tickets,
            "in_progress_tickets": in_progress,
            "closed_tickets": closed,
            "categories": categories
        }
