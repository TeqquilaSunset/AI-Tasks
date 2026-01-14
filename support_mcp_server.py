# -*- coding: utf-8 -*-
"""
Support MCP Server - MCP server for ticket and user management

All logging disabled to prevent interference with MCP protocol (STDIO).
"""

import sys
import logging
from pathlib import Path
from typing import Optional

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

# Disable all logging before imports
logging.disable(logging.CRITICAL)

from mcp.server.fastmcp import FastMCP
from src.services.ticket_service import TicketService

# Re-enable logging
logging.disable(logging.NOTSET)

# Initialize ticket service silently
logging.disable(logging.CRITICAL)
ticket_service = TicketService("data/tickets.json")
logging.disable(logging.NOTSET)

# Initialize FastMCP server
mcp = FastMCP("support-server")


@mcp.tool()
async def create_ticket(
    user_id: str,
    subject: str,
    content: str,
    priority: str = "medium",
    category: str = "general"
) -> str:
    """Create a new support ticket.

    Args:
        user_id: User ID (e.g., 'user_001')
        subject: Ticket subject/title
        content: Initial message describing the issue
        priority: Priority level - 'low', 'medium', or 'high'
        category: Ticket category

    Returns:
        Created ticket information with ID
    """
    try:
        ticket = ticket_service.create_ticket(
            user_id=user_id,
            subject=subject,
            content=content,
            priority=priority,
            category=category
        )

        return f"""Ticket created successfully!
Ticket ID: {ticket.id}
User: {user_id}
Subject: {subject}
Status: {ticket.status}
Priority: {ticket.priority}
Category: {category}
Created at: {ticket.created_at}

Use get_ticket('{ticket.id}') to view details."""

    except Exception as e:
        return f"Error creating ticket: {str(e)}"


@mcp.tool()
async def get_ticket(ticket_id: str) -> str:
    """Get detailed information about a specific ticket including message history.

    Args:
        ticket_id: Ticket ID (e.g., 'ticket_001')

    Returns:
        Complete ticket information with all messages
    """
    try:
        ticket = ticket_service.get_ticket(ticket_id)
        if not ticket:
            return f"Ticket not found: {ticket_id}"

        user = ticket_service.get_user(ticket.user_id)

        result = f"""Ticket Details:
{'=' * 60}
ID: {ticket.id}
Subject: {ticket.subject}
Status: {ticket.status}
Priority: {ticket.priority}
Category: {ticket.category}
Created: {ticket.created_at}
Updated: {ticket.updated_at}
"""

        if user:
            result += f"""
User Information:
  Name: {user.name}
  Email: {user.email}
  Company: {user.company}
  Tier: {user.tier}
"""

        result += f"""
Messages ({len(ticket.messages)}):
{'-' * 60}
"""

        for msg in ticket.messages:
            role_icon = "U" if msg["role"] == "user" else "A"
            result += f"\n[{role_icon}] {msg['role'].upper()} [{msg['timestamp']}]\n"
            result += f"    {msg['content']}\n"

        result += f"\n{'=' * 60}"

        return result

    except Exception as e:
        return f"Error fetching ticket: {str(e)}"


@mcp.tool()
async def list_tickets(
    status: Optional[str] = None,
    user_id: Optional[str] = None
) -> str:
    """List tickets with optional filtering.

    Args:
        status: Filter by status - 'open', 'in_progress', 'closed'
        user_id: Filter by user ID

    Returns:
        List of tickets matching the criteria
    """
    try:
        if user_id:
            tickets = ticket_service.get_user_tickets(user_id)
            title = f"Tickets for user {user_id}"
        else:
            tickets = ticket_service.get_all_tickets(status=status)
            title = f"All tickets" + (f" ({status})" if status else "")

        if not tickets:
            return "No tickets found."

        result = f"{title} ({len(tickets)}):\n"
        result += "=" * 80 + "\n\n"

        for ticket in tickets:
            user = ticket_service.get_user(ticket.user_id)
            user_name = user.name if user else "Unknown"

            status_icon = {
                "open": "[OPEN]",
                "in_progress": "[PROGRESS]",
                "closed": "[CLOSED]"
            }.get(ticket.status, "[?]")

            result += f"{status_icon} #{ticket.id} | {ticket.subject}\n"
            result += f"    User: {user_name} | {ticket.updated_at}\n"
            result += f"    Status: {ticket.status} | Priority: {ticket.priority} | Category: {ticket.category}\n"
            result += f"    Messages: {len(ticket.messages)}\n\n"

        return result

    except Exception as e:
        return f"Error listing tickets: {str(e)}"


@mcp.tool()
async def add_ticket_message(
    ticket_id: str,
    content: str,
    role: str = "agent"
) -> str:
    """Add a message to an existing ticket.

    Args:
        ticket_id: Ticket ID (e.g., 'ticket_001')
        content: Message content
        role: Message role - 'user' or 'agent'

    Returns:
        Confirmation with message details
    """
    try:
        message = ticket_service.add_message(
            ticket_id=ticket_id,
            content=content,
            role=role
        )

        return f"""Message added successfully!
Ticket: {ticket_id}
Message ID: {message.id}
Role: {message.role}
Timestamp: {message.timestamp}

Content: {content[:100]}{'...' if len(content) > 100 else ''}"""

    except Exception as e:
        return f"Error adding message: {str(e)}"


@mcp.tool()
async def update_ticket_status(
    ticket_id: str,
    status: str
) -> str:
    """Update the status of a ticket.

    Args:
        ticket_id: Ticket ID (e.g., 'ticket_001')
        status: New status - 'open', 'in_progress', or 'closed'

    Returns:
        Confirmation with updated ticket details
    """
    try:
        if status not in ["open", "in_progress", "closed"]:
            return "Invalid status. Must be one of: open, in_progress, closed"

        ticket = ticket_service.update_ticket_status(ticket_id, status)

        return f"""Ticket status updated successfully!
Ticket ID: {ticket.id}
New Status: {ticket.status}
Updated at: {ticket.updated_at}

Subject: {ticket.subject}"""

    except Exception as e:
        return f"Error updating status: {str(e)}"


@mcp.tool()
async def search_tickets(query: str) -> str:
    """Search tickets by subject or message content.

    Args:
        query: Search query (searches in ticket subjects and messages)

    Returns:
        List of matching tickets
    """
    try:
        results = ticket_service.search_tickets(query)

        if not results:
            return f"No tickets found matching: {query}"

        result = f"Search results for '{query}' ({len(results)} found):\n"
        result += "=" * 80 + "\n\n"

        for ticket in results:
            user = ticket_service.get_user(ticket.user_id)
            user_name = user.name if user else "Unknown"

            status_icon = {
                "open": "[OPEN]",
                "in_progress": "[PROGRESS]",
                "closed": "[CLOSED]"
            }.get(ticket.status, "[?]")

            result += f"{status_icon} #{ticket.id} | {ticket.subject}\n"
            result += f"    User: {user_name} | {ticket.status} | {ticket.priority}\n"
            result += f"    Updated: {ticket.updated_at}\n\n"

        return result

    except Exception as e:
        return f"Error searching tickets: {str(e)}"


@mcp.tool()
async def get_user(user_id: str) -> str:
    """Get information about a specific user.

    Args:
        user_id: User ID (e.g., 'user_001')

    Returns:
        User information and ticket summary
    """
    try:
        user = ticket_service.get_user(user_id)
        if not user:
            return f"User not found: {user_id}"

        tickets = ticket_service.get_user_tickets(user_id)

        result = f"""User Information:
{'=' * 60}
ID: {user.id}
Name: {user.name}
Email: {user.email}
Company: {user.company}
Tier: {user.tier}
Member since: {user.created_at}

Tickets Summary:
  Total: {len(tickets)}
  Open: {len([t for t in tickets if t.status == 'open'])}
  In Progress: {len([t for t in tickets if t.status == 'in_progress'])}
  Closed: {len([t for t in tickets if t.status == 'closed'])}
"""

        if tickets:
            result += f"\nRecent Tickets:\n"
            result += "-" * 60 + "\n"
            for ticket in tickets[:5]:
                status_icon = {
                    "open": "[OPEN]",
                    "in_progress": "[PROGRESS]",
                    "closed": "[CLOSED]"
                }.get(ticket.status, "[?]")
                result += f"{status_icon} #{ticket.id} | {ticket.subject}\n"
                result += f"    {ticket.status} | {ticket.updated_at}\n"

        result += f"\n{'=' * 60}"

        return result

    except Exception as e:
        return f"Error fetching user: {str(e)}"


@mcp.tool()
async def list_users() -> str:
    """List all users in the system.

    Returns:
        List of all users with basic information
    """
    try:
        users = ticket_service.get_all_users()

        if not users:
            return "No users found."

        result = f"All Users ({len(users)}):\n"
        result += "=" * 80 + "\n\n"

        for user in users:
            tickets = ticket_service.get_user_tickets(user.id)

            result += f"User: {user.name} ({user.id})\n"
            result += f"    Email: {user.email}\n"
            result += f"    Company: {user.company} | Tier: {user.tier}\n"
            result += f"    Tickets: {len(tickets)} total"
            result += f" ({len([t for t in tickets if t.status == 'open'])} open)\n\n"

        return result

    except Exception as e:
        return f"Error listing users: {str(e)}"


@mcp.tool()
async def get_support_stats() -> str:
    """Get support system statistics.

    Returns:
        Overall statistics including tickets, users, and categories
    """
    try:
        stats = ticket_service.get_statistics()

        result = f"""Support System Statistics:
{'=' * 60}
Users & Tickets:
  Total Users: {stats['total_users']}
  Total Tickets: {stats['total_tickets']}
  Open Tickets: {stats['open_tickets']}
  In Progress: {stats['in_progress_tickets']}
  Closed Tickets: {stats['closed_tickets']}

Categories:
"""
        for cat, count in stats['categories'].items():
            result += f"  - {cat}: {count}\n"

        if stats['total_tickets'] > 0:
            closed_rate = (stats['closed_tickets'] / stats['total_tickets']) * 100
            result += f"\nClosed Rate: {closed_rate:.1f}%\n"

        result += f"\n{'=' * 60}"

        return result

    except Exception as e:
        return f"Error fetching statistics: {str(e)}"


@mcp.tool()
async def get_ticket_context(ticket_id: str) -> str:
    """Get ticket context formatted for LLM consumption.

    This provides a structured context including ticket info,
    user details, and message history - useful for AI assistants
    generating responses.

    Args:
        ticket_id: Ticket ID (e.g., 'ticket_001')

    Returns:
        Formatted ticket context for LLM
    """
    try:
        context = ticket_service.get_ticket_context(ticket_id)

        if not context:
            return f"Ticket not found: {ticket_id}"

        result = f"""Ticket Context for LLM:
{'=' * 60}
Ticket: #{context.get('ticket_id')}
Subject: {context.get('subject')}
Status: {context.get('status')}
Priority: {context.get('priority')}
Category: {context.get('category')}

User:
  Name: {context.get('user', {}).get('name')}
  Email: {context.get('user', {}).get('email')}
  Company: {context.get('user', {}).get('company')}
  Tier: {context.get('user', {}).get('tier')}

Timeline:
  Created: {context.get('created_at')}
  Updated: {context.get('updated_at')}

Conversation History ({context.get('message_count')} messages):
{context.get('message_history')}

{'=' * 60}"""

        return result

    except Exception as e:
        return f"Error getting ticket context: {str(e)}"


def main():
    """Main entry point for MCP server."""
    mcp.run()


if __name__ == "__main__":
    main()
