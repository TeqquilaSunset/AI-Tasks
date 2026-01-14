# Support MCP Server - Documentation

## Overview

Support MCP Server provides Model Context Protocol tools for managing support tickets and users. It enables LLMs and AI assistants to interact with the support system through standardized MCP tools.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                   Support MCP Server                        │
│                 (support_mcp_server.py)                     │
└────────────────────┬────────────────────────────────────────┘
                     │
                     │ MCP Protocol (STDIO)
                     │
┌────────────────────▼────────────────────────────────────────┐
│              Ticket Service (ticket_service.py)             │
│         - CRUD operations                                   │
│         - Search & filtering                                │
│         - User management                                  │
└────────────────────┬────────────────────────────────────────┘
                     │
                     │ JSON I/O
                     │
┌────────────────────▼────────────────────────────────────────┐
│              data/tickets.json                              │
│         - Users data                                        │
│         - Tickets data                                      │
│         - Message history                                   │
└─────────────────────────────────────────────────────────────┘
```

## Installation

No additional installation required. The MCP server uses the same dependencies as the main project.

## Running the Server

### As STDIO Server (default)

```bash
python support_mcp_server.py
```

The server runs in STDIO mode for communication with MCP clients.

### Testing the Server

```bash
# Run automated tests
python test_support_mcp.py

# Interactive mode
python test_support_mcp.py --interactive

# List available tools only
python test_support_mcp.py --tools
```

## MCP Tools

### 1. create_ticket

Create a new support ticket.

**Parameters:**
- `user_id` (str): User ID (e.g., 'user_001')
- `subject` (str): Ticket subject/title
- `content` (str): Initial message describing the issue
- `priority` (str, optional): 'low', 'medium', or 'high' (default: 'medium')
- `category` (str, optional): Ticket category (default: 'general')

**Returns:**
Created ticket information with ID

**Example:**
```python
result = await call_tool("create_ticket", {
    "user_id": "user_001",
    "subject": "Cannot login to account",
    "content": "I get 'Invalid credentials' error",
    "priority": "high",
    "category": "authentication"
})
```

---

### 2. get_ticket

Get detailed information about a specific ticket including message history.

**Parameters:**
- `ticket_id` (str): Ticket ID (e.g., 'ticket_001')

**Returns:**
Complete ticket information with all messages and user details

**Example:**
```python
result = await call_tool("get_ticket", {
    "ticket_id": "ticket_001"
})
```

---

### 3. list_tickets

List tickets with optional filtering.

**Parameters:**
- `status` (str, optional): Filter by status - 'open', 'in_progress', 'closed'
- `user_id` (str, optional): Filter by user ID

**Returns:**
List of tickets matching the criteria

**Example:**
```python
# All open tickets
result = await call_tool("list_tickets", {
    "status": "open"
})

# Tickets for specific user
result = await call_tool("list_tickets", {
    "user_id": "user_001"
})

# All tickets (no filters)
result = await call_tool("list_tickets", {})
```

---

### 4. add_ticket_message

Add a message to an existing ticket.

**Parameters:**
- `ticket_id` (str): Ticket ID
- `content` (str): Message content
- `role` (str, optional): 'user' or 'agent' (default: 'agent')

**Returns:**
Confirmation with message details

**Example:**
```python
result = await call_tool("add_ticket_message", {
    "ticket_id": "ticket_001",
    "content": "Please try clearing your browser cache",
    "role": "agent"
})
```

---

### 5. update_ticket_status

Update the status of a ticket.

**Parameters:**
- `ticket_id` (str): Ticket ID
- `status` (str): New status - 'open', 'in_progress', or 'closed'

**Returns:**
Confirmation with updated ticket details

**Example:**
```python
result = await call_tool("update_ticket_status", {
    "ticket_id": "ticket_001",
    "status": "in_progress"
})
```

---

### 6. search_tickets

Search tickets by subject or message content.

**Parameters:**
- `query` (str): Search query

**Returns:**
List of matching tickets

**Example:**
```python
result = await call_tool("search_tickets", {
    "query": "authorization error"
})
```

---

### 7. get_user

Get information about a specific user.

**Parameters:**
- `user_id` (str): User ID

**Returns:**
User information and ticket summary

**Example:**
```python
result = await call_tool("get_user", {
    "user_id": "user_001"
})
```

---

### 8. list_users

List all users in the system.

**Parameters:**
None

**Returns:**
List of all users with basic information

**Example:**
```python
result = await call_tool("list_users", {})
```

---

### 9. get_support_stats

Get support system statistics.

**Parameters:**
None

**Returns:**
Overall statistics including tickets, users, and categories

**Example:**
```python
result = await call_tool("get_support_stats", {})
```

---

### 10. get_ticket_context

Get ticket context formatted for LLM consumption.

This provides a structured context including ticket info,
user details, and message history - useful for AI assistants
generating responses.

**Parameters:**
- `ticket_id` (str): Ticket ID

**Returns:**
Formatted ticket context for LLM

**Example:**
```python
result = await call_tool("get_ticket_context", {
    "ticket_id": "ticket_001"
})
```

## Integration Examples

### Example 1: AI Assistant with MCP Integration

```python
from src.clients.mcp_client import MCPClient

async def ai_support_assistant():
    # Connect to MCP server
    client = MCPClient()
    await client.connect_to_server("support_mcp_server.py")

    # Get open tickets
    tickets = await client.call_tool("list_tickets", {"status": "open"})

    # For each ticket, get context and generate response
    for ticket_line in tickets[0].text.split('\n'):
        if 'ticket_' in ticket_line:
            ticket_id = extract_ticket_id(ticket_line)

            # Get ticket context
            context = await client.call_tool("get_ticket_context", {
                "ticket_id": ticket_id
            })

            # Generate AI response using context
            response = await generate_ai_response(context)

            # Add response to ticket
            await client.call_tool("add_ticket_message", {
                "ticket_id": ticket_id,
                "content": response,
                "role": "agent"
            })

    await client.cleanup()
```

### Example 2: Automated Ticket Creation

```python
async def create_ticket_from_email(email_content, user_email):
    client = MCPClient()
    await client.connect_to_server("support_mcp_server.py")

    # Find user by email (would need tool for this)
    user_id = find_user_by_email(user_email)

    # Create ticket
    result = await client.call_tool("create_ticket", {
        "user_id": user_id,
        "subject": extract_subject(email_content),
        "content": email_content,
        "priority": "medium",
        "category": "email"
    })

    await client.cleanup()
    return result
```

### Example 3: Daily Support Report

```python
async def generate_daily_report():
    client = MCPClient()
    await client.connect_to_server("support_mcp_server.py")

    # Get statistics
    stats = await client.call_tool("get_support_stats", {})

    # Get open tickets
    open_tickets = await client.call_tool("list_tickets", {"status": "open"})

    # Generate report
    report = f"""
Daily Support Report
====================

{stats[0].text}

Open Tickets:
{open_tickets[0].text}
"""

    await client.cleanup()
    return report
```

## Configuration

The MCP server uses `data/tickets.json` by default. To change:

```python
# In support_mcp_server.py
ticket_service = TicketService("path/to/your/tickets.json")
```

## Logging

All logging goes to stderr (MCP requirement):

```
[INFO] support-mcp-server: Starting Support MCP server...
[INFO] support-mcp-server: Available tools: create_ticket, get_ticket, ...
[INFO] support-mcp-server: Creating ticket for user user_001: Login issue
```

## Error Handling

All tools return error messages in case of failures:

```python
result = await call_tool("get_ticket", {"ticket_id": "invalid_id"})
# Returns: "Ticket not found: invalid_id"
```

## Data Format

### Ticket Format
```json
{
  "id": "ticket_001",
  "user_id": "user_001",
  "subject": "Cannot login",
  "status": "open",
  "priority": "high",
  "category": "authentication",
  "created_at": "2024-03-10T08:30:00Z",
  "updated_at": "2024-03-10T09:15:00Z",
  "messages": [
    {
      "id": "msg_001",
      "role": "user",
      "content": "I cannot login",
      "timestamp": "2024-03-10T08:30:00Z"
    }
  ]
}
```

### User Format
```json
{
  "id": "user_001",
  "name": "Ivan Petrov",
  "email": "ivan@example.com",
  "company": "TechCorp",
  "tier": "premium",
  "created_at": "2024-01-15T10:00:00Z"
}
```

## Testing

### Run all tests:
```bash
python test_support_mcp.py
```

### Interactive testing:
```bash
python test_support_mcp.py --interactive
```

### List tools:
```bash
python test_support_mcp.py --tools
```

## MCP Protocol Compliance

- ✅ STDIO transport
- ✅ Tool discovery
- ✅ JSON-RPC messages
- ✅ Error handling
- ✅ Logging to stderr only

## Future Enhancements

- [ ] WebSocket transport support
- [ ] Authentication/authorization
- [ ] Webhook notifications
- [ ] File attachments
- [ ] Ticket assignments
- [ ] SLA tracking
- [ ] Email integration

## Troubleshooting

### Server not starting
```bash
# Check Python path
python support_mcp_server.py

# Check dependencies
pip install -r requirements.txt
```

### Tools not found
```bash
# Test server directly
python test_support_mcp.py --tools
```

### Data file not found
```bash
# Ensure data directory exists
mkdir -p data
ls -la data/tickets.json
```

## License

MIT License
