# MCP Server

A minimal Model Context Protocol (MCP) server that provides utility functions:

## Features

- Get current time
- Generate random numbers
- Get random weather conditions

## Available Methods

### `tools/list`
Returns a list of all available tools.

Example request:
```json
{
  "method": "tools/list",
  "params": {}
}
```

Response:
```json
{
  "result": {
    "tools": [
      {
        "name": "time/get",
        "description": "Get the current time",
        "parameters": {}
      },
      ...
    ]
  }
}
```

### `time/get`
Returns the current timestamp.

Example request:
```json
{
  "method": "time/get",
  "params": {}
}
```

Response:
```json
{
  "result": {
    "timestamp": "2025-12-16 10:30:45"
  }
}
```

### `random/number`
Generates a random integer between min and max values (defaults: 1-100).

Parameters:
- `min` (optional, default: 1)
- `max` (optional, default: 100)

Example request:
```json
{
  "method": "random/number",
  "params": {
    "min": 1,
    "max": 10
  }
}
```

Response:
```json
{
  "result": {
    "number": 7
  }
}
```

### `weather/get`
Returns a random weather condition (sunny, rainy, cloudy).

Example request:
```json
{
  "method": "weather/get",
  "params": {}
}
```

Response:
```json
{
  "result": {
    "condition": "sunny"
  }
}
```

## Running the Server

```bash
python server.py
```

The server will start on port 8080 by default.

## Testing

Run the test client:
```bash
python test_client.py
```

Note: Start the server first before running the test client.