"""
Test script for MCP server
"""
import json
import urllib.request
import urllib.error

SERVER_URL = 'http://localhost:8080'

def test_mcp_method(method_name, params=None):
    """Send an MCP request to the server"""
    payload = {
        'method': method_name,
        'params': params or {}
    }
    
    req = urllib.request.Request(
        SERVER_URL,
        data=json.dumps(payload).encode('utf-8'),
        headers={'Content-Type': 'application/json'},
        method='POST'
    )
    
    try:
        with urllib.request.urlopen(req) as response:
            result = json.loads(response.read().decode('utf-8'))
            return result
    except urllib.error.HTTPError as e:
        print(f"HTTP Error {e.code}: {e.reason}")
        return {'error': f'HTTP {e.code}'}
    except Exception as e:
        print(f"Error: {str(e)}")
        return {'error': str(e)}

def main():
    print("Testing MCP Server...")

    # Test time/get
    print("\n1. Testing time/get:")
    result = test_mcp_method('tools/list')
    print(json.dumps(result, indent=2))

    # Test random/number
    print("\n2. Testing random/number:")
    result = test_mcp_method('random/number')
    print(json.dumps(result, indent=2))

    # Test weather/get
    print("\n3. Testing weather/get:")
    result = test_mcp_method('weather/get')
    print(json.dumps(result, indent=2))

    # Test tools/list
    print("\n4. Testing tools/list:")
    result = test_mcp_method('tools/list')
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    main()