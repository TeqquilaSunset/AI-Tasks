"""
Minimal MCP (Model Context Protocol) Server
Implements endpoints for:
- Current time
- Random number
- Random weather
"""

import json
import random
from datetime import datetime
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs


class MCPHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        """Handle POST requests for MCP protocol"""
        content_length = int(self.headers['Content-Length'])
        post_data = self.rfile.read(content_length).decode('utf-8')
        
        try:
            # Parse the JSON request
            request = json.loads(post_data)
            
            # Handle MCP protocol request
            response = self.handle_mcp_request(request)
            
            # Send response
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps(response).encode('utf-8'))
        except Exception as e:
            self.send_error(400, f"Invalid request: {str(e)}")

    def handle_mcp_request(self, request):
        """Process MCP requests based on method name"""
        method = request.get('method', '')

        if method == 'time/get':
            return self.get_current_time(request)
        elif method == 'random/number':
            return self.get_random_number(request)
        elif method == 'weather/get':
            return self.get_weather(request)
        elif method == 'tools/list':
            return self.list_tools(request)
        else:
            return {
                'error': {
                    'code': -32601,
                    'message': f'Method {method} not implemented'
                }
            }

    def get_current_time(self, request):
        """Return current time"""
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        return {
            'result': {
                'timestamp': current_time
            }
        }

    def get_random_number(self, request):
        """Return a random number between specified bounds (default 1-100)"""
        params = request.get('params', {})
        min_val = params.get('min', 1)
        max_val = params.get('max', 100)
        
        random_num = random.randint(min_val, max_val)
        return {
            'result': {
                'number': random_num
            }
        }

    def get_weather(self, request):
        """Return random weather condition"""
        weather_conditions = ['sunny', 'rainy', 'cloudy']
        random_weather = random.choice(weather_conditions)
        return {
            'result': {
                'condition': random_weather
            }
        }

    def list_tools(self, request):
        """Return a list of available tools/instruments"""
        tools = [
            {
                'name': 'time/get',
                'description': 'Get the current time',
                'parameters': {}
            },
            {
                'name': 'random/number',
                'description': 'Generate a random number between min and max (default 1-100)',
                'parameters': {
                    'min': {'type': 'integer', 'default': 1},
                    'max': {'type': 'integer', 'default': 100}
                }
            },
            {
                'name': 'weather/get',
                'description': 'Get a random weather condition (sunny, rainy, cloudy)',
                'parameters': {}
            }
        ]
        return {
            'result': {
                'tools': tools
            }
        }

    def do_GET(self):
        """Serve basic info about the server at root path"""
        if self.path == '/':
            self.send_response(200)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            
            html = """
            <html>
                <head><title>MCP Server</title></head>
                <body>
                    <h1>MCP Server Running</h1>
                    <p>Endpoints:</p>
                    <ul>
                        <li>POST / - MCP protocol handler</li>
                        <li>Methods: time/get, random/number, weather/get</li>
                    </ul>
                </body>
            </html>
            """
            self.wfile.write(html.encode('utf-8'))
        else:
            self.send_error(404, "Not Found")


def run_server(port=8080):
    """Start the MCP server"""
    server_address = ('', port)
    httpd = HTTPServer(server_address, MCPHandler)
    print(f"MCP Server starting on port {port}...")
    print("Available methods:")
    print("- time/get: Get current time")
    print("- random/number: Get random number (with optional min/max params)")
    print("- weather/get: Get random weather (sunny, rainy, cloudy)")
    httpd.serve_forever()


if __name__ == "__main__":
    run_server()