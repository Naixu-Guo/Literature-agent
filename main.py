from dotenv import load_dotenv
load_dotenv()

from tools.toolset import toolset
import tools.calculator
import tools.literature_agent

# Try to import MCP tools if available
try:
    import tools.literature_agent_mcp_simple
    print("✓ Simple MCP tools loaded successfully")
except ImportError as e:
    print(f"⚠ MCP tools not available: {e}")

if __name__ == '__main__':
    toolset.serve(host="0.0.0.0", port=8001)