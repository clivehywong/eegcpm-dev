# MCP Integration

## Purpose
Model Context Protocol (MCP) servers to assist **development workflow only**.
These are NOT part of the final eegcpm product.

## Planned MCP Servers

### 1. Agent Orchestrator MCP
**Purpose:** Coordinate multi-agent discussions programmatically

```
Tools:
- spawn_agent(model, prompt, persona) → agent_id
- get_agent_response(agent_id) → response
- broadcast_to_agents(message, agent_ids) → responses
- close_session(session_id) → summary
```

### 2. Shared Memory MCP
**Purpose:** Persistent context sharing between agents during planning/implementation

```
Tools:
- write_context(key, value, scope) → success
- read_context(key, scope) → value
- list_context(scope) → keys
- clear_context(scope) → success

Scopes: session, project, global
```

## MCP Server Template

```python
# tools/mcp/template_server.py
from mcp.server import Server
from mcp.types import Tool, TextContent

app = Server("template-server")

@app.list_tools()
async def list_tools():
    return [
        Tool(
            name="example_tool",
            description="Description of what this tool does",
            inputSchema={
                "type": "object",
                "properties": {
                    "param1": {"type": "string", "description": "..."},
                },
                "required": ["param1"]
            }
        )
    ]

@app.call_tool()
async def call_tool(name: str, arguments: dict):
    if name == "example_tool":
        result = do_something(arguments["param1"])
        return [TextContent(type="text", text=result)]

if __name__ == "__main__":
    import asyncio
    asyncio.run(app.run())
```

## Configuration

Add to project `.claude/settings.json`:

```json
{
  "mcpServers": {
    "agent-orchestrator": {
      "command": "python",
      "args": ["tools/mcp/orchestrator_server.py"]
    },
    "shared-memory": {
      "command": "python",
      "args": ["tools/mcp/memory_server.py"]
    }
  }
}
```

## Note
These MCP servers are **development tools only** - they help coordinate the multi-agent workflow for building eegcpm. The final eegcpm product is standalone software with no LLM dependencies.
