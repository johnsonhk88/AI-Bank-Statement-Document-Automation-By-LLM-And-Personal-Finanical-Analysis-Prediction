# BankAI MCP server (stdio)

The MCP server is a **local stdio** process. It does not open a port. It forwards tools to the running BankAI HTTP API.

## Prerequisites

1. BankAI API on `http://localhost:8000` (`cd infra && docker compose up -d`).
2. A JWT:

```bash
TOKEN=$(curl -s -X POST http://localhost:8000/api/auth/login \
  -H 'Content-Type: application/json' \
  -d '{"email":"admin@bankai.local","password":"YOUR_PASSWORD"}' \
  | python3 -c 'import sys,json; print(json.load(sys.stdin)["access_token"])')
echo "$TOKEN"
```

## Run by hand

```bash
cd backend
export BANKAI_API_URL=http://localhost:8000
export BANKAI_TOKEN="$TOKEN"
PYTHONPATH=. python -m app.mcp
```

The process speaks MCP JSON-RPC with `Content-Length` framing on stdin/stdout.

## Claude Desktop / Cursor config

Add to the client MCP config (path varies; Claude Desktop: `claude_desktop_config.json`):

```json
{
  "mcpServers": {
    "bankai": {
      "command": "python",
      "args": ["-m", "app.mcp"],
      "cwd": "/ABS/PATH/TO/repo/backend",
      "env": {
        "PYTHONPATH": "/ABS/PATH/TO/repo/backend",
        "BANKAI_API_URL": "http://localhost:8000",
        "BANKAI_TOKEN": "paste-jwt-here"
      }
    }
  }
}
```

JWT TTL is 15 minutes — refresh `BANKAI_TOKEN` when tools start returning 401.

## Tools

| Tool | Backend |
|------|---------|
| `list_documents` | `GET /api/documents` |
| `list_llm_models` | `GET /api/llm-models` |
| `create_analysis` | `POST /api/agent-runs` |
| `get_run` | `GET /api/agent-runs/{id}` |
| `get_analytics_summary` | `GET /api/analytics/summary` |
