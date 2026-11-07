"""BankAI MCP stdio server (JSON-RPC + Content-Length framing).

    cd backend && PYTHONPATH=. python -m app.mcp

Requires a running BankAI API. Set:

    BANKAI_API_URL   default http://localhost:8000
    BANKAI_TOKEN     JWT from POST /api/auth/login
"""
from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.request

API = os.environ.get("BANKAI_API_URL", "http://localhost:8000").rstrip("/")
TOKEN = os.environ.get("BANKAI_TOKEN", "")


def _request(method: str, path: str, body: dict | None = None) -> dict | list | None:
    url = f"{API}{path}"
    data = None if body is None else json.dumps(body).encode()
    headers = {"Accept": "application/json"}
    if TOKEN:
        headers["Authorization"] = f"Bearer {TOKEN}"
    if data is not None:
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            raw = resp.read()
            return json.loads(raw.decode()) if raw else None
    except urllib.error.HTTPError as exc:
        err = exc.read().decode()[:1000]
        raise RuntimeError(f"HTTP {exc.code} {path}: {err}") from exc


TOOLS = [
    {
        "name": "list_documents",
        "description": "List uploaded bank-statement PDFs for the signed-in admin.",
        "inputSchema": {"type": "object", "properties": {}, "additionalProperties": False},
    },
    {
        "name": "list_llm_models",
        "description": "List LLM providers and models (local engines include live /v1/models).",
        "inputSchema": {"type": "object", "properties": {}, "additionalProperties": False},
    },
    {
        "name": "create_analysis",
        "description": "Start an analysis batch for already-uploaded document IDs.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "document_ids": {"type": "array", "items": {"type": "string"}},
                "question": {"type": "string"},
                "llm_provider_id": {"type": "string"},
                "llm_model_id": {"type": "string"},
                "agent": {"type": "string", "default": "crewai"},
            },
            "required": ["document_ids", "question", "llm_provider_id", "llm_model_id"],
        },
    },
    {
        "name": "get_run",
        "description": "Get an agent-run batch by id (status, markdown report, transactions).",
        "inputSchema": {
            "type": "object",
            "properties": {"run_id": {"type": "string"}},
            "required": ["run_id"],
        },
    },
    {
        "name": "get_analytics_summary",
        "description": "Quarterly or annual totals from stored transactions.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "year": {"type": "integer"},
                "quarter": {"type": "integer", "minimum": 1, "maximum": 4},
            },
            "required": ["year"],
        },
    },
]


def _call_tool(name: str, arguments: dict) -> str:
    if name == "list_documents":
        return json.dumps(_request("GET", "/api/documents?limit=50"), indent=2)
    if name == "list_llm_models":
        return json.dumps(_request("GET", "/api/llm-models"), indent=2)
    if name == "create_analysis":
        payload = {
            "document_ids": arguments["document_ids"],
            "question": arguments["question"],
            "llm_provider_id": arguments["llm_provider_id"],
            "llm_model_id": arguments["llm_model_id"],
            "agent": arguments.get("agent") or "crewai",
        }
        return json.dumps(_request("POST", "/api/agent-runs", payload), indent=2)
    if name == "get_run":
        return json.dumps(_request("GET", f"/api/agent-runs/{arguments['run_id']}"), indent=2)
    if name == "get_analytics_summary":
        q = f"?year={arguments['year']}"
        if arguments.get("quarter"):
            q += f"&quarter={arguments['quarter']}"
        return json.dumps(_request("GET", f"/api/analytics/summary{q}"), indent=2)
    raise ValueError(f"Unknown tool: {name}")


def _ok(id_, result: dict) -> dict:
    return {"jsonrpc": "2.0", "id": id_, "result": result}


def _handle(msg: dict) -> dict | None:
    mid = msg.get("id")
    method = msg.get("method")
    params = msg.get("params") or {}
    if method == "initialize":
        return _ok(mid, {
            "protocolVersion": "2024-11-05",
            "capabilities": {"tools": {}},
            "serverInfo": {"name": "bankai", "version": "0.1.0"},
        })
    if method == "notifications/initialized":
        return None
    if method == "tools/list":
        return _ok(mid, {"tools": TOOLS})
    if method == "tools/call":
        name = params.get("name")
        args = params.get("arguments") or {}
        try:
            text = _call_tool(name, args)
            return _ok(mid, {"content": [{"type": "text", "text": text}]})
        except Exception as exc:
            return _ok(mid, {"content": [{"type": "text", "text": f"error: {exc}"}], "isError": True})
    if method == "ping":
        return _ok(mid, {})
    if mid is None:
        return None
    return {"jsonrpc": "2.0", "id": mid, "error": {"code": -32601, "message": f"Unknown method {method}"}}


def _read_message() -> dict | None:
    headers: dict[str, str] = {}
    while True:
        line = sys.stdin.buffer.readline()
        if not line:
            return None
        if line in (b"\r\n", b"\n"):
            break
        decoded = line.decode("utf-8", errors="replace")
        if ":" not in decoded:
            continue
        key, value = decoded.split(":", 1)
        headers[key.strip().lower()] = value.strip()
    length = int(headers.get("content-length", "0") or 0)
    if length <= 0:
        return None
    body = sys.stdin.buffer.read(length)
    return json.loads(body.decode("utf-8"))


def _write_message(obj: dict) -> None:
    data = json.dumps(obj, separators=(",", ":")).encode("utf-8")
    sys.stdout.buffer.write(f"Content-Length: {len(data)}\r\n\r\n".encode("ascii") + data)
    sys.stdout.buffer.flush()


def main() -> None:
    while True:
        msg = _read_message()
        if msg is None:
            break
        reply = _handle(msg)
        if reply is not None:
            _write_message(reply)


if __name__ == "__main__":
    main()
