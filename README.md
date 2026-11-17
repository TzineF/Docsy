# Documentation QnA

Ask questions about a documentation URL. A LangGraph agent (Gemini 2.5 Flash) calls the
`fetch` MCP tool, reads the page, and answers using only that content. Wrapped in FastAPI.

## Setup

Dependencies are already pinned in `pyproject.toml` / `requirements.txt`:

```powershell
uv sync
# or
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

`.env` must contain your key:

```
GOOGLE_API_KEY=your-key-here
```

Optional overrides: `GEMINI_MODEL` (default `gemini-2.5-flash`), `AGENT_RECURSION_LIMIT`
(default `12`).

## Run

```powershell
.\.venv\Scripts\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Then:

- <http://127.0.0.1:8000> — test UI: paste a docs URL, ask a question, read the
  answer and expand the step-by-step trace
- <http://127.0.0.1:8000/docs> — Swagger UI
- <http://127.0.0.1:8000/health> — readiness check

> **Windows: don't use `--reload` or `--workers`.** With those flags uvicorn uses a
> `SelectorEventLoop`, which cannot spawn the MCP `fetch` subprocess, so tool discovery
> fails at startup. Plain single-process mode uses a `ProactorEventLoop` and works.

Stop the server with `Ctrl+C`.

## Endpoints

### `GET /health`

```json
{ "status": "ok", "model": "gemini-2.5-flash", "graph_ready": true }
```

### `POST /ask`

```json
{ "url": "https://example.com", "question": "What is the main heading on this page?" }
```

`question` is required, `url` is optional. When `url` is given the agent is instructed to
fetch that page before answering; without it the question is sent as-is (the agent can
still decide to fetch on its own).

Response:

```json
{
  "answer": "The main heading on the page is \"Example Domain\".",
  "trace": [
    { "type": "human", "content": "Documentation URL: ..." },
    { "type": "ai", "tool_calls": [{ "name": "fetch", "args": { "url": "https://example.com" } }] },
    { "type": "tool", "content": "Contents of https://example.com:\n..." },
    { "type": "ai", "content": "The main heading on the page is \"Example Domain\"." }
  ]
}
```

From the shell:

```powershell
curl.exe -s -X POST http://127.0.0.1:8000/ask `
  -H "Content-Type: application/json" `
  -d '{\"url\":\"https://example.com\",\"question\":\"What is the main heading on this page?\"}'
```

Errors: `503` if the graph is still starting, `500` with `{"detail": "Agent failed: ..."}`
when the model or fetch call fails.

## Other files

- `app/agent.py` — standalone CLI version of the same graph (`python app/agent.py`).
- `app/test.py` — raw MCP client that lists tools and fetches `https://example.com`.
