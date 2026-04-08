"""
FastAPI application for the Email Triage RL Environment.

Mirrors the reference app.py structure:
  - create_app() receives the Environment CLASS (not instance)
  - A separate shared _env instance handles stateful /web/* routes
  - main() is a callable entry point (required by pyproject.toml [project.scripts])
  - Port 7860 for HuggingFace Spaces
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict

from fastapi import Body
from fastapi.responses import HTMLResponse, RedirectResponse
from pydantic import BaseModel

try:
    from openenv.core.env_server.http_server import create_app
except ImportError as e:
    raise ImportError(
        "openenv-core is required. Run: pip install 'openenv-core[core]>=0.2.2'"
    ) from e

from models import EmailTriageAction, EmailTriageObservation
from server.email_triage_environment import EmailTriageEnvironment

# Force API-only mode (no Gradio)
os.environ["ENABLE_WEB_INTERFACE"] = "false"

# Build the OpenEnv FastAPI application — pass the CLASS, not an instance
app = create_app(
    EmailTriageEnvironment,
    EmailTriageAction,
    EmailTriageObservation,
    env_name="email_triage",
    max_concurrent_envs=1,
)

# ---------------------------------------------------------------------------
# Shared environment instance for stateful web playground routes
# (OpenEnv's /reset and /step are stateless per-request; /web/* needs persistence)
# ---------------------------------------------------------------------------

_env = EmailTriageEnvironment()


class WebStepRequest(BaseModel):
    action: Dict[str, Any]


@app.post("/web/reset", include_in_schema=False)
async def web_reset():
    obs = _env.reset()
    return {"observation": obs.model_dump(), "reward": obs.reward, "done": obs.done}


@app.post("/web/step", include_in_schema=False)
async def web_step(request: WebStepRequest = Body(...)):
    action = EmailTriageAction(**request.action)
    obs = _env.step(action)
    return {"observation": obs.model_dump(), "reward": obs.reward, "done": obs.done}


@app.get("/web/state", include_in_schema=False)
async def web_state():
    return _env.state.model_dump()


# ---------------------------------------------------------------------------
# Minimal web playground UI
# ---------------------------------------------------------------------------

_PLAYGROUND_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Email Triage RL — Playground</title>
<style>
  body { font-family: monospace; background: #0f172a; color: #e2e8f0; margin: 0; padding: 24px; }
  h1 { color: #60a5fa; margin-bottom: 4px; }
  .sub { color: #94a3b8; font-size: 0.85em; margin-bottom: 20px; }
  .card { background: #1e293b; border-radius: 8px; padding: 16px; margin: 12px 0; }
  button { background: #3b82f6; color: #fff; border: none; padding: 8px 16px;
           border-radius: 6px; cursor: pointer; margin: 4px 2px; font-family: monospace; }
  button:hover { background: #2563eb; }
  button.danger { background: #dc2626; }
  button.danger:hover { background: #b91c1c; }
  textarea { width: 98%; background: #0f172a; color: #e2e8f0; border: 1px solid #334155;
             border-radius: 6px; padding: 8px; font-family: monospace; resize: vertical; }
  pre { background: #0f172a; padding: 12px; border-radius: 6px; overflow-x: auto;
        white-space: pre-wrap; word-break: break-word; max-height: 500px; overflow-y: auto; }
  .label { color: #94a3b8; font-size: 0.85em; margin-bottom: 6px; }
  .badge { display:inline-block; padding: 2px 8px; border-radius: 4px; font-size: 0.8em; }
  .easy   { background: #166534; color: #bbf7d0; }
  .medium { background: #854d0e; color: #fef08a; }
  .hard   { background: #7f1d1d; color: #fecaca; }
</style>
</head>
<body>
<h1>📧 Email Triage RL — Playground</h1>
<div class="sub">OpenEnv Environment · Port 7860</div>

<div class="card">
  <button onclick="doReset()">🔄 Reset (new episode)</button>
  <button onclick="doHint()">💡 Hint</button>
</div>

<div class="card">
  <div class="label">Action JSON:</div>
  <textarea id="action" rows="3">{"action": "classify", "priority": "high", "category": "billing"}</textarea>
  <br><br>
  <button onclick="doStep()">▶ Step</button>
  <button class="danger" onclick="setAction({action:'flag_phishing'})">🚨 Flag Phishing</button>
  <button onclick="setAction({action:'assign_queue',queue:'support'})">📥 Support</button>
  <button onclick="setAction({action:'assign_queue',queue:'billing'})">📥 Billing</button>
  <button onclick="setAction({action:'assign_queue',queue:'sales'})">📥 Sales</button>
  <button onclick="setAction({action:'assign_queue',queue:'it'})">📥 IT</button>
  <button onclick="setAction({action:'assign_queue',queue:'hr'})">📥 HR</button>
  <button onclick="setAction({action:'escalate',to:'manager',reason:'urgent'})">⬆ Escalate</button>
  <button onclick="setAction({action:'use_tool',tool:'crm',params:{}})">🔧 CRM</button>
  <button onclick="setAction({action:'use_tool',tool:'ticketing',params:{}})">🎫 Ticket</button>
  <button onclick="setAction({action:'use_tool',tool:'notification',params:{}})">🔔 Notify</button>
</div>

<div class="card">
  <div class="label">Response:</div>
  <pre id="output">Click "Reset" to start a new episode.</pre>
</div>

<script>
async function post(url, body) {
  const r = await fetch(url, {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify(body)
  });
  return r.json();
}
async function doReset() {
  const d = await post('/web/reset', {});
  document.getElementById('output').textContent = JSON.stringify(d, null, 2);
}
async function doStep() {
  const action = document.getElementById('action').value;
  const d = await post('/web/step', {action: {action}});
  document.getElementById('output').textContent = JSON.stringify(d, null, 2);
}
async function doHint() {
  const d = await post('/web/step', {action: {action: 'hint'}});
  document.getElementById('output').textContent = JSON.stringify(d, null, 2);
}
function setAction(obj) {
  document.getElementById('action').value = JSON.stringify(obj);
}
</script>
</body>
</html>"""


@app.get("/web", response_class=HTMLResponse, include_in_schema=False)
async def web_ui():
    return HTMLResponse(content=_PLAYGROUND_HTML)


@app.get("/", response_class=RedirectResponse, include_in_schema=False)
async def root_redirect():
    return RedirectResponse(url="/web")


# ---------------------------------------------------------------------------
# Entry point (required by [project.scripts] in pyproject.toml)
# ---------------------------------------------------------------------------


def main(host: str = "0.0.0.0", port: int = 7860) -> None:
    """Start the Email Triage environment server.

    Called via:
        python -m server.app
        uv run server
    """
    import uvicorn

    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
