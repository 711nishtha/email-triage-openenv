"""
FastAPI Application — Email Triage RL (Grand Finale)
=====================================================
Port 7860 for HuggingFace Spaces.

Endpoints:
  POST /reset          OpenEnv reset
  POST /step           OpenEnv step
  GET  /state          Current state
  GET  /web            Web playground UI
  GET  /society/stats  Agent society + curriculum stats (JSON)
  GET  /society/log    Safety violation log tail (JSON)
  GET  /health         Liveness check
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict

from fastapi import Body
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from openenv.core.env_server.http_server import create_app
except ImportError as exc:
    raise ImportError(
        "openenv-core is required: pip install 'openenv-core[core]>=0.2.2'"
    ) from exc

from models import EmailTriageAction, EmailTriageObservation
from server.email_triage_environment import EmailTriageEnvironment

os.environ.setdefault("ENABLE_WEB_INTERFACE", "false")

# Build OpenEnv app — pass CLASS (not instance)
app = create_app(
    EmailTriageEnvironment,
    EmailTriageAction,
    EmailTriageObservation,
    env_name="email_triage",
    max_concurrent_envs=1,
)

# Shared stateful instance for /web/* and /society/* routes
_env = EmailTriageEnvironment()


# ---------------------------------------------------------------------------
# Society stats endpoint
# ---------------------------------------------------------------------------

@app.get("/society/stats", tags=["society"])
async def society_stats() -> JSONResponse:
    """Return curriculum phase, recent scores, safety violations count."""
    curriculum = _env._curriculum
    stats: dict[str, Any] = {}

    # CurriculumManager stats (new)
    if hasattr(curriculum, "get_stats"):
        stats = curriculum.get_stats()
    # Legacy Curriculum fallback
    elif hasattr(curriculum, "current_difficulty"):
        stats = {"tier": curriculum.current_difficulty.value}

    stats["episode_violations"] = len(_env.episode_violations)
    stats["current_task"] = (
        int(_env._current_task.task_id) if _env._current_task else None
    )
    stats["state_tier"] = _env.state.current_tier
    return JSONResponse(content=stats)


@app.get("/society/log", tags=["society"])
async def violation_log(tail: int = 20) -> JSONResponse:
    """Tail safety_violations.jsonl."""
    log_path = Path("safety_violations.jsonl")
    if not log_path.exists():
        return JSONResponse(content={"violations": []})
    lines = log_path.read_text().strip().splitlines()
    entries = []
    for line in lines[-tail:]:
        try:
            entries.append(json.loads(line))
        except json.JSONDecodeError:
            pass
    return JSONResponse(content={"violations": entries, "total": len(lines)})


@app.get("/health", tags=["ops"])
async def health() -> JSONResponse:
    return JSONResponse(content={"status": "ok", "env": "email_triage", "port": 7860})


# ---------------------------------------------------------------------------
# Web playground routes (stateful)
# ---------------------------------------------------------------------------

class WebStepRequest(BaseModel):
    action: Dict[str, Any]


@app.post("/web/reset", include_in_schema=False)
async def web_reset():
    obs = _env.reset()
    return {"observation": obs.model_dump(), "reward": obs.reward, "done": obs.done}


@app.post("/web/step", include_in_schema=False)
async def web_step(request: WebStepRequest = Body(...)):
    obs = _env.step(EmailTriageAction(**request.action))
    return {"observation": obs.model_dump(), "reward": obs.reward, "done": obs.done}


@app.get("/web/state", include_in_schema=False)
async def web_state():
    return _env.state.model_dump()


# ---------------------------------------------------------------------------
# Web playground UI
# ---------------------------------------------------------------------------

_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1.0">
<title>📧 Email Triage Agent Society</title>
<style>
  body{font-family:monospace;background:#0f172a;color:#e2e8f0;margin:0;padding:20px}
  h1{color:#60a5fa;margin-bottom:2px}
  .sub{color:#94a3b8;font-size:.85em;margin-bottom:16px}
  .row{display:flex;gap:12px;flex-wrap:wrap;margin:10px 0}
  .card{background:#1e293b;border-radius:8px;padding:14px;flex:1;min-width:260px}
  btn,button{background:#3b82f6;color:#fff;border:none;padding:7px 14px;border-radius:6px;
             cursor:pointer;margin:3px 2px;font-family:monospace;font-size:.85em}
  button:hover{background:#2563eb}
  .red{background:#dc2626}.red:hover{background:#b91c1c}
  textarea{width:98%;background:#0f172a;color:#e2e8f0;border:1px solid #334155;
           border-radius:6px;padding:8px;font-family:monospace;resize:vertical}
  pre{background:#0f172a;padding:10px;border-radius:6px;overflow:auto;
      white-space:pre-wrap;word-break:break-word;max-height:400px;font-size:.82em}
  .lbl{color:#94a3b8;font-size:.8em;margin-bottom:4px}
  .tag{display:inline-block;padding:2px 7px;border-radius:4px;font-size:.75em}
  .easy{background:#166534;color:#bbf7d0}.med{background:#854d0e;color:#fef08a}
  .hard{background:#7f1d1d;color:#fecaca}
</style>
<script>setTimeout(()=>location.reload(),60000);</script>
</head>
<body>
<h1>📧 Email Triage — Agent Society</h1>
<div class="sub">Grand Finale · OpenEnv · Port 7860 · Multi-Agent Debate Colony</div>

<div class="row">
  <div class="card" style="flex:0 0 auto">
    <button onclick="doReset()">🔄 Reset</button>
    <button onclick="doHint()">💡 Hint</button>
    <button onclick="loadStats()">📊 Stats</button>
  </div>
</div>

<div class="row">
  <div class="card">
    <div class="lbl">Action JSON (sent to environment):</div>
    <textarea id="action" rows="3">{"action": "classify", "priority": "high", "category": "billing"}</textarea>
    <div style="margin-top:8px">
      <button onclick="doStep()">▶ Step</button>
      <button class="red" onclick="set({action:'flag_phishing'})">🚨 Phishing</button>
      <button onclick="set({action:'assign_queue',queue:'support'})">📥 Support</button>
      <button onclick="set({action:'assign_queue',queue:'billing'})">📥 Billing</button>
      <button onclick="set({action:'assign_queue',queue:'sales'})">📥 Sales</button>
      <button onclick="set({action:'assign_queue',queue:'it'})">📥 IT</button>
      <button onclick="set({action:'assign_queue',queue:'hr'})">📥 HR</button>
      <button onclick="set({action:'escalate',to:'manager',reason:'urgent'})">⬆ Escalate</button>
      <button onclick="set({action:'use_tool',tool:'crm',params:{}})">🔧 CRM</button>
      <button onclick="set({action:'use_tool',tool:'ticketing',params:{}})">🎫 Ticket</button>
      <button onclick="set({action:'use_tool',tool:'notification',params:{}})">🔔 Notify</button>
    </div>
  </div>
</div>

<div class="row">
  <div class="card">
    <div class="lbl">Response:</div>
    <pre id="out">Click Reset to start.</pre>
  </div>
  <div class="card">
    <div class="lbl">Society Stats:</div>
    <pre id="stats">Click 📊 Stats to load.</pre>
  </div>
</div>

<script>
async function post(url,body){
  const r=await fetch(url,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});
  return r.json();
}
async function get(url){const r=await fetch(url);return r.json();}
async function doReset(){const d=await post('/web/reset',{});show('out',d);}
async function doStep(){
  const a=document.getElementById('action').value;
  const d=await post('/web/step',{action:{action:a}});show('out',d);
}
async function doHint(){const d=await post('/web/step',{action:{action:'hint'}});show('out',d);}
async function loadStats(){const d=await get('/society/stats');show('stats',d);}
function set(obj){document.getElementById('action').value=JSON.stringify(obj);}
function show(id,d){document.getElementById(id).textContent=JSON.stringify(d,null,2);}
</script>
</body>
</html>"""


@app.get("/web", response_class=HTMLResponse, include_in_schema=False)
async def web_ui():
    return HTMLResponse(content=_HTML)


@app.get("/", response_class=RedirectResponse, include_in_schema=False)
async def root():
    return RedirectResponse(url="/web")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(host: str = "0.0.0.0", port: int = 7860) -> None:
    """Start server. Called by [project.scripts] entry point."""
    import uvicorn
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
