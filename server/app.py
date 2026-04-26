"""
server/app.py — FastAPI Application (Email Triage RL, Grand Finale)
====================================================================
Port 7860 for HuggingFace Spaces.

ROUTING DESIGN
--------------
OpenEnv validator endpoints (registered by create_app, root-level):
  POST /reset          → start new episode, returns observation JSON
  POST /step           → send action, returns observation + reward + done
  GET  /state          → current environment state

Web playground UI (human-friendly, does NOT block validator):
  GET  /web            → full playground UI (HTML, dark theme)
  POST /web/reset      → stateful reset for playground session
  POST /web/step       → stateful step for playground session
  GET  /web/state      → playground session state

Monitoring endpoints:
  GET  /society/stats  → curriculum phase, scores, safety violations (JSON)
  GET  /society/log    → tail of safety_violations.jsonl (JSON)
  GET  /health         → liveness check (JSON)

  GET  /               → 301 redirect to /web (browsers only)

CRITICAL: The redirect at GET / does NOT affect POST /reset.
The validator uses POST /reset which is registered by create_app() at the
root level and is completely independent of the browser redirect.
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

# Ensure repo root is importable regardless of working directory
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from openenv.core.env_server.http_server import create_app
except ImportError as exc:
    raise ImportError(
        "openenv-core is required: pip install 'openenv-core[core]>=0.2.2'"
    ) from exc

from models import EmailTriageAction, EmailTriageObservation
from server.email_triage_environment import EmailTriageEnvironment

# Disable Gradio web interface — we provide our own
os.environ.setdefault("ENABLE_WEB_INTERFACE", "false")

# ---- Build OpenEnv FastAPI app — pass CLASS (not instance) ----
# create_app registers: POST /reset, POST /step, GET /state
app = create_app(
    EmailTriageEnvironment,
    EmailTriageAction,
    EmailTriageObservation,
    env_name="email_triage",
    max_concurrent_envs=1,
)

# ---- Shared stateful instance for /web/* and /society/* routes ----
# OpenEnv's /reset and /step are stateless per-request;
# the web playground needs a persistent session for interactive use.
_env = EmailTriageEnvironment()


# ---------------------------------------------------------------------------
# Monitoring endpoints
# ---------------------------------------------------------------------------


@app.get("/society/stats", tags=["monitoring"])
async def society_stats() -> JSONResponse:
    """
    Return curriculum phase, recent scores, and safety violation count.
    Visible at /society/stats — useful for live dashboard and judge demo.
    """
    curriculum = _env._curriculum
    stats: dict[str, Any] = {}

    if hasattr(curriculum, "get_stats"):
        # CurriculumManager (Grand Finale)
        stats = curriculum.get_stats()
    elif hasattr(curriculum, "current_difficulty"):
        # Legacy Curriculum fallback
        stats = {"tier": curriculum.current_difficulty.value}

    stats["episode_violations"] = len(_env.episode_violations)
    stats["current_task"] = (
        int(_env._current_task.task_id) if _env._current_task else None
    )
    stats["state_tier"] = _env.state.current_tier
    return JSONResponse(content=stats)


@app.get("/society/log", tags=["monitoring"])
async def violation_log(tail: int = 20) -> JSONResponse:
    """Tail the safety_violations.jsonl file (last N entries)."""
    log_path = Path("safety_violations.jsonl")
    if not log_path.exists():
        return JSONResponse(content={"violations": [], "total": 0})
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
    """Liveness check — always returns 200 if the server is up."""
    return JSONResponse(content={
        "status": "ok",
        "env": "email_triage",
        "port": 7860,
        "society_mode": os.environ.get("SOCIETY_MODE", "full"),
    })


# ---------------------------------------------------------------------------
# Web playground stateful routes
# ---------------------------------------------------------------------------


class WebStepRequest(BaseModel):
    action: Dict[str, Any]


@app.post("/web/reset", include_in_schema=False)
async def web_reset():
    """Reset the playground session and start a new episode."""
    obs = _env.reset()
    return {"observation": obs.model_dump(), "reward": obs.reward, "done": obs.done}


@app.post("/web/step", include_in_schema=False)
async def web_step(request: WebStepRequest = Body(...)):
    """Take a step in the playground session."""
    obs = _env.step(EmailTriageAction(**request.action))
    return {"observation": obs.model_dump(), "reward": obs.reward, "done": obs.done}


@app.get("/web/state", include_in_schema=False)
async def web_state():
    """Return current playground session state."""
    return _env.state.model_dump()


# ---------------------------------------------------------------------------
# Web playground UI (HTML, dark theme, agent society dashboard)
# ---------------------------------------------------------------------------

_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>📧 Email Triage — Agent Society</title>
<style>
  *{box-sizing:border-box;margin:0;padding:0}
  body{font-family:'Courier New',monospace;background:#0a0f1e;color:#e2e8f0;min-height:100vh;padding:20px}
  h1{color:#60a5fa;font-size:1.4em;margin-bottom:4px}
  .sub{color:#64748b;font-size:.8em;margin-bottom:18px;letter-spacing:.05em}
  .grid{display:grid;grid-template-columns:1fr 1fr;gap:14px;margin-bottom:14px}
  @media(max-width:700px){.grid{grid-template-columns:1fr}}
  .card{background:#111827;border:1px solid #1e293b;border-radius:10px;padding:16px}
  .card-title{color:#94a3b8;font-size:.75em;letter-spacing:.1em;text-transform:uppercase;margin-bottom:10px}
  button{background:#1d4ed8;color:#fff;border:none;padding:7px 13px;border-radius:6px;
         cursor:pointer;margin:2px 2px;font-family:inherit;font-size:.8em;transition:background .15s}
  button:hover{background:#2563eb}
  .btn-red{background:#991b1b}.btn-red:hover{background:#b91c1c}
  .btn-gray{background:#1e293b}.btn-gray:hover{background:#334155}
  textarea{width:100%;background:#0a0f1e;color:#e2e8f0;border:1px solid #1e293b;
           border-radius:6px;padding:10px;font-family:inherit;font-size:.82em;resize:vertical}
  pre{background:#0a0f1e;border:1px solid #1e293b;padding:12px;border-radius:6px;
      overflow:auto;white-space:pre-wrap;word-break:break-word;max-height:350px;
      font-size:.8em;line-height:1.5}
  .badge{display:inline-block;padding:2px 8px;border-radius:4px;font-size:.7em;font-weight:bold}
  .b-blue{background:#1e3a5f;color:#60a5fa}
  .b-green{background:#14532d;color:#4ade80}
  .b-red{background:#450a0a;color:#f87171}
  .b-yellow{background:#431407;color:#fb923c}
  .actions-row{display:flex;flex-wrap:wrap;gap:4px;margin-top:10px}
  #score-bar{height:6px;background:#1e293b;border-radius:3px;margin:8px 0;overflow:hidden}
  #score-fill{height:100%;background:linear-gradient(90deg,#1d4ed8,#7c3aed);width:0%;transition:width .5s}
  .metric{display:flex;justify-content:space-between;margin:4px 0;font-size:.8em}
  .metric-val{color:#60a5fa;font-weight:bold}
  .warn{color:#fb923c}
  .ok{color:#4ade80}
</style>
</head>
<body>
<h1>📧 Email Triage — Agent Society</h1>
<div class="sub">
  Grand Finale · OpenEnv · Port 7860 · Multi-Agent Debate Colony
  <span class="badge b-blue" style="margin-left:8px">LIVE</span>
</div>

<!-- Control Row -->
<div style="margin-bottom:14px">
  <button onclick="doReset()">🔄 New Episode</button>
  <button onclick="doHint()" class="btn-gray">💡 Hint</button>
  <button onclick="loadStats()">📊 Stats</button>
  <button onclick="loadLog()" class="btn-gray">🛡 Safety Log</button>
</div>

<!-- Main Grid -->
<div class="grid">
  <!-- Action Panel -->
  <div class="card">
    <div class="card-title">Action (send to environment)</div>
    <textarea id="action" rows="3">{"action": "classify", "priority": "high", "category": "billing"}</textarea>
    <div class="actions-row">
      <button onclick="doStep()">▶ Step</button>
      <button onclick="set({action:'flag_phishing'})" class="btn-red">🚨 Flag Phishing</button>
      <button onclick="set({action:'assign_queue',queue:'support'})" class="btn-gray">📥 Support</button>
      <button onclick="set({action:'assign_queue',queue:'billing'})" class="btn-gray">📥 Billing</button>
      <button onclick="set({action:'assign_queue',queue:'sales'})" class="btn-gray">📥 Sales</button>
      <button onclick="set({action:'assign_queue',queue:'it'})" class="btn-gray">📥 IT</button>
      <button onclick="set({action:'assign_queue',queue:'hr'})" class="btn-gray">📥 HR</button>
      <button onclick="set({action:'escalate',to:'manager',reason:'urgent'})" class="btn-gray">⬆ Escalate</button>
      <button onclick="set({action:'use_tool',tool:'crm',params:{}})" class="btn-gray">🔧 CRM</button>
      <button onclick="set({action:'use_tool',tool:'ticketing',params:{}})" class="btn-gray">🎫 Ticket</button>
      <button onclick="set({action:'use_tool',tool:'notification',params:{}})" class="btn-gray">🔔 Notify</button>
    </div>
  </div>

  <!-- Society Stats -->
  <div class="card">
    <div class="card-title">Agent Society Stats</div>
    <div id="score-bar"><div id="score-fill"></div></div>
    <div id="metrics">
      <div class="metric"><span>Phase</span><span class="metric-val" id="m-phase">—</span></div>
      <div class="metric"><span>Recent avg score</span><span class="metric-val" id="m-avg">—</span></div>
      <div class="metric"><span>Episodes</span><span class="metric-val" id="m-ep">—</span></div>
      <div class="metric"><span>Safety violations</span><span class="metric-val warn" id="m-viol">—</span></div>
      <div class="metric"><span>Phishing caught</span><span class="metric-val ok" id="m-phish">—</span></div>
      <div class="metric"><span>Fast-track streak</span><span class="metric-val" id="m-streak">—</span></div>
    </div>
  </div>
</div>

<!-- Output Grid -->
<div class="grid">
  <div class="card">
    <div class="card-title">Environment Response</div>
    <pre id="out">Click "New Episode" to start.</pre>
  </div>
  <div class="card">
    <div class="card-title">Raw Stats / Log</div>
    <pre id="stats">Click 📊 Stats or 🛡 Safety Log.</pre>
  </div>
</div>

<script>
const $ = id => document.getElementById(id);

async function post(url, body) {
  const r = await fetch(url, {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify(body)
  });
  return r.json();
}
async function get(url) {
  const r = await fetch(url);
  return r.json();
}

async function doReset() {
  const d = await post('/web/reset', {});
  show('out', d);
  const r = d.observation?.reward || 0;
  $('score-fill').style.width = (r * 100) + '%';
  await loadStats();
}

async function doStep() {
  let raw = $('action').value.trim();
  let actionPayload;
  try { actionPayload = JSON.parse(raw); }
  catch { actionPayload = {action: raw}; }
  const d = await post('/web/step', {action: actionPayload});
  show('out', d);
  const r = d.observation?.reward || d.reward || 0;
  $('score-fill').style.width = Math.min(r * 100, 100) + '%';
}

async function doHint() {
  const d = await post('/web/step', {action: {action: 'hint'}});
  show('out', d);
}

async function loadStats() {
  const d = await get('/society/stats');
  show('stats', d);
  // Update metrics panel
  const pname = d.phase_name || d.tier || d.phase || '—';
  $('m-phase').textContent = pname;
  $('m-avg').textContent = d.recent_avg !== undefined ? d.recent_avg.toFixed(3) : '—';
  $('m-ep').textContent = d.episode !== undefined ? d.episode : '—';
  $('m-viol').textContent = d.safety_violations !== undefined ? d.safety_violations : (d.episode_violations || '—');
  $('m-phish').textContent = d.phishing_caught !== undefined ? d.phishing_caught : '—';
  $('m-streak').textContent = d.fast_track_streak !== undefined ? d.fast_track_streak : '—';
  if (d.recent_avg !== undefined) {
    $('score-fill').style.width = (d.recent_avg * 100) + '%';
  }
}

async function loadLog() {
  const d = await get('/society/log?tail=15');
  show('stats', d);
}

function set(obj) {
  $('action').value = JSON.stringify(obj);
}

function show(id, d) {
  $(id).textContent = JSON.stringify(d, null, 2);
}

// Auto-refresh stats every 30s
setInterval(loadStats, 30000);
// Initial stats load
loadStats();
</script>
</body>
</html>"""


@app.get("/web", response_class=HTMLResponse, include_in_schema=False)
async def web_ui():
    """Web playground UI — interactive email triage with agent society dashboard."""
    return HTMLResponse(content=_HTML)


@app.get("/", response_class=RedirectResponse, include_in_schema=False)
async def root_redirect():
    """
    Browser convenience redirect: GET / → /web
    This only affects GET requests to /.
    The validator's POST /reset is NOT affected — it goes directly to /reset.
    """
    return RedirectResponse(url="/web")


# ---------------------------------------------------------------------------
# Entry point (called by CMD in Dockerfile and by pyproject.toml scripts)
# ---------------------------------------------------------------------------


def main(host: str = "0.0.0.0", port: int = 7860) -> None:
    """
    Start the Email Triage environment server.

    Called via:
        uvicorn server.app:app --host 0.0.0.0 --port 7860
        python -m server.app
    """
    import uvicorn
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
