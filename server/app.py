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
  GET  /web            → full playground UI (cream, editorial theme)
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

app = create_app(
    EmailTriageEnvironment,
    EmailTriageAction,
    EmailTriageObservation,
    env_name="email_triage",
    max_concurrent_envs=1,
)

_env = EmailTriageEnvironment()


# ---------------------------------------------------------------------------
# Monitoring endpoints  (unchanged)
# ---------------------------------------------------------------------------

@app.get("/society/stats", tags=["monitoring"])
async def society_stats() -> JSONResponse:
    curriculum = _env._curriculum
    stats: dict[str, Any] = {}
    if hasattr(curriculum, "get_stats"):
        stats = curriculum.get_stats()
    elif hasattr(curriculum, "current_difficulty"):
        stats = {"tier": curriculum.current_difficulty.value}
    stats["episode_violations"] = len(_env.episode_violations)
    stats["current_task"] = (
        int(_env._current_task.task_id) if _env._current_task else None
    )
    stats["state_tier"] = _env.state.current_tier
    return JSONResponse(content=stats)


@app.get("/society/log", tags=["monitoring"])
async def violation_log(tail: int = 20) -> JSONResponse:
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
    return JSONResponse(content={
        "status": "ok",
        "env": "email_triage",
        "port": 7860,
        "society_mode": os.environ.get("SOCIETY_MODE", "full"),
    })


# ---------------------------------------------------------------------------
# Web playground stateful routes  (unchanged)
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

_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Email Triage — Agent Society</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Cormorant+Garamond:ital,wght@0,400;0,500;0,600;1,400;1,500&family=DM+Mono:wght@400;500&family=DM+Sans:wght@300;400;500&display=swap" rel="stylesheet">
<style>
/* ── Tokens ─────────────────────────────────────────────── */
:root {
  --cream:     #f5f0e8;
  --cream-2:   #ede7d9;
  --cream-3:   #e3dace;
  --ink:       #1c1a17;
  --ink-2:     #3d3a35;
  --ink-muted: #7a7570;
  --rust:      #8b3a2a;
  --rust-lt:   #c4614d;
  --sage:      #4a5e4f;
  --sage-lt:   #6b8a72;
  --gold:      #9a7d3a;
  --gold-lt:   #c5a355;
  --amber:     #d97706;
  --border:    rgba(28,26,23,.11);
  --border-s:  rgba(28,26,23,.20);
  --shadow:    0 1px 8px rgba(28,26,23,.07);
  --shadow-lg: 0 6px 32px rgba(28,26,23,.11);
  --r:5px;
  --serif: 'Cormorant Garamond', Georgia, serif;
  --mono:  'DM Mono', 'Courier New', monospace;
  --sans:  'DM Sans', system-ui, sans-serif;
}

*,*::before,*::after{box-sizing:border-box;margin:0;padding:0}
html{scroll-behavior:smooth}

body{
  font-family:var(--sans);
  background:var(--cream);
  color:var(--ink);
  min-height:100vh;
  font-size:14px;
  line-height:1.6;
}

/* grain */
body::before{
  content:'';position:fixed;inset:0;pointer-events:none;z-index:0;
  background-image:url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='180' height='180'%3E%3Cfilter id='n'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.8' numOctaves='4' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='180' height='180' filter='url(%23n)' opacity='0.025'/%3E%3C/svg%3E");
}

.shell{max-width:1140px;margin:0 auto;padding:28px 20px 72px;position:relative;z-index:1}

/* ── Header ───────────────────────────────────────────────── */
.hdr{
  display:flex;align-items:flex-end;justify-content:space-between;
  flex-wrap:wrap;gap:12px;
  border-bottom:1px solid var(--border-s);
  padding-bottom:18px;margin-bottom:28px;
}
.hdr-title{font-family:var(--serif);font-size:1.9rem;font-weight:500;
  letter-spacing:-.02em;line-height:1.1;color:var(--ink)}
.hdr-title em{font-style:italic;color:var(--rust)}
.hdr-sub{font-family:var(--mono);font-size:.68rem;color:var(--ink-muted);
  letter-spacing:.07em;text-transform:uppercase;margin-top:4px;line-height:1.9}
.dot{display:inline-block;width:7px;height:7px;border-radius:50%;
  background:var(--sage);animation:blink 2s ease-in-out infinite;vertical-align:middle;margin-right:5px}
@keyframes blink{0%,100%{opacity:1}50%{opacity:.3}}
.pill{display:inline-flex;align-items:center;gap:4px;padding:2px 9px;border-radius:20px;
  font-family:var(--mono);font-size:.62rem;letter-spacing:.05em;text-transform:uppercase;
  border:1px solid currentColor}
.pl{color:var(--sage);background:rgba(74,94,79,.07)}
.pp{color:var(--gold);background:rgba(154,125,58,.07)}
.pw{color:var(--rust);background:rgba(139,58,42,.07)}

/* ── Tabs ─────────────────────────────────────────────────── */
.tabs{display:flex;gap:0;border-bottom:1px solid var(--border-s);margin-bottom:24px}
.tab{font-family:var(--sans);font-size:.8rem;font-weight:500;
  padding:9px 18px;cursor:pointer;border:none;background:transparent;
  color:var(--ink-muted);border-bottom:2px solid transparent;
  margin-bottom:-1px;transition:color .15s,border-color .15s}
.tab:hover{color:var(--ink)}
.tab.active{color:var(--rust);border-bottom-color:var(--rust)}

/* ── Section label ────────────────────────────────────────── */
.sec{font-family:var(--mono);font-size:.63rem;letter-spacing:.12em;
  text-transform:uppercase;color:var(--ink-muted);margin-bottom:8px;
  display:flex;align-items:center;gap:8px}
.sec::after{content:'';flex:1;height:1px;background:var(--border)}
.sec-n{font-family:var(--serif);font-style:italic;font-size:.9rem;
  color:var(--rust);min-width:1.4em}

/* ── Cards ────────────────────────────────────────────────── */
.card{background:#fff;border:1px solid var(--border);border-radius:var(--r);
  padding:18px;box-shadow:var(--shadow);transition:box-shadow .2s}
.card:hover{box-shadow:var(--shadow-lg)}
.card-title{font-family:var(--serif);font-size:1rem;font-weight:500;
  color:var(--ink);margin-bottom:12px;letter-spacing:-.01em}

.g2{display:grid;grid-template-columns:1fr 1fr;gap:14px}
.g3{display:grid;grid-template-columns:repeat(3,1fr);gap:10px}
@media(max-width:660px){.g2,.g3{grid-template-columns:1fr}}

/* ── Metric tiles ─────────────────────────────────────────── */
.mtile{background:var(--cream-2);border:1px solid var(--border);
  border-radius:var(--r);padding:12px 14px;transition:background .15s}
.mtile:hover{background:var(--cream-3)}
.ml{font-family:var(--mono);font-size:.6rem;letter-spacing:.1em;
  text-transform:uppercase;color:var(--ink-muted);margin-bottom:3px}
.mv{font-family:var(--serif);font-size:1.5rem;font-weight:600;
  color:var(--ink);line-height:1}
.mv.accent{color:var(--rust)}
.mv.pos{color:var(--sage)}
.mv.warn{color:var(--amber)}

/* ── Progress ─────────────────────────────────────────────── */
.prog-wrap{height:3px;background:var(--cream-3);border-radius:2px;
  overflow:hidden;margin:10px 0}
.prog-fill{height:100%;background:linear-gradient(90deg,var(--rust),var(--gold));
  width:0%;transition:width .5s cubic-bezier(.4,0,.2,1)}

/* ── Textarea / input ─────────────────────────────────────── */
textarea,input[type=text]{
  width:100%;font-family:var(--mono);font-size:.78rem;color:var(--ink);
  background:var(--cream);border:1px solid var(--border-s);border-radius:var(--r);
  padding:10px 12px;resize:vertical;line-height:1.6;outline:none;
  transition:border-color .15s,box-shadow .15s}
textarea:focus,input[type=text]:focus{
  border-color:var(--rust-lt);box-shadow:0 0 0 3px rgba(139,58,42,.07)}
input[type=text]{resize:none}

/* ── Buttons ──────────────────────────────────────────────── */
.br{display:flex;flex-wrap:wrap;gap:5px;margin-top:10px}
button{font-family:var(--sans);font-size:.74rem;font-weight:500;
  border:none;border-radius:4px;padding:7px 13px;cursor:pointer;
  transition:background .15s,transform .1s,box-shadow .15s;letter-spacing:.01em}
button:active{transform:translateY(1px)}

.b-ink{background:var(--ink);color:var(--cream)}
.b-ink:hover{background:var(--ink-2);box-shadow:0 3px 10px rgba(28,26,23,.15)}
.b-rust{background:var(--rust);color:#fff}
.b-rust:hover{background:var(--rust-lt);box-shadow:0 3px 10px rgba(139,58,42,.22)}
.b-sage{background:var(--sage);color:#fff}
.b-sage:hover{background:var(--sage-lt)}
.b-ghost{background:transparent;color:var(--ink-2);border:1px solid var(--border-s)}
.b-ghost:hover{background:var(--cream-2);border-color:var(--ink-muted)}
.b-gold{background:var(--gold);color:#fff}
.b-gold:hover{background:var(--gold-lt)}

/* ── Pre output ───────────────────────────────────────────── */
pre{font-family:var(--mono);font-size:.71rem;line-height:1.75;color:var(--ink-2);
  background:var(--cream);border:1px solid var(--border);border-radius:var(--r);
  padding:12px 14px;overflow:auto;white-space:pre-wrap;word-break:break-word;
  max-height:280px}

/* ── Email panel ─────────────────────────────────────────── */
.email-card{
  background:#fff;border:1px solid var(--border-s);border-radius:var(--r);
  overflow:hidden;box-shadow:var(--shadow)}
.email-header{
  padding:14px 16px;border-bottom:1px solid var(--border);
  background:var(--cream-2)}
.email-from{font-family:var(--mono);font-size:.7rem;color:var(--ink-muted);margin-bottom:2px}
.email-subject{font-family:var(--serif);font-size:1.1rem;font-weight:500;
  color:var(--ink);letter-spacing:-.01em;line-height:1.25}
.email-body{padding:16px;font-size:.85rem;line-height:1.7;color:var(--ink-2);
  min-height:80px}
.email-badge{display:inline-block;padding:2px 8px;border-radius:3px;
  font-family:var(--mono);font-size:.62rem;letter-spacing:.06em;text-transform:uppercase}
.bd-easy{background:rgba(74,94,79,.12);color:var(--sage)}
.bd-medium{background:rgba(154,125,58,.12);color:var(--gold)}
.bd-hard{background:rgba(139,58,42,.12);color:var(--rust)}
.bd-phish{background:rgba(139,58,42,.18);color:var(--rust)}

/* ── Task list (inbox) ────────────────────────────────────── */
.inbox{display:flex;flex-direction:column;gap:0}
.inbox-row{
  display:flex;align-items:center;gap:10px;padding:11px 14px;
  border-bottom:1px solid var(--border);cursor:pointer;
  transition:background .12s;font-size:.82rem;background:#fff}
.inbox-row:last-child{border-bottom:none}
.inbox-row:hover{background:var(--cream-2)}
.inbox-row.selected{background:var(--cream-3);border-left:3px solid var(--rust)}
.inbox-diff{width:52px;flex-shrink:0;text-align:center}
.inbox-sender{font-family:var(--mono);font-size:.68rem;color:var(--ink-muted);
  width:140px;flex-shrink:0;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.inbox-subj{flex:1;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;
  font-weight:500}
.inbox-id{font-family:var(--mono);font-size:.65rem;color:var(--ink-muted);
  width:28px;text-align:right;flex-shrink:0}

/* ── Score ring ───────────────────────────────────────────── */
.score-ring-wrap{display:flex;align-items:center;gap:14px;margin-bottom:12px}
.score-ring{position:relative;width:64px;height:64px;flex-shrink:0}
.score-ring svg{transform:rotate(-90deg)}
.score-ring circle{fill:none;stroke:var(--cream-3);stroke-width:5}
.score-ring .arc{stroke:var(--rust);stroke-linecap:round;
  transition:stroke-dashoffset .5s cubic-bezier(.4,0,.2,1);stroke-dasharray:163;stroke-dashoffset:163}
.score-label{position:absolute;inset:0;display:flex;align-items:center;
  justify-content:center;font-family:var(--serif);font-size:.95rem;
  font-weight:600;color:var(--rust)}
.score-info{flex:1}
.score-task{font-family:var(--serif);font-size:.95rem;font-weight:500;
  color:var(--ink);margin-bottom:2px}
.score-step{font-family:var(--mono);font-size:.68rem;color:var(--ink-muted)}

/* ── Step log ─────────────────────────────────────────────── */
.step-log{margin-top:10px;max-height:100px;overflow-y:auto;
  border-top:1px solid var(--border);padding-top:8px}
.sl{display:flex;align-items:baseline;gap:8px;padding:2px 0;font-size:.72rem}
.sl-n{font-family:var(--mono);color:var(--ink-muted);min-width:20px;flex-shrink:0}
.sl-a{flex:1;color:var(--ink-2);font-family:var(--mono);
  overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.sl-r{font-family:var(--serif);font-style:italic;
  min-width:40px;text-align:right;font-size:.8rem}
.sl-r.hi{color:var(--sage)}.sl-r.lo{color:var(--rust)}.sl-r.mid{color:var(--gold)}

/* ── Hint box ─────────────────────────────────────────────── */
.hint-box{background:rgba(154,125,58,.08);border:1px solid rgba(154,125,58,.25);
  border-radius:var(--r);padding:10px 14px;margin:10px 0;font-size:.8rem;
  color:var(--ink-2);display:none}
.hint-box.show{display:block}
.hint-label{font-family:var(--mono);font-size:.6rem;letter-spacing:.1em;
  text-transform:uppercase;color:var(--gold);margin-bottom:4px}

/* ── Result flash ─────────────────────────────────────────── */
.result-flash{padding:10px 14px;border-radius:var(--r);margin:10px 0;
  font-size:.8rem;display:none;animation:fadein .3s ease}
.result-flash.show{display:block}
.rf-ok{background:rgba(74,94,79,.1);border:1px solid rgba(74,94,79,.25);color:var(--sage)}
.rf-bad{background:rgba(139,58,42,.09);border:1px solid rgba(139,58,42,.22);color:var(--rust)}
@keyframes fadein{from{opacity:0;transform:translateY(4px)}to{opacity:1;transform:none}}

/* ── Guide banner ─────────────────────────────────────────── */
.guide{background:var(--cream-2);border:1px solid var(--border);
  border-radius:var(--r);padding:12px 16px;margin-bottom:20px;
  display:flex;align-items:flex-start;gap:12px}
.guide-icon{font-size:1.3rem;flex-shrink:0;line-height:1}
.guide-text{font-size:.82rem;color:var(--ink-2);line-height:1.6}
.guide-text strong{color:var(--ink);font-weight:500}

/* ── Divider ─────────────────────────────────────────────── */
.div{height:1px;background:var(--border);margin:16px 0}

/* ── Collapsible API section ─────────────────────────────── */
.collapse-btn{background:none;border:none;padding:0;font-family:var(--sans);
  font-size:.78rem;color:var(--rust);cursor:pointer;display:flex;
  align-items:center;gap:5px;margin-top:6px}
.collapse-btn:hover{color:var(--rust-lt)}
.collapse-body{display:none;margin-top:10px}
.collapse-body.open{display:block}

/* ── Animations ──────────────────────────────────────────── */
@keyframes fadeup{from{opacity:0;transform:translateY(8px)}to{opacity:1;transform:none}}
.fu{animation:fadeup .35s ease both}
.fu:nth-child(2){animation-delay:.06s}
.fu:nth-child(3){animation-delay:.12s}
.fu:nth-child(4){animation-delay:.18s}

.mb8{margin-bottom:8px}
.mb16{margin-bottom:16px}
.mb20{margin-bottom:20px}
</style>
</head>
<body>
<div class="shell">

<!-- ── Header ───────────────────────────────────────────── -->
<header class="hdr fu">
  <div>
    <div class="hdr-title"><em>Email Triage</em> — Agent Society</div>
    <div class="hdr-sub">
      <span class="dot"></span>Grand Finale · OpenEnv · Port 7860 · RL Environment
    </div>
  </div>
  <div style="display:flex;gap:6px;flex-wrap:wrap;align-items:center">
    <span class="pill pl">Live</span>
    <span class="pill pp">Port 7860</span>
    <span class="pill pw" id="mode-pill">Society: full</span>
  </div>
</header>

<!-- ── Tabs ─────────────────────────────────────────────── -->
<div class="tabs fu">
  <button class="tab active" onclick="switchTab('play')">📧 Play</button>
  <button class="tab" onclick="switchTab('guide')">📖 How It Works</button>
  <button class="tab" onclick="switchTab('stats')">📊 Live Stats</button>
  <button class="tab" onclick="switchTab('api')">⚙ API</button>
</div>

<!-- ═══════════════════════════════════════════════════════
     TAB: PLAY
     ════════════════════════════════════════════════════════ -->
<div id="tab-play">

  <!-- Guide banner -->
  <div class="guide fu">
    <div class="guide-icon">📬</div>
    <div class="guide-text">
      <strong>How to play:</strong> Pick an email from the inbox on the left.
      Read it, then choose the correct triage action using the buttons.
      Earn a higher score by being accurate — use hints if you're stuck.
      Phishing emails must always be <strong>flagged</strong>, never replied to.
    </div>
  </div>

  <div class="g2 mb20">

    <!-- ── LEFT: Inbox + Email view ─────────────────────── -->
    <div style="display:flex;flex-direction:column;gap:12px">

      <!-- Inbox -->
      <div class="fu">
        <div class="sec mb8"><span class="sec-n">01</span> Inbox — select an email</div>
        <div style="background:#fff;border:1px solid var(--border);border-radius:var(--r);overflow:hidden;box-shadow:var(--shadow)">

          <!-- Filter bar -->
          <div style="padding:10px 12px;border-bottom:1px solid var(--border);background:var(--cream-2);display:flex;gap:6px;flex-wrap:wrap">
            <button class="b-ghost" style="font-size:.68rem;padding:4px 10px" onclick="filterInbox('all')">All</button>
            <button class="b-ghost" style="font-size:.68rem;padding:4px 10px" onclick="filterInbox('easy')">Easy</button>
            <button class="b-ghost" style="font-size:.68rem;padding:4px 10px" onclick="filterInbox('medium')">Medium</button>
            <button class="b-ghost" style="font-size:.68rem;padding:4px 10px;color:var(--rust);border-color:var(--rust-lt)" onclick="filterInbox('phishing')">⚠ Phishing</button>
          </div>

          <div class="inbox" id="inbox-list" style="max-height:240px;overflow-y:auto"></div>
        </div>
      </div>

      <!-- Email view -->
      <div class="fu">
        <div class="sec mb8"><span class="sec-n">02</span> Email</div>
        <div class="email-card" id="email-panel">
          <div class="email-header">
            <div class="email-from" id="e-from">—</div>
            <div class="email-subject" id="e-subject">Select an email from the inbox above</div>
          </div>
          <div class="email-body" id="e-body" style="color:var(--ink-muted);font-style:italic">
            Click any row in the inbox to load the email.
          </div>
        </div>
      </div>

    </div>

    <!-- ── RIGHT: Action + Result ───────────────────────── -->
    <div style="display:flex;flex-direction:column;gap:12px">

      <!-- Score ring + step log -->
      <div class="card fu">
        <div class="score-ring-wrap">
          <div class="score-ring">
            <svg viewBox="0 0 56 56" width="64" height="64">
              <circle cx="28" cy="28" r="26"/>
              <circle class="arc" id="score-arc" cx="28" cy="28" r="26"/>
            </svg>
            <div class="score-label" id="score-label">—</div>
          </div>
          <div class="score-info">
            <div class="score-task" id="score-task">No episode started</div>
            <div class="score-step" id="score-step">Step 0 · Reward 0.000</div>
          </div>
        </div>
        <div class="prog-wrap"><div class="prog-fill" id="prog-fill"></div></div>
        <div class="step-log" id="step-log"></div>
      </div>

      <!-- Action panel -->
      <div class="card fu">
        <div class="card-title">Take Action</div>

        <!-- Guided action buttons — shown based on email type -->
        <div id="guided-actions" style="margin-bottom:12px">
          <div style="font-family:var(--mono);font-size:.62rem;letter-spacing:.09em;text-transform:uppercase;color:var(--ink-muted);margin-bottom:7px">Quick actions</div>
          <div class="br">
            <button class="b-rust" onclick="sendAction({action:'flag_phishing'})">🚨 Flag Phishing</button>
            <button class="b-ghost" onclick="sendAction({action:'classify',priority:'urgent',category:'support'})">🔴 Urgent</button>
            <button class="b-ghost" onclick="sendAction({action:'classify',priority:'high',category:'billing'})">🟠 High · Billing</button>
            <button class="b-ghost" onclick="sendAction({action:'classify',priority:'medium',category:'general'})">🟡 Medium</button>
            <button class="b-ghost" onclick="sendAction({action:'classify',priority:'low',category:'general'})">⚪ Low</button>
          </div>
          <div class="br" style="margin-top:6px">
            <button class="b-ghost" onclick="sendAction({action:'assign_queue',queue:'support'})">→ Support</button>
            <button class="b-ghost" onclick="sendAction({action:'assign_queue',queue:'billing'})">→ Billing</button>
            <button class="b-ghost" onclick="sendAction({action:'assign_queue',queue:'sales'})">→ Sales</button>
            <button class="b-ghost" onclick="sendAction({action:'assign_queue',queue:'it'})">→ IT</button>
            <button class="b-ghost" onclick="sendAction({action:'assign_queue',queue:'hr'})">→ HR</button>
            <button class="b-ghost" onclick="sendAction({action:'escalate',to:'manager',reason:'urgent'})">⬆ Escalate</button>
          </div>
          <div class="br" style="margin-top:6px">
            <button class="b-ghost" onclick="sendAction({action:'use_tool',tool:'crm',params:{}})">🔧 CRM</button>
            <button class="b-ghost" onclick="sendAction({action:'use_tool',tool:'ticketing',params:{}})">🎫 Ticket</button>
            <button class="b-ghost" onclick="sendAction({action:'use_tool',tool:'notification',params:{}})">🔔 Notify</button>
            <button class="b-ghost" onclick="sendAction({action:'use_tool',tool:'calendar',params:{}})">📅 Calendar</button>
            <button class="b-ghost" onclick="sendAction({action:'reply',tone:'professional',summary:'acknowledged'})">✉ Reply</button>
          </div>
        </div>

        <div class="div"></div>

        <!-- Custom JSON input -->
        <div style="font-family:var(--mono);font-size:.62rem;letter-spacing:.09em;text-transform:uppercase;color:var(--ink-muted);margin-bottom:6px">Custom action JSON</div>
        <textarea id="action-input" rows="2" placeholder='{"action": "classify", "priority": "high", "category": "billing"}'></textarea>
        <div class="br">
          <button class="b-ink" onclick="sendCustomAction()">▶ Send Custom</button>
          <button class="b-gold" onclick="requestHint()">💡 Hint</button>
          <button class="b-sage" onclick="startEpisode()">↺ New Episode</button>
        </div>

        <!-- Hint box -->
        <div class="hint-box" id="hint-box">
          <div class="hint-label">Hint</div>
          <div id="hint-text"></div>
        </div>

        <!-- Result flash -->
        <div class="result-flash rf-ok" id="rf-ok">
          ✓ <span id="rf-ok-text"></span>
        </div>
        <div class="result-flash rf-bad" id="rf-bad">
          ✗ <span id="rf-bad-text"></span>
        </div>
      </div>

      <!-- Episode controls -->
      <div class="fu" style="display:flex;gap:8px;flex-wrap:wrap">
        <button class="b-ink" onclick="startEpisode()">↺ New Episode</button>
        <button class="b-ghost" onclick="switchTab('stats')">📊 View Stats</button>
        <button class="b-ghost" onclick="showRawResponse()">{ } Raw</button>
      </div>

    </div>
  </div><!-- /g2 -->

</div><!-- /tab-play -->

<!-- ═══════════════════════════════════════════════════════
     TAB: GUIDE
     ════════════════════════════════════════════════════════ -->
<div id="tab-guide" style="display:none">
  <div style="max-width:680px">
    <div class="sec mb8"><span class="sec-n">01</span> What is this?</div>
    <div class="card mb20">
      <p style="font-size:.88rem;line-height:1.8;color:var(--ink-2);margin-bottom:12px">
        This is a <strong>Reinforcement Learning environment</strong> for training AI agents to triage enterprise emails.
        The environment presents real email scenarios, and agents (or you, manually) must choose the correct action.
        Correct actions earn rewards between <strong>0.001 and 0.999</strong>.
      </p>
      <p style="font-size:.88rem;line-height:1.8;color:var(--ink-2)">
        Behind the scenes, a colony of 5 specialized AI agents debates every email before acting:
        <em>Triage Agent, Phishing Forensic Agent, Safety Auditor, Memory Agent, and Debate Coordinator.</em>
      </p>
    </div>

    <div class="sec mb8"><span class="sec-n">02</span> Task types</div>
    <div class="card mb20">
      <div style="display:grid;grid-template-columns:auto 1fr;gap:8px 14px;font-size:.83rem;line-height:1.7">
        <span class="email-badge bd-easy">Easy</span>
        <span style="color:var(--ink-2)">Single-step: classify priority/category, or route to a queue. 10 emails.</span>
        <span class="email-badge bd-medium">Medium</span>
        <span style="color:var(--ink-2)">Multi-step: use a tool (CRM/ticketing) then route, or classify then escalate. 20 emails.</span>
        <span class="email-badge bd-hard">Hard</span>
        <span style="color:var(--ink-2)">Complex pipelines with state checks — SLA breaches, churn risk, subtle phishing. 30 emails.</span>
        <span class="email-badge bd-phish">Phishing</span>
        <span style="color:var(--rust)">Always flag_phishing — any other action gives the minimum reward.</span>
      </div>
    </div>

    <div class="sec mb8"><span class="sec-n">03</span> Actions reference</div>
    <div class="card mb20">
      <pre style="max-height:none;font-size:.72rem">{"action": "classify",    "priority": "low|medium|high|urgent", "category": "billing|support|sales|hr|it|general"}
{"action": "assign_queue","queue": "billing|support|sales|hr|it|general"}
{"action": "flag_phishing"}
{"action": "escalate",    "to": "manager|director|vp|on-call", "reason": "..."}
{"action": "reply",       "tone": "professional|empathetic|welcoming", "summary": "..."}
{"action": "use_tool",    "tool": "calendar|crm|ticketing|notification", "params": {}}</pre>
    </div>

    <div class="sec mb8"><span class="sec-n">04</span> Reward shaping</div>
    <div class="card">
      <div style="font-size:.83rem;line-height:1.9;color:var(--ink-2)">
        <div>✓ Correct action → reward up to <strong style="color:var(--sage)">0.999</strong></div>
        <div>↗ Partial credit for partially-correct multi-step actions</div>
        <div>💡 Each hint reduces reward by ×0.85</div>
        <div>🚨 CRITICAL safety violation → hard floor <strong style="color:var(--rust)">0.001</strong></div>
        <div>⚠ WARNING violation → reward ×0.5</div>
        <div>☠ Missing phishing on a phishing email → action score = 0</div>
      </div>
    </div>
  </div>
</div>

<!-- ═══════════════════════════════════════════════════════
     TAB: STATS
     ════════════════════════════════════════════════════════ -->
<div id="tab-stats" style="display:none">

  <div class="sec mb8 fu"><span class="sec-n">01</span> Agent Society Metrics</div>
  <div class="g3 mb20 fu">
    <div class="mtile">
      <div class="ml">Phase</div>
      <div class="mv" id="m-phase">—</div>
    </div>
    <div class="mtile">
      <div class="ml">Recent avg</div>
      <div class="mv accent" id="m-avg">—</div>
    </div>
    <div class="mtile">
      <div class="ml">Episodes</div>
      <div class="mv" id="m-ep">—</div>
    </div>
    <div class="mtile">
      <div class="ml">Safety violations</div>
      <div class="mv warn" id="m-viol">—</div>
    </div>
    <div class="mtile">
      <div class="ml">Phishing caught</div>
      <div class="mv pos" id="m-phish">—</div>
    </div>
    <div class="mtile">
      <div class="ml">Fast-track streak</div>
      <div class="mv" id="m-streak">—</div>
    </div>
  </div>

  <div class="g2 fu">
    <div>
      <div class="sec mb8"><span class="sec-n">02</span> Raw stats JSON</div>
      <div class="card"><pre id="stats-raw">Click button below to load.</pre></div>
      <div class="br" style="margin-top:10px">
        <button class="b-sage" onclick="loadStats()">↻ Refresh</button>
        <button class="b-ghost" onclick="loadLog()">⚠ Safety Log</button>
      </div>
    </div>
    <div>
      <div class="sec mb8"><span class="sec-n">03</span> Phase curriculum</div>
      <div class="card">
        <div style="font-size:.82rem;line-height:2;color:var(--ink-2)">
          <div><strong style="color:var(--sage)">Phase 0 — Warmup</strong><br>5 emails · Task 1 only · 15% phishing</div>
          <div class="div" style="margin:8px 0"></div>
          <div><strong style="color:var(--gold)">Phase 1 — Standard</strong><br>10 emails · Task 1+2 · 20% phishing</div>
          <div class="div" style="margin:8px 0"></div>
          <div><strong style="color:var(--amber)">Phase 2 — Mixed</strong><br>20 emails · Task 1+2+3 · 25% phishing</div>
          <div class="div" style="margin:8px 0"></div>
          <div><strong style="color:var(--rust)">Phase 3 — Adversarial</strong><br>30 emails · Task 1+2+3 · 35% phishing</div>
        </div>
      </div>
    </div>
  </div>
</div>

<!-- ═══════════════════════════════════════════════════════
     TAB: API
     ════════════════════════════════════════════════════════ -->
<div id="tab-api" style="display:none">
  <div style="max-width:680px">
    <div class="sec mb8"><span class="sec-n">01</span> Connect your agent</div>
    <div class="card mb20">
      <p style="font-size:.83rem;color:var(--ink-2);line-height:1.8;margin-bottom:10px">
        This environment exposes a standard OpenEnv HTTP API. Connect any agent using the Python client:
      </p>
      <pre>from client import EmailTriageEnv, EmailTriageAction

env_url = "https://nishtha711-email-triage-openenv.hf.space"

with EmailTriageEnv(base_url=env_url).sync() as env:
    result = env.reset()
    obs = result.observation
    print(obs.task.description)   # task description
    print(obs.task.email_subject) # email subject
    print(obs.task.email_body)    # email body

    result = env.step(EmailTriageAction(
        action='{"action": "classify", "priority": "high", "category": "billing"}'
    ))
    print(result.reward)  # 0.001 – 0.999</pre>
    </div>

    <div class="sec mb8"><span class="sec-n">02</span> Live API tester</div>
    <div class="card mb20">
      <div class="card-title">POST /reset</div>
      <button class="b-ink" onclick="apiTest('reset')">Send POST /reset</button>
      <pre id="api-reset-out" style="margin-top:10px;display:none"></pre>
    </div>
    <div class="card mb20">
      <div class="card-title">POST /step</div>
      <textarea id="api-step-body" rows="2" style="margin-bottom:8px">{"action": "classify", "priority": "high", "category": "billing"}</textarea>
      <button class="b-ink" onclick="apiTest('step')">Send POST /step</button>
      <pre id="api-step-out" style="margin-top:10px;display:none"></pre>
    </div>
    <div class="card">
      <div class="card-title">GET /society/stats</div>
      <button class="b-ghost" onclick="apiTest('stats')">Send GET /society/stats</button>
      <pre id="api-stats-out" style="margin-top:10px;display:none"></pre>
    </div>
  </div>
</div>

<!-- ── Raw response modal (hidden) ─────────────────────── -->
<div id="raw-modal" style="display:none;position:fixed;inset:0;background:rgba(28,26,23,.4);z-index:100;padding:40px 20px;overflow:auto" onclick="if(event.target===this)this.style.display='none'">
  <div style="max-width:700px;margin:0 auto;background:#fff;border-radius:8px;padding:24px;box-shadow:0 20px 60px rgba(0,0,0,.2)">
    <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:14px">
      <div style="font-family:var(--serif);font-size:1.1rem;font-weight:500">Last Response</div>
      <button class="b-ghost" onclick="document.getElementById('raw-modal').style.display='none'">✕ Close</button>
    </div>
    <pre id="raw-out" style="max-height:500px">No response yet.</pre>
  </div>
</div>

</div><!-- /shell -->

<script>
// ── Task data (embedded from tasks.py) ──────────────────
const TASKS = [{"id": 0, "diff": "easy", "desc": "Classify the priority and category of this invoice-dispute email.", "sender": "client@bigcorp.com", "subject": "Invoice #4892 \u2014 Incorrect Amount Charged", "body": "We were charged $4,200 for March but our contract states $3,800. Please correct immediately.", "hint1": "You need to classify this email by priority and category.", "hint2": "Billing disputes from clients are high priority.", "answer": "{\"action\": \"classify\", \"priority\": \"high\", \"category\": \"billing\"}", "phishing": false}, {"id": 1, "diff": "easy", "desc": "Route this IT helpdesk VPN request to the correct queue.", "sender": "employee@company.com", "subject": "Cannot Connect to VPN", "body": "VPN has been down since this morning. Tried rebooting but still failing.", "hint1": "Assign this email to a queue.", "hint2": "VPN issues belong in the IT support queue.", "answer": "{\"action\": \"assign_queue\", \"queue\": \"it\"}", "phishing": false}, {"id": 2, "diff": "easy", "desc": "This email is a phishing attempt. Flag it appropriately.", "sender": "security-alert@paypa1-support.com", "subject": "Urgent: Your Account Has Been Suspended", "body": "Your PayPal account is suspended. Click here: http://paypa1-login.ru/verify", "hint1": "Examine the sender domain very carefully.", "hint2": "'paypa1-support.com' is a lookalike domain \u2014 this is phishing.", "answer": "{\"action\": \"flag_phishing\"}", "phishing": true}, {"id": 3, "diff": "easy", "desc": "Classify this low-urgency vendor newsletter.", "sender": "newsletter@vendor.com", "subject": "Our Q1 Product Updates", "body": "Excited to share our Q1 roadmap. No action required.", "hint1": "Classify by priority and category.", "hint2": "Newsletters with no required action are low priority general emails.", "answer": "{\"action\": \"classify\", \"priority\": \"low\", \"category\": \"general\"}", "phishing": false}, {"id": 4, "diff": "easy", "desc": "Route this inbound sales lead to the correct team queue.", "sender": "prospect@startup.io", "subject": "Interested in Enterprise Plan", "body": "We are a 200-person startup interested in an enterprise demo.", "hint1": "Assign this email to a queue.", "hint2": "Demo requests from prospects go to the sales queue.", "answer": "{\"action\": \"assign_queue\", \"queue\": \"sales\"}", "phishing": false}, {"id": 5, "diff": "easy", "desc": "Classify this critical outage report from a key customer.", "sender": "cto@enterprise-client.com", "subject": "CRITICAL: Your API is DOWN", "body": "Your API has been returning 503 errors for 30 minutes. We are losing revenue. P1.", "hint1": "Classify priority and category.", "hint2": "API outages reported by customers are urgent support issues.", "answer": "{\"action\": \"classify\", \"priority\": \"urgent\", \"category\": \"support\"}", "phishing": false}, {"id": 6, "diff": "easy", "desc": "Route this employee parental-leave HR question.", "sender": "employee2@company.com", "subject": "Question About Parental Leave Policy", "body": "Could you clarify how many weeks of parental leave we are entitled to?", "hint1": "Route this to the right team.", "hint2": "Parental leave questions belong in the HR queue.", "answer": "{\"action\": \"assign_queue\", \"queue\": \"hr\"}", "phishing": false}, {"id": 7, "diff": "easy", "desc": "Flag this CEO-fraud / wire-transfer phishing attempt.", "sender": "ceo.john@company-secure-msg.com", "subject": "Confidential Wire Transfer Needed", "body": "This is CEO John. Please wire $50,000 to a new vendor immediately. Do not tell anyone.", "hint1": "Check the sender domain \u2014 is it your real company domain?", "hint2": "CEO-fraud uses lookalike domains and urgency. Flag as phishing.", "answer": "{\"action\": \"flag_phishing\"}", "phishing": true}, {"id": 8, "diff": "easy", "desc": "Classify this refund request from a customer.", "sender": "customer@gmail.com", "subject": "Refund Request \u2014 Order #9901", "body": "I ordered the wrong plan and want a refund for the unused month.", "hint1": "Classify priority and category.", "hint2": "Refund requests are medium priority billing emails.", "answer": "{\"action\": \"classify\", \"priority\": \"medium\", \"category\": \"billing\"}", "phishing": false}, {"id": 9, "diff": "easy", "desc": "Classify this internal mandatory password-reset notification.", "sender": "security@company.com", "subject": "Mandatory Password Reset Required", "body": "All employees must reset their passwords by Friday via the self-service portal.", "hint1": "Classify this IT security notification.", "hint2": "Internal IT security policies are medium priority IT emails.", "answer": "{\"action\": \"classify\", \"priority\": \"medium\", \"category\": \"it\"}", "phishing": false}, {"id": 10, "diff": "medium", "desc": "Follow-up on existing ticket \u2014 check CRM then route to support.", "sender": "customer@acme.com", "subject": "Re: Ticket #TK-1042 Still Not Resolved", "body": "It has been 3 days since I reported this. Ticket #TK-1042 still open.", "hint1": "Use a tool first, then route the email.", "hint2": "Look up the ticket in the CRM, then assign to the support queue.", "answer": "Step 1: {\"action\":\"use_tool\",\"tool\":\"crm\",\"params\":{\"ticket\":\"TK-1042\"}} ", "phishing": false}, {"id": 11, "diff": "medium", "desc": "Schedule a demo for this sales prospect and send a confirmation reply.", "sender": "vp@prospect-corp.com", "subject": "Ready to See a Demo", "body": "We are ready to move forward. Can you book a 45-minute demo for next week?", "hint1": "Use a tool and then reply.", "hint2": "Check calendar availability, then send a professional reply.", "answer": "Step 1: {\"action\":\"use_tool\",\"tool\":\"calendar\",\"params\":{\"search\":\"demo slot\"}} ", "phishing": false}, {"id": 12, "diff": "medium", "desc": "Create a bug-report ticket and send an empathetic acknowledgment reply.", "sender": "user@customer.com", "subject": "Login Page Crashes on Mobile", "body": "Every time I try to log in on iPhone the app crashes. Running iOS 17.", "hint1": "Create a ticket and acknowledge the user.", "hint2": "Use the ticketing tool, then send an empathetic reply.", "answer": "Step 1: {\"action\":\"use_tool\",\"tool\":\"ticketing\",\"params\":{\"type\":\"bug\"}} ", "phishing": false}, {"id": 13, "diff": "medium", "desc": "Classify this billing dispute then escalate to manager.", "sender": "cfo@bigclient.com", "subject": "Overcharge of $28,000 \u2014 Legal Action If Not Resolved", "body": "We have been overcharged $28,000. If not resolved in 24 hours we involve legal.", "hint1": "Classify and then escalate.", "hint2": "Large billing disputes with legal threats need urgent classification then manager escalation.", "answer": "Step 1: {\"action\":\"classify\",\"priority\":\"urgent\",\"category\":\"billing\"} ", "phishing": false}, {"id": 14, "diff": "medium", "desc": "Thread-based sales follow-up \u2014 check CRM history then route to sales.", "sender": "buyer@retailer.com", "subject": "Re: Pricing Discussion", "body": "Following up on our call \u2014 ready to sign if you can match $499/mo.", "hint1": "Look up context and route to the right team.", "hint2": "Check CRM for account history, then route to the sales queue.", "answer": "Step 1: {\"action\":\"use_tool\",\"tool\":\"crm\",\"params\":{\"search\":\"retailer account\"}} ", "phishing": false}, {"id": 15, "diff": "medium", "desc": "Create an access-request ticket for this IT request then route to IT queue.", "sender": "dev@company.com", "subject": "Need Admin Rights for CI/CD Setup", "body": "I need temporary admin access to configure our new Jenkins server. Manager approved.", "hint1": "Create a ticket and route it.", "hint2": "Access requests need a ticket and go to the IT queue.", "answer": "Step 1: {\"action\":\"use_tool\",\"tool\":\"ticketing\",\"params\":{\"type\":\"access_request\"}} ", "phishing": false}, {"id": 16, "diff": "medium", "desc": "Log this new customer in CRM then send a welcoming onboarding reply.", "sender": "admin@new-customer.com", "subject": "Just Signed Up \u2014 Need Onboarding Help", "body": "We just completed our contract. Looking forward to getting started!", "hint1": "Update CRM and send a reply.", "hint2": "Add to CRM first, then send a welcoming reply to the new customer.", "answer": "Step 1: {\"action\":\"use_tool\",\"tool\":\"crm\",\"params\":{\"action\":\"create_account\"}} ", "phishing": false}, {"id": 17, "diff": "medium", "desc": "Escalated complaint in thread \u2014 classify urgent then escalate to director.", "sender": "vp@unhappy-client.com", "subject": "Re: Unresolved Issues \u2014 Escalating to Leadership", "body": "Third time reaching out. Escalating to your director level now.", "hint1": "Classify urgency and escalate.", "hint2": "Repeat escalations from VPs are urgent and should go to director level.", "answer": "Step 1: {\"action\":\"classify\",\"priority\":\"urgent\",\"category\":\"support\"} ", "phishing": false}, {"id": 18, "diff": "medium", "desc": "Route this job application to HR and send an acknowledgment reply.", "sender": "applicant@gmail.com", "subject": "Application for Senior Engineer Role", "body": "Please find my resume for the Senior Software Engineer position on LinkedIn.", "hint1": "Route and reply.", "hint2": "Job applications go to HR and need an acknowledgment reply.", "answer": "Step 1: {\"action\":\"assign_queue\",\"queue\":\"hr\"} ", "phishing": false}, {"id": 19, "diff": "medium", "desc": "Payment failure from Stripe \u2014 look up customer in CRM then route to billing.", "sender": "noreply@stripe.com", "subject": "Payment Failed for Customer ID C-8821", "body": "Automated notice: payment of $1,200 for C-8821 failed. Card declined.", "hint1": "Look up the customer and route.", "hint2": "Check CRM for customer details, then send to billing queue.", "answer": "Step 1: {\"action\":\"use_tool\",\"tool\":\"crm\",\"params\":{\"customer_id\":\"C-8821\"}} ", "phishing": true}];

// ── State ────────────────────────────────────────────────
let currentTask = null;
let selectedRow = null;
let stepCount = 0;
let hintCount = 0;
let lastResponse = null;
let currentFilter = 'all';

// ── Helpers ──────────────────────────────────────────────
const $ = id => document.getElementById(id);
const show = (id, d) => { $(id).textContent = JSON.stringify(d, null, 2); $(id).style.display = 'block'; };

async function post(url, body) {
  const r = await fetch(url, { method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify(body) });
  return r.json();
}
async function get(url) { return (await fetch(url)).json(); }

// ── Tabs ─────────────────────────────────────────────────
function switchTab(name) {
  ['play','guide','stats','api'].forEach(t => {
    $('tab-'+t).style.display = t===name ? '' : 'none';
  });
  document.querySelectorAll('.tab').forEach((el,i) => {
    el.classList.toggle('active', ['play','guide','stats','api'][i]===name);
  });
  if (name === 'stats') loadStats();
}

// ── Build inbox ───────────────────────────────────────────
function buildInbox(filter='all') {
  const list = $('inbox-list');
  list.innerHTML = '';
  const filtered = TASKS.filter(t => {
    if (filter === 'all') return true;
    if (filter === 'phishing') return t.phishing;
    return t.diff === filter;
  });
  filtered.forEach(t => {
    const row = document.createElement('div');
    row.className = 'inbox-row';
    row.dataset.id = t.id;
    const diffBadge = `<span class="email-badge bd-${t.phishing?'phish':t.diff}">${t.phishing?'⚠':t.diff}</span>`;
    row.innerHTML = `
      <span class="inbox-id">#${t.id}</span>
      <span class="inbox-diff">${diffBadge}</span>
      <span class="inbox-sender">${t.sender}</span>
      <span class="inbox-subj">${t.subject}</span>
    `;
    row.onclick = () => selectTask(t, row);
    list.appendChild(row);
  });
}

function filterInbox(f) {
  currentFilter = f;
  buildInbox(f);
}

function selectTask(task, row) {
  // Deselect previous
  document.querySelectorAll('.inbox-row').forEach(r => r.classList.remove('selected'));
  row.classList.add('selected');
  selectedRow = row;
  currentTask = task;
  hintCount = 0;

  // Populate email view
  $('e-from').textContent = task.sender;
  $('e-subject').textContent = task.subject;
  $('e-body').textContent = task.body;
  $('e-body').style.fontStyle = 'normal';
  $('e-body').style.color = 'var(--ink-2)';

  // Pre-fill action input with a sensible default
  if (task.phishing) {
    $('action-input').value = '{"action": "flag_phishing"}';
  } else if (task.answer && task.answer.startsWith('{')) {
    $('action-input').value = task.answer;
  } else {
    $('action-input').value = '';
  }

  // Clear hints and result flash
  $('hint-box').classList.remove('show');
  $('rf-ok').classList.remove('show');
  $('rf-bad').classList.remove('show');

  // Update score task label
  $('score-task').textContent = `Task #${task.id} — ${task.diff}`;
  $('score-step').textContent = `Step 0 · Select an action`;
}

// ── Episode ───────────────────────────────────────────────
async function startEpisode() {
  stepCount = 0;
  hintCount = 0;
  $('step-log').innerHTML = '';
  $('hint-box').classList.remove('show');
  $('rf-ok').classList.remove('show');
  $('rf-bad').classList.remove('show');

  const d = await post('/web/reset', {});
  lastResponse = d;
  const obs = d.observation || {};
  const task = obs.task;

  // Auto-select the task in inbox if we can match it
  if (task && task.task_id !== undefined) {
    const match = TASKS.find(t => t.id === task.task_id);
    if (match) {
      const rows = $('inbox-list').querySelectorAll('.inbox-row');
      rows.forEach(r => { if (parseInt(r.dataset.id) === match.id) selectTask(match, r); });
    }
    $('score-task').textContent = `Task #${task.task_id} — ${task.difficulty || ''}`;
  }

  updateScore(0, 0);
  addStepLine(0, '— episode started —', 0);
}

// ── Send action ───────────────────────────────────────────
async function sendAction(obj) {
  $('action-input').value = JSON.stringify(obj);
  await sendCustomAction();
}

async function sendCustomAction() {
  let raw = $('action-input').value.trim();
  let payload;
  try { payload = JSON.parse(raw); }
  catch { payload = { action: raw }; }

  stepCount++;
  const d = await post('/web/step', { action: payload });
  lastResponse = d;

  const obs = d.observation || {};
  const reward = obs.reward ?? d.reward ?? 0;
  const done = d.done;
  const achieved = obs.task_achieved;
  const result = obs.last_action_result || '';

  updateScore(reward, stepCount);
  addStepLine(stepCount, payload.action || raw, reward);

  // Flash result
  $('rf-ok').classList.remove('show');
  $('rf-bad').classList.remove('show');
  if (achieved) {
    $('rf-ok-text').textContent = `Task achieved! Reward: ${reward.toFixed(3)}`;
    $('rf-ok').classList.add('show');
  } else if (reward > 0.5) {
    $('rf-ok-text').textContent = result || `Reward: ${reward.toFixed(3)}`;
    $('rf-ok').classList.add('show');
  } else if (result.includes('SAFETY') || reward < 0.05) {
    $('rf-bad-text').textContent = result || `Low reward: ${reward.toFixed(3)}`;
    $('rf-bad').classList.add('show');
  }
}

// ── Hint ──────────────────────────────────────────────────
async function requestHint() {
  // Show local hint first (no server call needed for guidance)
  if (currentTask) {
    const h = hintCount === 0 ? currentTask.hint1 :
               hintCount === 1 ? currentTask.hint2 :
               `Answer: ${currentTask.answer}`;
    $('hint-text').textContent = h;
    $('hint-box').classList.add('show');
    hintCount++;
  }

  // Also call server hint (for reward tracking)
  const d = await post('/web/step', { action: { action: 'hint' } });
  lastResponse = d;
  const obs = d.observation || {};
  $('score-step').textContent = `Step ${stepCount} · Hint used (reward ×0.85)`;
}

// ── Score / progress ──────────────────────────────────────
function updateScore(reward, step) {
  const pct = Math.min(Math.max(reward * 100, 0), 100);
  $('prog-fill').style.width = pct + '%';

  // Arc (circumference ≈ 163 for r=26)
  const offset = 163 * (1 - Math.min(reward, 1));
  $('score-arc').style.strokeDashoffset = offset;
  $('score-arc').style.stroke = reward > 0.7 ? 'var(--sage)' : reward > 0.4 ? 'var(--gold)' : 'var(--rust)';
  $('score-label').textContent = reward > 0 ? reward.toFixed(2) : '—';
  $('score-step').textContent = `Step ${step} · Reward ${reward.toFixed(3)}`;
}

function addStepLine(step, action, reward) {
  const log = $('step-log');
  const cls = reward > 0.7 ? 'hi' : reward > 0.3 ? 'mid' : 'lo';
  const el = document.createElement('div');
  el.className = 'sl';
  el.innerHTML = `
    <span class="sl-n">${step}</span>
    <span class="sl-a" title="${action}">${action}</span>
    <span class="sl-r ${cls}">${reward > 0 ? reward.toFixed(3) : ''}</span>
  `;
  log.appendChild(el);
  log.scrollTop = log.scrollHeight;
}

// ── Raw response modal ─────────────────────────────────────
function showRawResponse() {
  $('raw-out').textContent = lastResponse ? JSON.stringify(lastResponse, null, 2) : 'No response yet.';
  $('raw-modal').style.display = 'block';
}

// ── Stats ─────────────────────────────────────────────────
async function loadStats() {
  const d = await get('/society/stats');
  show('stats-raw', d);
  const pn = d.phase_name ?? d.tier ?? (d.phase !== undefined ? `Phase ${d.phase}` : '—');
  $('m-phase').textContent = pn;
  $('m-avg').textContent   = d.recent_avg !== undefined ? d.recent_avg.toFixed(3) : '—';
  $('m-ep').textContent    = d.episode !== undefined ? d.episode : '—';
  const viol = d.total_safety_violations ?? d.safety_violations ?? d.episode_violations;
  $('m-viol').textContent  = viol !== undefined ? viol : '—';
  $('m-phish').textContent = d.phishing_caught !== undefined ? d.phishing_caught : '—';
  $('m-streak').textContent = d.fast_track_streak !== undefined ? d.fast_track_streak : '—';
  $('mode-pill').textContent = 'Society: ' + (d.society_mode ?? 'full');
}

async function loadLog() {
  const d = await get('/society/log?tail=15');
  show('stats-raw', d);
}

// ── API tester ────────────────────────────────────────────
async function apiTest(which) {
  if (which === 'reset') {
    const d = await post('/reset', {});
    show('api-reset-out', d);
  } else if (which === 'step') {
    const raw = $('api-step-body').value.trim();
    const d = await post('/step', { action: raw });
    show('api-step-out', d);
  } else if (which === 'stats') {
    const d = await get('/society/stats');
    show('api-stats-out', d);
  }
}

// ── Init ──────────────────────────────────────────────────
buildInbox('all');
loadStats();
setInterval(loadStats, 30000);
</script>
</body>
</html>"""


@app.get("/web", response_class=HTMLResponse, include_in_schema=False)
async def web_ui():
    return HTMLResponse(content=_HTML)


@app.get("/", response_class=RedirectResponse, include_in_schema=False)
async def root_redirect():
    return RedirectResponse(url="/web")


def main(host: str = "0.0.0.0", port: int = 7860) -> None:
    import uvicorn
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
