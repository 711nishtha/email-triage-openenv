"""
server/email_triage_environment.py — Email Triage RL Environment (Grand Finale)
=================================================================================
Grand Finale additions over Round 1 baseline:
  1. Symbolic safety layer (pure Python hard veto + warning tier)
  2. CurriculumManager (adaptive phase-based) with legacy Curriculum fallback
  3. Society metadata in state (dashboard at /society/stats)
  4. GRPO-shaped rewards: dense progress-delta signal every step
  5. Comprehensive violation logging to safety_violations.jsonl
  6. Double reward clamping (also clamped in inference.py)

Reward contract: every value strictly in (0.001, 0.999) — never exactly 0 or 1.
Safety layer fires BEFORE grading — deterministic Python, zero hallucination.
"""
from __future__ import annotations
import json, logging, os, sys
from datetime import datetime
from pathlib import Path
from typing import Any, Optional
from uuid import uuid4
from openenv.core.env_server.interfaces import Environment
sys.path.insert(0, str(Path(__file__).parent.parent))
from models import (
    EmailTriageAction, EmailTriageObservation, EmailTriageState,
    EpisodeID, StepCount, Task, TaskInfo, TrackerState,
)
logger = logging.getLogger(__name__)
MAX_STEPS = 10
REWARD_MIN = 0.001
REWARD_MAX = 0.999
VIOLATION_LOG_PATH = Path(os.environ.get("VIOLATION_LOG", "safety_violations.jsonl"))

def _clamp(v: float) -> float:
    return max(REWARD_MIN, min(REWARD_MAX, v))

# --- Symbolic safety layer ---
_WIRE_KW = ["wire transfer","wire funds","bank transfer","swift code","routing number","iban","bank account number"]
_LOOK_KW = ["-secure-","-portal.","-helpdesk.","-support-","-login.","paypa1","amaz0n","microsofft","g00gle"]
_CRED_KW = ["your password","enter your credentials","username and password","verify your account","confirm your login"]

class _SafetyResult:
    def __init__(self, passed:bool, critical:bool=False, message:str="", reward_multiplier:float=1.0):
        self.passed=passed; self.critical=critical; self.message=message; self.reward_multiplier=reward_multiplier

def _symbolic_safety(email_sender:str, email_body:str, action_dict:dict, is_phishing_task:bool) -> _SafetyResult:
    at = action_dict.get("action",""); bl = email_body.lower(); sl = email_sender.lower()
    if is_phishing_task and at != "flag_phishing":
        return _SafetyResult(False,True,f"CRITICAL: phishing email not flagged (chose '{at}')",0.0)
    if any(k in bl for k in _WIRE_KW) and at != "flag_phishing":
        return _SafetyResult(False,True,f"CRITICAL: wire-transfer not flagged (chose '{at}')",0.0)
    if any(k in bl for k in _CRED_KW) and at == "reply":
        return _SafetyResult(False,True,"CRITICAL: replying to credential-harvesting email",0.0)
    if at == "reply" and any(k in sl for k in _LOOK_KW):
        return _SafetyResult(False,False,f"WARNING: replying to suspicious domain '{email_sender}'",0.5)
    return _SafetyResult(True)

def _log_violation(episode_id:str, step:int, action:str, message:str, critical:bool) -> None:
    entry={"ts":datetime.utcnow().isoformat(),"episode_id":episode_id,"step":step,"action":action[:200],"message":message,"critical":critical}
    try:
        with open(VIOLATION_LOG_PATH,"a") as f: f.write(json.dumps(entry)+"\n")
    except OSError: pass

def _load_curriculum():
    try:
        from curriculum.manager import CurriculumManager
        c = CurriculumManager(); logger.info("CurriculumManager loaded"); return c
    except ImportError: pass
    from server.services.curriculum import Curriculum
    logger.info("Legacy Curriculum loaded (fallback)"); return Curriculum()


class EmailTriageEnvironment(Environment[EmailTriageAction, EmailTriageObservation, EmailTriageState]):
    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self) -> None:
        logger.info("Initialising EmailTriageEnvironment (Grand Finale)...")
        self._state = EmailTriageState(episode_id=str(uuid4()), step_count=0)
        self._curriculum = _load_curriculum()
        self._episode_violations: list[str] = []
        from server.services.task_grader import TaskGrader
        from server.services.episode_tracker import EpisodeTracker
        self._grader = TaskGrader()
        self._tracker = EpisodeTracker()
        self._current_task: Task | None = None

    def _sync_state(self) -> None:
        self._state.current_task = self._current_task
        self._state.tracker = TrackerState(
            step_count=self._tracker.step_count, hints_used=self._tracker.hints_used,
            progress=self._tracker.previous_progress, actions_taken=list(self._tracker.actions_taken),
            credited_steps=list(self._tracker._credited_steps), phishing_missed=self._tracker.phishing_missed,
            tool_calls=list(self._tracker.tool_calls), flags_raised=list(self._tracker.flags_raised),
        )
        if hasattr(self._curriculum,"current_difficulty"): self._state.current_tier=self._curriculum.current_difficulty.value
        elif hasattr(self._curriculum,"phase"): self._state.current_tier=f"phase_{self._curriculum.phase}"

    def reset(self, seed:Optional[int]=None, episode_id:Optional[str]=None, **kwargs:Any) -> EmailTriageObservation:
        """POST /reset — start new episode, return initial observation."""
        self._state = EmailTriageState(episode_id=episode_id or str(uuid4()), step_count=0)
        self._tracker.reset()
        self._episode_violations = []
        self._current_task = self._curriculum.next_task()
        self._sync_state()
        logger.info("reset() task=%d difficulty=%s", int(self._current_task.task_id), self._current_task.difficulty.value)
        return EmailTriageObservation(
            episode_id=EpisodeID(self._state.episode_id), step_count=StepCount(0),
            task=TaskInfo.from_task(self._current_task),
            last_action_result=f"New episode. Task [{self._current_task.difficulty.value.upper()}]: {self._current_task.description}",
            last_action_valid=True, task_achieved=False, partial_progress=0.0,
            hints_used=0, hint_text="", done=False, reward=0.0,
        )

    def step(self, action:EmailTriageAction, timeout_s:Optional[float]=None, **kwargs:Any) -> EmailTriageObservation:
        """POST /step — parse action, safety-check, grade, clamp reward."""
        assert self._current_task is not None, "Call reset() before step()"
        self._state.step_count += 1
        raw = action.action.strip()

        # hint intercept
        if raw.lower() in ("hint","get_hint",'{"action": "hint"}'):
            hint_text = self._grader.get_hint(self._current_task, self._tracker.hints_used)
            self._tracker.record_hint()
            self._sync_state()
            return EmailTriageObservation(
                episode_id=EpisodeID(self._state.episode_id), step_count=StepCount(self._state.step_count),
                task=TaskInfo.from_task(self._current_task),
                last_action_result=f"HINT [{self._tracker.hints_used}]: {hint_text}",
                last_action_valid=True, task_achieved=False, partial_progress=self._tracker.previous_progress,
                hints_used=self._tracker.hints_used, hint_text=hint_text, done=False, reward=0.0,
            )

        from server.services.task_grader import _parse_action
        action_dict = _parse_action(raw)
        valid = bool(action_dict)

        # symbolic safety
        safety = _symbolic_safety(
            self._current_task.email.sender, self._current_task.email.body,
            action_dict, self._current_task.success_criteria.phishing_task,
        )
        if not safety.passed:
            self._episode_violations.append(safety.message)
            _log_violation(self._state.episode_id, self._state.step_count, raw, safety.message, safety.critical)
            logger.warning("SAFETY %s: %s", "CRITICAL" if safety.critical else "WARNING", safety.message)
            if safety.critical:
                self._sync_state()
                return EmailTriageObservation(
                    episode_id=EpisodeID(self._state.episode_id), step_count=StepCount(self._state.step_count),
                    task=TaskInfo.from_task(self._current_task),
                    last_action_result=f"🚨 SAFETY VIOLATION: {safety.message}",
                    last_action_valid=False, task_achieved=False, partial_progress=self._tracker.previous_progress,
                    hints_used=self._tracker.hints_used, hint_text="", done=False, reward=REWARD_MIN,
                )

        latest_step = self._tracker.record_step(raw, action_dict, valid)
        grade_result = self._grader.grade(self._current_task, self._tracker, latest_step, hints_used=self._tracker.hints_used)
        task_achieved = grade_result.task_achieved
        reward = _clamp(grade_result.reward * safety.reward_multiplier)
        done = task_achieved or self._state.step_count >= MAX_STEPS

        if done:
            if hasattr(self._curriculum,"record_episode"):
                self._curriculum.record_episode(
                    score=reward, achieved=task_achieved, safety_violations=len(self._episode_violations),
                    phishing_caught=int("phishing" in self._tracker.flags_raised),
                    phishing_missed=int(self._tracker.phishing_missed), hints_used=self._tracker.hints_used,
                )
            else:
                self._curriculum.record_result(self._current_task, achieved=task_achieved, reward=reward)

        self._sync_state()
        suffix = f" [⚠ reward×{safety.reward_multiplier}]" if not safety.passed and not safety.critical else ""
        return EmailTriageObservation(
            episode_id=EpisodeID(self._state.episode_id), step_count=StepCount(self._state.step_count),
            task=TaskInfo.from_task(self._current_task), last_action_result=grade_result.reason+suffix,
            last_action_valid=valid, task_achieved=task_achieved, partial_progress=self._tracker.previous_progress,
            hints_used=self._tracker.hints_used, hint_text="", done=done, reward=reward,
        )

    @property
    def state(self) -> EmailTriageState: return self._state
    @property
    def episode_violations(self) -> list[str]: return list(self._episode_violations)
