"""
ui.py — Optional Streamlit UI for the Email Triage OpenEnv.
Runs on port 8501. Backend must be available on port 7860.
"""

import streamlit as st
import httpx
import json
from typing import Any, Dict, List, Optional

BACKEND_URL = "http://localhost:7860"

st.set_page_config(
    page_title="Email Triage OpenEnv",
    page_icon="📧",
    layout="wide",
)

# ─────────────────────────────────────────────
# Session State Init
# ─────────────────────────────────────────────

if "observation" not in st.session_state:
    st.session_state.observation = None
if "rewards" not in st.session_state:
    st.session_state.rewards = []
if "done" not in st.session_state:
    st.session_state.done = False
if "step" not in st.session_state:
    st.session_state.step = 0
if "log" not in st.session_state:
    st.session_state.log = []


def backend_call(method: str, path: str, data: Optional[Dict] = None) -> Optional[Dict]:
    try:
        url = f"{BACKEND_URL}{path}"
        if method == "POST":
            resp = httpx.post(url, json=data, timeout=10)
        else:
            resp = httpx.get(url, timeout=10)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        st.error(f"Backend error: {e}")
        return None


def check_health() -> bool:
    result = backend_call("GET", "/health")
    return result is not None and result.get("status") == "ok"


# ─────────────────────────────────────────────
# UI Layout
# ─────────────────────────────────────────────

st.title("📧 Enterprise Email Triage — OpenEnv")
st.markdown("*AI agent evaluation environment for email triage tasks.*")

# Sidebar
with st.sidebar:
    st.header("⚙️ Control Panel")

    # Health check
    healthy = check_health()
    if healthy:
        st.success("✅ Backend Online")
    else:
        st.error("❌ Backend Offline")

    st.divider()

    # Task selection
    task_id = st.selectbox("Select Task", ["easy", "medium", "hard"], index=1)

    if st.button("🔄 Reset Environment", use_container_width=True):
        result = backend_call("POST", "/reset", {"task_id": task_id})
        if result:
            st.session_state.observation = result
            st.session_state.rewards = []
            st.session_state.done = False
            st.session_state.step = 0
            st.session_state.log = [f"✅ Reset to task: {task_id}"]
            st.rerun()

    st.divider()

    # Cumulative score
    if st.session_state.rewards:
        latest_reward = st.session_state.rewards[-1]
        st.metric("Latest Reward", f"{latest_reward:.4f}")
        st.metric("Steps Taken", st.session_state.step)
        if st.session_state.done:
            st.success(f"🏁 Episode Complete")

    # Activity log
    st.subheader("📋 Activity Log")
    for entry in reversed(st.session_state.log[-10:]):
        st.text(entry)


# ─────────────────────────────────────────────
# Main Content
# ─────────────────────────────────────────────

if st.session_state.observation is None:
    st.info("👈 Select a task and click **Reset Environment** to begin.")
    st.stop()

obs = st.session_state.observation
inbox = obs.get("inbox", [])
triaged = obs.get("triaged", [])

col1, col2 = st.columns([2, 1])

with col1:
    st.subheader(f"📥 Inbox ({len(inbox)} emails)")

    if not inbox:
        st.success("✅ All emails triaged! Click **Mark Done** to finish.")
    else:
        for email in inbox:
            with st.expander(f"📨 [{email['id']}] {email['subject']} — *{email['sender']}*"):
                st.caption(f"🕐 {email['timestamp']}")
                st.write(email["body"])

                if not st.session_state.done:
                    col_a, col_b, col_c = st.columns(3)
                    with col_a:
                        priority = st.selectbox(
                            "Priority",
                            ["urgent", "high", "medium", "low"],
                            key=f"priority_{email['id']}",
                        )
                    with col_b:
                        category = st.selectbox(
                            "Category",
                            ["phishing", "urgent_business", "internal_task", "marketing",
                             "hr", "legal", "it_support", "finance", "spam"],
                            key=f"category_{email['id']}",
                        )
                    with col_c:
                        route_to = st.selectbox(
                            "Route To",
                            ["security", "executive", "manager", "it", "hr", "finance", "archive", "trash"],
                            key=f"route_{email['id']}",
                        )

                    if st.button(f"✅ Triage", key=f"triage_{email['id']}"):
                        action = {
                            "action_type": "triage",
                            "email_id": email["id"],
                            "priority": priority,
                            "category": category,
                            "route_to": route_to,
                        }
                        result = backend_call("POST", "/step", action)
                        if result:
                            reward = result["reward"]
                            st.session_state.rewards.append(reward)
                            st.session_state.step += 1
                            st.session_state.observation = result["observation"]
                            st.session_state.done = result["done"]
                            st.session_state.log.append(
                                f"Step {st.session_state.step}: Triaged {email['id']} → {category}/{priority} (reward={reward:.4f})"
                            )
                            st.rerun()

with col2:
    st.subheader("✅ Triaged")
    if triaged:
        for t in triaged:
            color = "🔴" if t["priority"] == "urgent" else "🟡" if t["priority"] == "high" else "🟢"
            icon = "🚨" if t["category"] == "phishing" else "📧"
            st.markdown(f"{color} {icon} **{t['email_id']}**  \n`{t['category']}` → `{t['route_to']}`")
    else:
        st.info("No emails triaged yet.")

    st.divider()

    # Done button
    if not st.session_state.done:
        if st.button("🏁 Mark Done", use_container_width=True, type="primary"):
            result = backend_call("POST", "/step", {"action_type": "done"})
            if result:
                reward = result["reward"]
                st.session_state.rewards.append(reward)
                st.session_state.step += 1
                st.session_state.observation = result["observation"]
                st.session_state.done = True
                summary = result.get("info", {}).get("episode_summary", {})
                if summary:
                    score = summary.get("episode_score", reward)
                    st.session_state.log.append(
                        f"Episode complete! Final score: {score:.4f} | Triaged: {summary.get('num_triaged', '?')}/{summary.get('num_emails', '?')}"
                    )
                st.rerun()
    else:
        # Show episode summary
        state_data = backend_call("GET", "/state")
        if state_data:
            st.metric("Final Score", f"{state_data.get('cumulative_reward', 0):.4f}")

    # Reward chart
    if st.session_state.rewards:
        st.subheader("📊 Reward History")
        import pandas as pd
        df = pd.DataFrame({
            "Step": list(range(1, len(st.session_state.rewards) + 1)),
            "Reward": st.session_state.rewards,
        })
        st.line_chart(df.set_index("Step"))
