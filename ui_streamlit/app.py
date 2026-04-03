import streamlit as st
import requests
import json
import os
from datetime import datetime

# Configuration
BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")

# Page Config
st.set_page_config(
    page_title="Email Triage OpenEnv UI",
    page_icon="📧",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        background-color: #f8f9fa;
    }
    .stButton>button {
        width: 100%;
        border-radius: 5px;
        height: 3em;
        background-color: #2C3E50;
        color: white;
    }
    .stButton>button:hover {
        background-color: #34495e;
        color: white;
    }
    .email-card {
        padding: 1.5rem;
        border-radius: 10px;
        background-color: white;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin-bottom: 1rem;
        border-left: 5px solid #2C3E50;
    }
    .phishing-warning {
        border-left: 5px solid #E74C3C !important;
        background-color: #FDEDEC;
    }
    .importance-high { color: #E74C3C; font-weight: bold; }
    .importance-medium { color: #F39C12; font-weight: bold; }
    .importance-low { color: #27AE60; font-weight: bold; }
    </style>
    """, unsafe_allow_html=True)

# Session State Initialization
if "episode_id" not in st.session_state:
    st.session_state.episode_id = None
if "observation" not in st.session_state:
    st.session_state.observation = None
if "total_reward" not in st.session_state:
    st.session_state.total_reward = 0.0
if "step_count" not in st.session_state:
    st.session_state.step_count = 0
if "history" not in st.session_state:
    st.session_state.history = []
if "last_reward" not in st.session_state:
    st.session_state.last_reward = None
if "last_feedback" not in st.session_state:
    st.session_state.last_feedback = []
if "done" not in st.session_state:
    st.session_state.done = False

# API Helpers
def reset_env(task_id):
    try:
        resp = requests.post(f"{BACKEND_URL}/reset", json={"task_id": task_id})
        if resp.status_code == 200:
            data = resp.json()
            st.session_state.observation = data
            st.session_state.episode_id = data.get("info", {}).get("episode_id", "unknown")
            st.session_state.total_reward = 0.0
            st.session_state.step_count = 0
            st.session_state.history = []
            st.session_state.last_reward = None
            st.session_state.last_feedback = []
            st.session_state.done = False
            return True
        else:
            st.error(f"Reset failed: {resp.text}")
    except Exception as e:
        st.error(f"Connection error: {e}")
    return False

def step_env(action):
    try:
        resp = requests.post(f"{BACKEND_URL}/step", json=action)
        if resp.status_code == 200:
            data = resp.json()
            st.session_state.observation = data["observation"]
            st.session_state.total_reward = data["observation"].get("current_score", st.session_state.total_reward)
            st.session_state.step_count = data["observation"].get("step", st.session_state.step_count)
            st.session_state.last_reward = data["reward"]
            st.session_state.last_feedback = data["info"].get("signals", [])
            st.session_state.done = data["done"]
            
            # Update history
            st.session_state.history.append({
                "step": st.session_state.step_count,
                "action": action,
                "reward": data["reward"],
                "signals": data["info"].get("signals", [])
            })
            
            if st.session_state.done:
                st.session_state.final_info = data["info"]
            return True
        else:
            st.error(f"Step failed: {resp.text}")
    except Exception as e:
        st.error(f"Connection error: {e}")
    return False

# Sidebar - Task Selection & Info
with st.sidebar:
    st.title("📧 Email Triage Control")
    task_option = st.selectbox("Select Task Difficulty", ["easy", "medium", "hard"], index=0)
    if st.button("🔄 Reset / Start Episode"):
        reset_env(task_option)
    
    st.divider()
    if st.session_state.episode_id:
        st.info(f"**Episode ID:** `{st.session_state.episode_id}`")
        st.metric("Step", f"{st.session_state.step_count} / {st.session_state.observation.get('max_steps', '?')}")
        st.metric("Total Reward", f"{st.session_state.total_reward:.4f}")
        
        if st.session_state.last_reward is not None:
            color = "normal" if st.session_state.last_reward >= 0 else "inverse"
            st.metric("Last Reward", f"{st.session_state.last_reward:+.4f}", delta_color=color)

# Main UI Layout
if not st.session_state.observation:
    st.info("Welcome! Please select a task and click 'Reset / Start Episode' to begin.")
else:
    obs = st.session_state.observation
    inbox = obs.get("inbox", [])
    
    if st.session_state.done:
        st.balloons()
        st.success("### 🏁 Episode Complete!")
        final_info = getattr(st.session_state, "final_info", {})
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Final Score", f"{final_info.get('final_score', 0.0):.4f}")
        col2.metric("Total Steps", st.session_state.step_count)
        
        if "grader_breakdown" in final_info:
            st.write("#### Performance Breakdown")
            st.json(final_info["grader_breakdown"])
        
        if st.button("Play Again"):
            reset_env(task_option)
            st.rerun()
            
    elif not inbox:
        st.warning("No emails in inbox. You might be done!")
        if st.button("Finish Episode"):
            step_env({"action_type": "done"})
            st.rerun()
    else:
        # Columns: Email Viewer | Action Panel
        col_view, col_action = st.columns([1.5, 1])
        
        current_email = inbox[0]
        sender_profile = obs.get("sender_profiles", {}).get(current_email["sender"], {})
        
        with col_view:
            st.subheader("📬 Current Email")
            
            is_phishing_suspect = sender_profile.get("is_flagged_suspicious", False) or sender_profile.get("reputation_score", 1.0) < 0.1
            card_class = "phishing-warning" if is_phishing_suspect else ""
            
            with st.container():
                st.markdown(f"""
                <div class="email-card {card_class}">
                    <h3>{current_email['subject']}</h3>
                    <p><b>From:</b> {current_email['sender_display_name']} &lt;{current_email['sender']}&gt;</p>
                    <p><b>To:</b> {', '.join(current_email['recipients'])}</p>
                    <p><b>Timestamp:</b> {current_email['timestamp']}</p>
                    <hr>
                    <p style="white-space: pre-wrap;">{current_email['body']}</p>
                </div>
                """, unsafe_allow_html=True)
                
                if is_phishing_suspect:
                    st.error("⚠️ **SECURITY WARNING:** This sender is flagged as suspicious or has very low reputation.")
            
            # Metadata & Tools
            tabs = st.tabs(["Details", "Thread History", "Sender Profile", "Tools"])
            
            with tabs[0]:
                st.write(f"**Email ID:** `{current_email['email_id']}`")
                if current_email.get('links'):
                    st.write("**Links found:**")
                    for link in current_email['links']:
                        st.code(link)
                if current_email.get('attachment_names'):
                    st.write(f"**Attachments:** {', '.join(current_email['attachment_names'])}")

            with tabs[1]:
                thread_id = current_email.get("thread_id")
                if thread_id and thread_id in obs.get("thread_history", {}):
                    history = obs["thread_history"][thread_id]
                    for msg in history:
                        with st.expander(f"RE: {msg['subject']} ({msg['timestamp']})"):
                            st.write(f"**From:** {msg['sender_display_name']}")
                            st.write(msg['body'])
                else:
                    st.write("No thread history available.")
            
            with tabs[2]:
                if sender_profile:
                    st.json(sender_profile)
                else:
                    st.write("No profile data for this sender.")
            
            with tabs[3]:
                st.write("#### Available Mock Tools")
                available_tools = obs.get("available_tools", [])
                if not available_tools:
                    st.write("No tools available for this task.")
                else:
                    for tool in available_tools:
                        if st.button(f"🔧 Run {tool}", key=f"tool_{tool}"):
                            # We can simulate tool call by sending a 'use_tool' step
                            # or just show a mock result. Human UI might prefer seeing result first.
                            # But to get reward/feedback, we must send it to backend.
                            params = {}
                            if tool == "calendar_check":
                                params = {"user": current_email['sender'], "date": "Thursday"}
                            elif tool == "kb_search":
                                params = {"query": current_email['subject']}
                            elif tool == "sender_lookup":
                                params = {"email": current_email['sender']}
                            
                            step_env({
                                "action_type": "use_tool",
                                "tool_name": tool,
                                "tool_params": params
                            })
                            st.rerun()

        with col_action:
            st.subheader("🛠️ Take Action")
            
            # Action selection
            action_mode = st.radio(
                "Action Type",
                ["Categorize", "Route", "Flag", "Respond", "Escalate", "Ignore"],
                horizontal=True
            )
            
            with st.form("action_form"):
                # Dynamic inputs based on action_mode
                email_id = current_email['email_id']
                payload = {"email_id": email_id}
                
                if action_mode == "Categorize":
                    payload["action_type"] = "triage"
                    payload["category"] = st.selectbox("Select Category", [
                        "security_incident", "executive_request", "hr_matter", "vendor_contract", 
                        "it_support", "team_update", "customer_escalation", "phishing", "newsletter", "other"
                    ])
                
                elif action_mode == "Route":
                    payload["action_type"] = "triage"
                    payload["route_to"] = st.selectbox("Select Target Team", [
                        "security_team", "ceo_office", "cfo_office", "hr_team", "legal_team", 
                        "it_helpdesk", "engineering_team", "sales_team", "finance_team", 
                        "operations_team", "customer_success", "executive_assistant", 
                        "archive", "spam_folder"
                    ])
                
                elif action_mode == "Flag":
                    payload["action_type"] = "triage"
                    payload["priority"] = st.selectbox("Priority Level", ["critical", "high", "medium", "low", "spam"])
                    is_phish = st.checkbox("Mark as Security Threat (Phishing)")
                    if is_phish:
                        payload["category"] = "phishing"
                        payload["route_to"] = "security_team"
                
                elif action_mode == "Respond":
                    payload["action_type"] = "triage"
                    reply_text = st.text_area("Draft Reply", placeholder="Write your response here...")
                    payload["reasoning"] = f"RESPONSE DRAFT: {reply_text}"
                
                elif action_mode == "Escalate":
                    payload["action_type"] = "escalate"
                    payload["escalation_target"] = st.selectbox("Escalate To", ["security_team", "ceo_office", "legal_team"])
                    payload["escalation_reason"] = st.text_input("Reason for escalation")
                
                elif action_mode == "Ignore":
                    payload["action_type"] = "triage"
                    payload["priority"] = "low"
                    payload["category"] = "other"
                    payload["route_to"] = "archive"
                
                # Common fields for triage actions
                if payload.get("action_type") == "triage":
                    with st.expander("Additional Triage Details (Optional)"):
                        if "priority" not in payload:
                            payload["priority"] = st.selectbox("Priority", ["critical", "high", "medium", "low", "spam"], index=2)
                        if "category" not in payload:
                            payload["category"] = st.selectbox("Category", [
                                "other", "security_incident", "executive_request", "hr_matter", 
                                "vendor_contract", "it_support", "team_update", "customer_escalation", 
                                "phishing", "newsletter"
                            ], index=0)
                        if "route_to" not in payload:
                            payload["route_to"] = st.selectbox("Route To", [
                                "archive", "security_team", "ceo_office", "cfo_office", "hr_team", 
                                "legal_team", "it_helpdesk", "engineering_team", "sales_team", 
                                "finance_team", "operations_team", "customer_success", 
                                "executive_assistant", "spam_folder"
                            ], index=0)
                        payload["reasoning"] = st.text_area("Reasoning / Internal Note", value=payload.get("reasoning", ""))

                confidence = st.slider("Confidence Score", 0.0, 1.0, 0.8)
                
                submitted = st.form_submit_button("🚀 Submit Action")
                if submitted:
                    if payload.get("action_type") == "triage":
                        current_reasoning = payload.get("reasoning", "")
                        payload["reasoning"] = f"[Confidence: {confidence:.2f}] {current_reasoning}".strip()
                    
                    if step_env(payload):
                        st.rerun()

            # Feedback from last step
            if st.session_state.last_feedback:
                st.write("---")
                st.write("#### 📝 Last Step Feedback")
                for signal in st.session_state.last_feedback:
                    if signal.startswith("+"):
                        st.success(signal)
                    elif signal.startswith("-") or "NOT flagged" in signal or "MISSED" in signal:
                        st.error(signal)
                    else:
                        st.info(signal)

    # History Log at the bottom
    with st.expander("📜 Episode History"):
        if not st.session_state.history:
            st.write("No actions taken yet.")
        else:
            for item in reversed(st.session_state.history):
                st.write(f"**Step {item['step']}** | Action: `{item['action']['action_type']}` | Reward: `{item['reward']:+.4f}`")
                if item['signals']:
                    st.caption("Signals: " + " | ".join(item['signals']))
