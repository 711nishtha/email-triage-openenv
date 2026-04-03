# Email Triage OpenEnv — User Manual

Welcome to the **Advanced Enterprise Email Triage** interactive interface. This guide will help you understand how to use the Streamlit UI to manually triage emails, use tools, and monitor performance.

---

## Getting Started

1. **Access the UI**: Open the Streamlit app URL (locally at `http://localhost:7860` or on Hugging Face Spaces).
2. **Select Difficulty**: In the left sidebar, use the dropdown to choose between **Easy**, **Medium**, or **Hard**.
3. **Start Episode**: Click the **"🔄 Reset / Start Episode"** button to load a fresh set of emails and begin.

---

## Interface Overview

The interface is divided into three main sections:
1. **Sidebar (Control Panel)**: Episode status, rewards, and task selection.
2. **Left Column (Email Viewer)**: Content and metadata of the current email.
3. **Right Column (Action Panel)**: Tools and forms to submit your triage decisions.

---

##  1. Email Viewer

This panel displays the email currently at the top of your inbox.

- **Email Card**: Shows the Subject, From/To addresses, Timestamp, and the full Body.
- **⚠️ Security Warning**: If an email is from a suspicious domain or has a low reputation score, a red warning and border will appear.
- **Metadata Tabs**:
    - **Details**: Shows Email ID, extracted links, and attachment names.
    - **Thread History**: (Medium/Hard) Shows prior emails in the same conversation for context.
    - **Sender Profile**: Deep dive into the sender's reputation, job title, and VIP status.
    - **Tools**: Interactive buttons to run mock tools (e.g., `calendar_check`).

---

##  2. Action Panel

Use this panel to process the current email. Select an **Action Type** at the top to change the form fields.

### Action Types:
| Action | Purpose | Fields |
| :--- | :--- | :--- |
| **Categorize** | Quickly assign a functional category. | Category Dropdown |
| **Route** | Send the email to a specific department. | Target Team Dropdown |
| **Flag** | Set priority and mark security threats. | Priority Dropdown + Phishing Checkbox |
| **Respond** | Draft a reply to the sender. | Text Area for Draft |
| **Escalate** | Send directly to Security or Legal. | Target + Reason for Escalation |
| **Ignore** | Archive the email as low priority. | (Auto-fills with Archive/Low) |

### Advanced Settings:
- **Additional Triage Details**: Expand this to manually override Category, Priority, and Route simultaneously.
- **Confidence Score**: A slider (0.0 to 1.0) to indicate how sure you are about your decision. This is logged for performance analysis.
- **Submit Action**: Click this to process the email. The UI will then load the next email in the inbox.

---

##  3. Feedback & History

- **Last Step Feedback**: After every submission, you'll see a color-coded list of "signals."
    - 🟢 **Green**: Correct decisions (Bonus rewards).
    - 🔴 **Red**: Mistakes or missed threats (Penalty points).
    - 🔵 **Blue**: Informational notes from the grader.
- **Episode History**: Expand this at the bottom to see a log of every action taken in the current session.

---

##  4. Completion Summary

Once all emails are processed or the step limit is reached:
- **Success Card**: Shows your **Final Score** (normalized 0.0 to 1.0).
- **Performance Breakdown**: A detailed JSON view of how the grader scored your performance across various metrics (Priority, Phishing detection, etc.).
- **Play Again**: Click to reset and try a different task.

---

## 💡 Pro Tips
- **Hard Task**: Always check the **Sender Profile** tab. Phishing attempts often use spoofed domains that look nearly identical to internal ones.
- **Medium Task**: Check **Thread History** before deciding on a route; the previous conversation often holds the key to the correct department.
- **Tools**: Running a tool costs a small "efficiency" penalty if used unnecessarily, but provides critical info for Medium and Hard tasks.
