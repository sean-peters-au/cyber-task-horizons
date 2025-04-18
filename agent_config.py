# agent_config.py
from inspect_ai.agent import react
from inspect_ai.tool import bash_session, text_editor

# Placeholder prompt - needs refinement based on modular-public's actual prompt
SYSTEM_PROMPT = """
You are a helpful assistant. Your goal is to complete the given task.
You have access to the following tools: bash_session, text_editor.
Reason step-by-step about your plan.
When you have the final answer or have completed the task, use the submit() tool.
"""

paper_agent = react(
    prompt=SYSTEM_PROMPT,
    tools=[
        bash_session(timeout=7200), # 2-hour timeout for potentially long tasks? Adjust as needed.
        text_editor()
    ],
    # Consider adding attempts > 1 if tasks allow retries, though paper likely used 1.
)