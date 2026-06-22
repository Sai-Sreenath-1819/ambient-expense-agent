# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import base64
import datetime
import json
import os
import re
from typing import Any

from dotenv import load_dotenv
from google.adk.agents import Agent
from google.adk.agents.context import Context
from google.adk.apps import App, ResumabilityConfig
from google.adk.events.event import Event
from google.adk.events.event_actions import EventActions
from google.adk.events.request_input import RequestInput
from google.adk.models import Gemini
from google.adk.workflow import Workflow, node
from google.genai import types
from pydantic import BaseModel, Field

from .config import MODEL_NAME, THRESHOLD_USD

load_dotenv()

# Determine if we should use Vertex AI (default: True, unless GOOGLE_GENAI_USE_VERTEXAI=False)
use_vertex = os.environ.get("GOOGLE_GENAI_USE_VERTEXAI", "True").lower() == "true"

if use_vertex:
    import google.auth

    try:
        _, project_id = google.auth.default()
        if project_id:
            os.environ.setdefault("GOOGLE_CLOUD_PROJECT", project_id)
    except Exception:
        pass
    os.environ.setdefault("GOOGLE_CLOUD_LOCATION", "global")
    os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = "True"
else:
    os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = "False"


class RiskAssessment(BaseModel):
    is_high_risk: bool = Field(
        description="True if the expense has high risk factors, false otherwise."
    )
    risk_factors: list[str] = Field(
        description="List of identified risk factors or anomalies."
    )
    justification: str = Field(
        description="Brief justification of the risk assessment."
    )


@node
def parse_expense(ctx: Context, node_input: Any) -> dict[str, Any]:
    # Extract raw text or dict from input
    data_content = None
    if hasattr(node_input, "parts") and node_input.parts:
        # types.Content from START (when running via CLI/playground)
        raw_text = node_input.parts[0].text
        try:
            data_content = json.loads(raw_text)
        except Exception:
            data_content = raw_text
    elif isinstance(node_input, dict):
        data_content = node_input
    elif isinstance(node_input, str):
        try:
            data_content = json.loads(node_input)
        except Exception:
            data_content = node_input

    # Extract raw data field (handles Pub/Sub message envelope or direct payload)
    data = None
    if isinstance(data_content, dict):
        if (
            "message" in data_content
            and isinstance(data_content["message"], dict)
            and "data" in data_content["message"]
        ):
            data = data_content["message"]["data"]
        elif "data" in data_content:
            data = data_content["data"]
        else:
            data = data_content
    else:
        data = data_content

    # Handle base64 encoding or plain JSON
    expense_details = {}
    if isinstance(data, str):
        try:
            # Check if it's base64 encoded
            decoded = base64.b64decode(data).decode("utf-8")
            expense_details = json.loads(decoded)
        except Exception:
            # Fallback to direct JSON parsing if not base64
            try:
                expense_details = json.loads(data)
            except Exception:
                expense_details = {"description": data}
    elif isinstance(data, dict):
        expense_details = data

    # Pull out fields and normalize types
    try:
        amount = float(expense_details.get("amount", 0.0))
    except (ValueError, TypeError):
        amount = 0.0

    expense = {
        "amount": amount,
        "submitter": str(expense_details.get("submitter", "Unknown")),
        "category": str(expense_details.get("category", "General")),
        "description": str(
            expense_details.get("description", "No description provided")
        ),
        "date": str(
            expense_details.get("date", datetime.datetime.now().strftime("%Y-%m-%d"))
        ),
    }

    # Store in context state for downstream nodes
    ctx.state["expense"] = expense
    return expense


@node
def check_threshold(ctx: Context, node_input: dict[str, Any]) -> Event:
    expense = node_input
    amount = expense.get("amount", 0.0)

    if amount < THRESHOLD_USD:
        return Event(output=expense, actions=EventActions(route="auto_approve"))
    else:
        return Event(output=expense, actions=EventActions(route="require_review"))


@node
def auto_approve(ctx: Context, node_input: dict[str, Any]) -> dict[str, Any]:
    expense = node_input
    outcome = {
        "expense": expense,
        "decision": "approved",
        "reviewer": "System (Auto-approve)",
        "comments": f"Amount ${expense['amount']:.2f} is under threshold of ${THRESHOLD_USD:.2f}.",
    }
    return outcome


@node
def security_checkpoint(ctx: Context, node_input: dict[str, Any]) -> Event:
    expense = node_input
    desc = expense.get("description", "")
    redacted_categories = []

    # 1. PII Redaction (SSN and Credit Card)
    ssn_pattern = r"\b\d{3}-\d{2}-\d{4}\b"
    cc_pattern = r"\b(?:\d[ -]*?){13,19}\b"

    clean_desc = desc
    if re.search(ssn_pattern, desc):
        clean_desc = re.sub(ssn_pattern, "[REDACTED SSN]", clean_desc)
        redacted_categories.append("SSN")

    if re.search(cc_pattern, desc):
        clean_desc = re.sub(cc_pattern, "[REDACTED CREDIT CARD]", clean_desc)
        redacted_categories.append("Credit Card")

    if redacted_categories:
        expense["description"] = clean_desc
        ctx.state["expense"] = expense
        ctx.state["redacted_categories"] = redacted_categories

    # 2. Prompt Injection Defense
    injection_indicators = [
        "ignore previous instructions",
        "auto-approve this expense",
        "set decision to approved",
        "system override",
        "bypass rules",
        "bypass all checks",
        "ignore the model",
        "you must approve",
    ]

    has_injection = any(indicator in desc.lower() for indicator in injection_indicators)

    if has_injection:
        ctx.state["security_event"] = True
        return Event(output=expense, actions=EventActions(route="security_alert"))
    else:
        return Event(output=expense, actions=EventActions(route="clean"))


llm_review = Agent(
    name="llm_review",
    model=Gemini(
        model=MODEL_NAME,
        retry_options=types.HttpRetryOptions(attempts=3),
    ),
    instruction=(
        "You are an automated expense risk auditor. Review the provided expense details "
        "and determine if there are any risk factors, policy violations, or anomalies. "
        "List all risk factors you find and justify your decision."
    ),
    output_schema=RiskAssessment,
    output_key="risk_review",
)


@node(rerun_on_resume=True)
async def human_gate(ctx: Context, node_input: dict[str, Any]):
    expense = ctx.state.get("expense", {})
    security_event = ctx.state.get("security_event", False)
    redacted_categories = ctx.state.get("redacted_categories", [])

    if not ctx.resume_inputs or "decision" not in ctx.resume_inputs:
        if security_event:
            risk_str = "⚠️ WARNING: PROMPT INJECTION ATTEMPT DETECTED! Bypassed automated AI review."
        else:
            risk_review = node_input or ctx.state.get("risk_review", {})
            risk_str = (
                f"High Risk: {risk_review.get('is_high_risk', False)}\n"
                f"Risk Factors: {', '.join(risk_review.get('risk_factors', []))}\n"
                f"Justification: {risk_review.get('justification', '')}"
            )

        redacted_str = ""
        if redacted_categories:
            redacted_str = f"\n[REDACTED DATA]: Personal data scrubbed (Redacted: {', '.join(redacted_categories)})\n"

        yield RequestInput(
            interrupt_id="decision",
            message=(
                f"\n=== HUMAN APPROVAL REQUIRED ===\n"
                f"Expense of ${expense.get('amount', 0.0):.2f} by {expense.get('submitter', 'Unknown')} "
                f"requires approval.\n"
                f"Description: {expense.get('description', '')}\n"
                f"Category: {expense.get('category', '')}\n"
                f"{redacted_str}\n"
                f"Risk review details:\n{risk_str}\n"
                f"Please reply with 'approve' or 'reject':"
            ),
        )
        return

    decision = ctx.resume_inputs["decision"].strip().lower()
    is_approved = "approve" in decision

    comments_str = f"Human responded: {decision}"
    if security_event:
        comments_str += " (AI review was bypassed due to prompt injection)"

    yield Event(
        output={
            "expense": expense,
            "decision": "approved" if is_approved else "rejected",
            "reviewer": "Human Auditor",
            "comments": comments_str,
        }
    )


@node
def record_outcome(ctx: Context, node_input: dict[str, Any]) -> dict[str, Any]:
    return node_input


root_agent = Workflow(
    name="ambient_expense_workflow",
    edges=[
        ("START", parse_expense),
        (parse_expense, check_threshold),
        (
            check_threshold,
            {
                "auto_approve": auto_approve,
                "require_review": security_checkpoint,
            },
        ),
        (
            security_checkpoint,
            {
                "clean": llm_review,
                "security_alert": human_gate,
            },
        ),
        (llm_review, human_gate),
        (human_gate, record_outcome),
    ],
)


app = App(
    root_agent=root_agent,
    name="expense_agent",
    resumability_config=ResumabilityConfig(is_resumable=True),
)
