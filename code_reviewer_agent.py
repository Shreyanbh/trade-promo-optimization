"""
Code Reviewer Agent — Technical QA gatekeeper.
Reviews code and design decisions before they are implemented.
Produces structured findings: severity (INFO / WARN / CRITICAL), category, recommendation.
"""
import json

from agentic_ai.agents.base_agent import BaseAgent
from agentic_ai.communication.message_schema import MessageType, Topic
from src.config.settings import PATHS
from src.utils.logger import get_logger

log = get_logger("agent.code_reviewer")

REVIEWER_TOOLS = [
    {
        "name": "submit_review",
        "description": "Submit a structured code or design review with findings and an overall verdict.",
        "input_schema": {
            "type": "object",
            "properties": {
                "review_id":    {"type": "string"},
                "artifact":     {"type": "string", "description": "What was reviewed (e.g. 'ETL pipeline design')"},
                "submitted_by": {"type": "string", "description": "Agent who submitted for review"},
                "phase":        {"type": "string"},
                "findings": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "severity":       {"type": "string", "enum": ["INFO", "WARN", "CRITICAL"]},
                            "category":       {"type": "string"},
                            "finding":        {"type": "string"},
                            "recommendation": {"type": "string"},
                        },
                        "required": ["severity", "category", "finding", "recommendation"],
                    },
                },
                "verdict":   {"type": "string", "enum": ["APPROVED", "APPROVED_WITH_NOTES", "NEEDS_REVISION"]},
                "summary":   {"type": "string"},
            },
            "required": ["review_id", "artifact", "submitted_by", "phase", "findings", "verdict", "summary"],
        },
    },
    {
        "name": "request_changes",
        "description": "Send change requests back to the submitting agent before approval.",
        "input_schema": {
            "type": "object",
            "properties": {
                "to_agent":     {"type": "string"},
                "artifact":     {"type": "string"},
                "change_items": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["to_agent", "artifact", "change_items"],
        },
    },
]


class CodeReviewerAgent(BaseAgent):
    def __init__(self, message_bus, project_state, anthropic_client):
        super().__init__(
            agent_id="code_reviewer",
            role="Code Reviewer / Technical QA",
            topics=[Topic.DATA_SCIENCE.value, Topic.DATA_ENGINEERING.value, Topic.PIPELINE.value],
            message_bus=message_bus,
            project_state=project_state,
            anthropic_client=anthropic_client,
            tools=REVIEWER_TOOLS,
        )
        self.system_prompt = (
            "You are the Technical Code Reviewer and QA gatekeeper. "
            "Every significant piece of code or design decision must pass through your review "
            "before it is implemented. You check for: correctness, edge cases, security, "
            "performance, maintainability, and data science best practices. "
            "Produce structured findings with severity (INFO/WARN/CRITICAL) and clear "
            "actionable recommendations. Be thorough but pragmatic — block only what truly "
            "needs blocking; annotate everything else."
        )

    async def _handle_task(self, message) -> None:
        payload = message.payload or {}
        prompt  = (
            f"You have received a review request from {message.from_agent}:\n"
            f"{json.dumps(payload, indent=2)}\n\n"
            "Review the artifact thoroughly. Identify all issues by severity. "
            "Use submit_review to record your findings and verdict."
        )
        await self._run_tool_loop(prompt)

    def _tool_submit_review(self, review_id: str, artifact: str, submitted_by: str,
                             phase: str, findings: list, verdict: str, summary: str) -> str:
        import os
        record = {
            "review_id":    review_id,
            "artifact":     artifact,
            "submitted_by": submitted_by,
            "phase":        phase,
            "findings":     findings,
            "verdict":      verdict,
            "summary":      summary,
        }
        path = os.path.join(PATHS["reports"], "code_reviews.json")
        existing = []
        if os.path.exists(path):
            with open(path) as f:
                existing = json.load(f)
        existing.append(record)
        with open(path, "w") as f:
            json.dump(existing, f, indent=2)
        return json.dumps({"status": "review_submitted", "verdict": verdict, "review_id": review_id})

    def _tool_request_changes(self, to_agent: str, artifact: str,
                               change_items: list) -> str:
        return json.dumps({
            "status": "change_request_sent",
            "to": to_agent,
            "artifact": artifact,
            "n_items": len(change_items),
        })
