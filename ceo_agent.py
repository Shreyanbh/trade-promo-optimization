"""
CEO Agent — Executive Sponsor and final approver.
All phase leads must obtain CEO approval before implementing changes.
"""
import json
from datetime import datetime, timezone

from agentic_ai.agents.base_agent import BaseAgent
from agentic_ai.communication.message_schema import MessageType, Topic
from src.config.settings import PATHS
from src.utils.logger import get_logger

log = get_logger("agent.ceo")

CEO_TOOLS = [
    {
        "name": "approve_phase",
        "description": "Approve a phase or initiative proposed by a lead. Records the decision with rationale and any conditions.",
        "input_schema": {
            "type": "object",
            "properties": {
                "phase":      {"type": "string", "description": "Phase or initiative name (e.g. phase1, model_deployment)"},
                "requestor":  {"type": "string", "description": "Agent ID of the lead requesting approval"},
                "decision":   {"type": "string", "enum": ["APPROVED", "APPROVED_WITH_CONDITIONS", "REJECTED"]},
                "rationale":  {"type": "string", "description": "CEO rationale for the decision"},
                "conditions": {"type": "string", "description": "Any conditions attached to the approval (empty string if none)"},
            },
            "required": ["phase", "requestor", "decision", "rationale"],
        },
    },
    {
        "name": "request_more_info",
        "description": "Ask a lead for additional information before making a decision.",
        "input_schema": {
            "type": "object",
            "properties": {
                "to_agent":  {"type": "string", "description": "Agent ID to ask"},
                "question":  {"type": "string", "description": "What the CEO needs to know"},
            },
            "required": ["to_agent", "question"],
        },
    },
    {
        "name": "issue_directive",
        "description": "Broadcast a strategic directive to the entire team.",
        "input_schema": {
            "type": "object",
            "properties": {
                "directive": {"type": "string", "description": "The strategic instruction"},
                "priority":  {"type": "integer", "minimum": 1, "maximum": 5},
            },
            "required": ["directive"],
        },
    },
    {
        "name": "review_business_case",
        "description": "Review the business case and financial projections before approving deployment.",
        "input_schema": {
            "type": "object",
            "properties": {
                "total_revenue":    {"type": "number"},
                "net_benefit":      {"type": "number"},
                "roi_pct":          {"type": "number"},
                "n_segments":       {"type": "integer"},
                "recommendation":   {"type": "string", "description": "CEO recommendation"},
            },
            "required": ["total_revenue", "net_benefit", "roi_pct", "n_segments", "recommendation"],
        },
    },
]


class CEOAgent(BaseAgent):
    def __init__(self, message_bus, project_state, anthropic_client):
        super().__init__(
            agent_id="ceo",
            role="Chief Executive Officer",
            topics=[Topic.MANAGEMENT.value],
            message_bus=message_bus,
            project_state=project_state,
            anthropic_client=anthropic_client,
            tools=CEO_TOOLS,
        )
        self.system_prompt = (
            "You are the CEO and executive sponsor of the Trade Promo Optimisation project. "
            "Your role is to provide strategic oversight, approve phase transitions, and ensure "
            "the project delivers measurable business value. You must be consulted and give explicit "
            "approval before any lead proceeds with a major implementation phase. "
            "You ask sharp business questions, set conditions on approvals, and issue directives "
            "to keep the project on track. Be decisive but demanding — quality and ROI matter."
        )
        self._approvals: list[dict] = []

    async def _handle_task(self, message) -> None:
        payload = message.payload or {}
        prompt  = (
            f"You received an approval request from {message.from_agent}:\n"
            f"{json.dumps(payload, indent=2)}\n\n"
            "Review the request carefully. Ask for more information if needed, then make a decision "
            "using the approve_phase tool. Always attach conditions if you have concerns."
        )
        await self._run_tool_loop(prompt)

    def _tool_approve_phase(self, phase: str, requestor: str, decision: str,
                             rationale: str, conditions: str = "") -> str:
        import os
        record = {
            "timestamp":   datetime.now(timezone.utc).isoformat(),
            "phase":       phase,
            "requestor":   requestor,
            "decision":    decision,
            "rationale":   rationale,
            "conditions":  conditions,
        }
        self._approvals.append(record)
        # Persist to outputs
        import json as _json
        path = os.path.join(PATHS["reports"], "ceo_approvals.json")
        existing = []
        if os.path.exists(path):
            with open(path) as f:
                existing = _json.load(f)
        existing.append(record)
        with open(path, "w") as f:
            _json.dump(existing, f, indent=2)
        return _json.dumps({"status": "recorded", "decision": decision, "phase": phase})

    def _tool_request_more_info(self, to_agent: str, question: str) -> str:
        return json.dumps({"status": "sent", "to": to_agent, "question": question})

    def _tool_issue_directive(self, directive: str, priority: int = 3) -> str:
        return json.dumps({"status": "broadcast", "directive": directive, "priority": priority})

    def _tool_review_business_case(self, total_revenue: float, net_benefit: float,
                                    roi_pct: float, n_segments: int,
                                    recommendation: str) -> str:
        verdict = "APPROVED" if roi_pct >= 100 else "NEEDS_REVIEW"
        return json.dumps({
            "status": verdict,
            "total_revenue": total_revenue,
            "net_benefit": net_benefit,
            "roi_pct": roi_pct,
            "n_segments": n_segments,
            "recommendation": recommendation,
        })
