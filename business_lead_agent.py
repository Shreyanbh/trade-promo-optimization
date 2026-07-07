import json
from agentic_ai.agents.base_agent import BaseAgent
from agentic_ai.communication.message_schema import MessageType, Topic


class BusinessLeadAgent(BaseAgent):
    def __init__(self, message_bus, project_state, anthropic_client):
        tools = [
            {
                "name": "define_kpis",
                "description": "Define business KPIs for the project",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "kpis": {
                            "type": "object",
                            "description": "Dict of KPI name to description/target",
                        }
                    },
                    "required": ["kpis"],
                },
            },
            {
                "name": "review_segment_definitions",
                "description": "Review customer segment profiles from a business perspective",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "segment_summary": {"type": "string"},
                        "approved":        {"type": "boolean"},
                        "feedback":        {"type": "string"},
                    },
                    "required": ["segment_summary", "approved"],
                },
            },
            {
                "name": "approve_report",
                "description": "Approve the final business report",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "report_path": {"type": "string"},
                        "approved":    {"type": "boolean"},
                    },
                    "required": ["report_path", "approved"],
                },
            },
            {
                "name": "assign_ba_task",
                "description": "Assign a task to BA1 or BA2",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "to_agent": {"type": "string", "enum": ["business_analyst_1", "business_analyst_2"]},
                        "task":     {"type": "string"},
                    },
                    "required": ["to_agent", "task"],
                },
            },
            {
                "name": "notify_pm",
                "description": "Notify the Project Manager",
                "input_schema": {
                    "type": "object",
                    "properties": {"summary": {"type": "string"}},
                    "required": ["summary"],
                },
            },
        ]
        super().__init__(
            agent_id="business_lead",
            role="Business Lead",
            topics=[Topic.BUSINESS, Topic.MANAGEMENT],
            message_bus=message_bus,
            project_state=project_state,
            anthropic_client=anthropic_client,
            tools=tools,
        )
        self.system_prompt = (
            "You are the Business Lead. You translate business requirements into project goals. "
            "At project start: define KPIs (e.g., promo lift %, customer retention rate, segment revenue). "
            "Phase 4: Review segment definitions and approve from a business perspective. "
            "Phase 5: Assign reporting tasks to BA1 and BA2, review and approve final report, notify PM."
        )

    async def _handle_task(self, message):
        payload_str = json.dumps(message.payload)
        prompt = (
            f"Message from {message.from_agent} [{message.message_type}]:\n{payload_str}\n"
            "Take appropriate business action: define KPIs, review segments, approve reports, or delegate to BAs."
        )
        await self._run_tool_loop(prompt)

    async def _tool_define_kpis(self, kpis: dict) -> str:
        await self.project_state.update("kpis", kpis)
        return f"KPIs defined: {list(kpis.keys())}"

    async def _tool_review_segment_definitions(self, segment_summary: str, approved: bool, feedback: str = "") -> str:
        await self.project_state.log_activity(self.agent_id, "review_segments",
                                              {"approved": approved, "feedback": feedback})
        return f"Segments {'approved' if approved else 'rejected'}: {feedback}"

    async def _tool_approve_report(self, report_path: str, approved: bool) -> str:
        await self.project_state.log_activity(self.agent_id, "approve_report",
                                              {"report": report_path, "approved": approved})
        return f"Report {'approved' if approved else 'rejected'}: {report_path}"

    async def _tool_assign_ba_task(self, to_agent: str, task: str) -> str:
        await self.send_message(
            to=to_agent,
            topic=Topic.BUSINESS,
            message_type=MessageType.TASK_ASSIGNMENT,
            payload={"task": task},
        )
        return f"Task assigned to {to_agent}: {task}"

    async def _tool_notify_pm(self, summary: str) -> str:
        await self.send_message(
            to="project_manager",
            topic=Topic.MANAGEMENT,
            message_type=MessageType.STATUS_UPDATE,
            payload={"summary": summary, "from_team": "business"},
        )
        return f"PM notified: {summary}"
