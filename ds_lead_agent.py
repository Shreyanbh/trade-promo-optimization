import json
from agentic_ai.agents.base_agent import BaseAgent
from agentic_ai.communication.message_schema import MessageType, Topic


class DataScienceLeadAgent(BaseAgent):
    def __init__(self, message_bus, project_state, anthropic_client):
        tools = [
            {
                "name": "assign_ds_task",
                "description": "Assign a data science task to DS1 or DS2",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "to_agent": {"type": "string", "enum": ["data_scientist_1", "data_scientist_2"]},
                        "task":     {"type": "string"},
                        "phase":    {"type": "string"},
                    },
                    "required": ["to_agent", "task", "phase"],
                },
            },
            {
                "name": "approve_eda",
                "description": "Review and approve EDA output",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "report_path": {"type": "string"},
                        "approved":    {"type": "boolean"},
                        "feedback":    {"type": "string"},
                    },
                    "required": ["report_path", "approved"],
                },
            },
            {
                "name": "approve_features",
                "description": "Approve the feature list for modelling",
                "input_schema": {
                    "type": "object",
                    "properties": {"feature_list": {"type": "array", "items": {"type": "string"}}},
                    "required": ["feature_list"],
                },
            },
            {
                "name": "approve_model",
                "description": "Approve a trained model based on metrics",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "model_name": {"type": "string"},
                        "metrics":    {"type": "object"},
                        "approved":   {"type": "boolean"},
                    },
                    "required": ["model_name", "approved"],
                },
            },
            {
                "name": "notify_pm",
                "description": "Notify the Project Manager of phase completion",
                "input_schema": {
                    "type": "object",
                    "properties": {"summary": {"type": "string"}, "phase": {"type": "string"}},
                    "required": ["summary", "phase"],
                },
            },
        ]
        super().__init__(
            agent_id="ds_lead",
            role="Data Science Lead",
            topics=[Topic.DATA_SCIENCE, Topic.MANAGEMENT],
            message_bus=message_bus,
            project_state=project_state,
            anthropic_client=anthropic_client,
            tools=tools,
        )
        self.system_prompt = (
            "You are the Data Science Lead. You manage DS1 and DS2. "
            "Phase 2: Assign EDA to DS1, review output, then notify PM. "
            "Phase 3: Assign feature engineering to DS1, approve features, notify PM. "
            "Phase 4: Assign segmentation and recommender to DS2, review metrics, approve models, notify PM. "
            "Always review outputs before approving. Provide constructive feedback if not approved."
        )

    async def _handle_task(self, message):
        payload_str = json.dumps(message.payload)
        prompt = (
            f"Message from {message.from_agent} [{message.message_type}]:\n{payload_str}\n"
            "Decide how to handle this: delegate to a team member, approve an output, or notify PM."
        )
        await self._run_tool_loop(prompt)

    async def _tool_assign_ds_task(self, to_agent: str, task: str, phase: str) -> str:
        await self.send_message(
            to=to_agent,
            topic=Topic.DATA_SCIENCE,
            message_type=MessageType.TASK_ASSIGNMENT,
            payload={"task": task, "phase": phase},
        )
        return f"Task assigned to {to_agent}: {task}"

    async def _tool_approve_eda(self, report_path: str, approved: bool, feedback: str = "") -> str:
        await self.project_state.log_activity(self.agent_id, "approve_eda",
                                              {"report": report_path, "approved": approved, "feedback": feedback})
        return f"EDA {'approved' if approved else 'rejected'}: {feedback}"

    async def _tool_approve_features(self, feature_list: list[str]) -> str:
        await self.project_state.update("approved_features", feature_list)
        return f"Approved {len(feature_list)} features"

    async def _tool_approve_model(self, model_name: str, approved: bool, metrics: dict = None) -> str:
        if approved and metrics:
            await self.project_state.register_model(model_name, {
                "approved": True, "approved_by": self.agent_id, "metrics": metrics
            })
        return f"Model '{model_name}' {'approved' if approved else 'rejected'}"

    async def _tool_notify_pm(self, summary: str, phase: str) -> str:
        await self.send_message(
            to="project_manager",
            topic=Topic.MANAGEMENT,
            message_type=MessageType.STATUS_UPDATE,
            payload={"summary": summary, "phase": phase, "from_team": "data_science"},
        )
        return f"PM notified: {summary}"
