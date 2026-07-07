import json
from agentic_ai.agents.base_agent import BaseAgent
from agentic_ai.communication.message_schema import MessageType, Topic


class DataEngineeringLeadAgent(BaseAgent):
    def __init__(self, message_bus, project_state, anthropic_client):
        tools = [
            {
                "name": "assign_de_task",
                "description": "Assign a data engineering task to DE1 or DE2",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "to_agent": {"type": "string", "enum": ["data_engineer_1", "data_engineer_2"]},
                        "task":     {"type": "string"},
                    },
                    "required": ["to_agent", "task"],
                },
            },
            {
                "name": "notify_pm",
                "description": "Send a status update to the Project Manager",
                "input_schema": {
                    "type": "object",
                    "properties": {"summary": {"type": "string"}},
                    "required": ["summary"],
                },
            },
            {
                "name": "approve_pipeline",
                "description": "Approve a pipeline step as complete",
                "input_schema": {
                    "type": "object",
                    "properties": {"pipeline_name": {"type": "string"}, "notes": {"type": "string"}},
                    "required": ["pipeline_name"],
                },
            },
        ]
        super().__init__(
            agent_id="de_lead",
            role="Data Engineering Lead",
            topics=[Topic.DATA_ENGINEERING, Topic.MANAGEMENT],
            message_bus=message_bus,
            project_state=project_state,
            anthropic_client=anthropic_client,
            tools=tools,
        )
        self.system_prompt = (
            "You are the Data Engineering Lead. You manage DE1 and DE2. "
            "When assigned Phase 1 tasks, delegate: DE1 handles ingestion and ETL; "
            "DE2 handles archiving, quality checks, and storage optimization. "
            "Once both report completion, notify the Project Manager that Phase 1 is complete."
        )
        self._de1_done = False
        self._de2_done = False

    async def _handle_task(self, message):
        payload_str = json.dumps(message.payload)
        if message.message_type == MessageType.STATUS_UPDATE:
            if message.from_agent == "data_engineer_1":
                self._de1_done = True
            elif message.from_agent == "data_engineer_2":
                self._de2_done = True

        prompt = (
            f"Message from {message.from_agent} [{message.message_type}]:\n{payload_str}\n"
            f"DE1 done: {self._de1_done}, DE2 done: {self._de2_done}\n"
            "Use your tools to delegate or report as needed."
        )
        await self._run_tool_loop(prompt)

    async def _tool_assign_de_task(self, to_agent: str, task: str) -> str:
        await self.send_message(
            to=to_agent,
            topic=Topic.DATA_ENGINEERING,
            message_type=MessageType.TASK_ASSIGNMENT,
            payload={"task": task},
        )
        return f"Task assigned to {to_agent}: {task}"

    async def _tool_notify_pm(self, summary: str) -> str:
        await self.send_message(
            to="project_manager",
            topic=Topic.MANAGEMENT,
            message_type=MessageType.STATUS_UPDATE,
            payload={"summary": summary, "from_team": "data_engineering"},
        )
        return f"PM notified: {summary}"

    async def _tool_approve_pipeline(self, pipeline_name: str, notes: str = "") -> str:
        await self.project_state.log_activity(self.agent_id, "approve_pipeline",
                                              {"pipeline": pipeline_name, "notes": notes})
        return f"Pipeline '{pipeline_name}' approved"
