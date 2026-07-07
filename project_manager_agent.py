import json
from agentic_ai.agents.base_agent import BaseAgent
from agentic_ai.communication.message_schema import MessageType, Topic


class ProjectManagerAgent(BaseAgent):
    def __init__(self, message_bus, project_state, anthropic_client):
        tools = [
            {
                "name": "assign_task",
                "description": "Assign a task to a specific agent",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "to_agent":          {"type": "string"},
                        "task_description":  {"type": "string"},
                        "priority":          {"type": "integer"},
                        "phase":             {"type": "string"},
                    },
                    "required": ["to_agent", "task_description", "phase"],
                },
            },
            {
                "name": "check_project_status",
                "description": "Check overall project milestone completion",
                "input_schema": {"type": "object", "properties": {}},
            },
            {
                "name": "update_milestone",
                "description": "Mark a project milestone as complete",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "phase":  {"type": "string"},
                        "status": {"type": "string"},
                    },
                    "required": ["phase", "status"],
                },
            },
            {
                "name": "broadcast_message",
                "description": "Broadcast a message to all agents",
                "input_schema": {
                    "type": "object",
                    "properties": {"message": {"type": "string"}},
                    "required": ["message"],
                },
            },
        ]
        super().__init__(
            agent_id="project_manager",
            role="Project Manager",
            topics=[Topic.ALL, Topic.MANAGEMENT],
            message_bus=message_bus,
            project_state=project_state,
            anthropic_client=anthropic_client,
            tools=tools,
        )
        self.system_prompt = (
            "You are the Project Manager for a Trade Promo Optimization and Customer Recommendation "
            "Segmentation project. Your job is to orchestrate all team agents, assign work in the "
            "correct phase order (Phase 1→5), track milestones, resolve blockers, and ensure the "
            "project completes successfully.\n\n"
            "Phase order:\n"
            "1. Data Ingestion (assign to de_lead)\n"
            "2. Data Processing + EDA (assign to ds_lead)\n"
            "3. Feature Engineering (assign to ds_lead)\n"
            "4. Segmentation + Recommendation Models (assign to ds_lead, then business_lead for review)\n"
            "5. Export + Reporting + Dashboard (assign to data_scientist_2, business_analyst_1, business_analyst_2)\n\n"
            "When you receive a 'project_start' broadcast, begin assigning Phase 1 tasks. "
            "When a phase completes, assign the next phase. When all phases complete, broadcast 'project_complete'."
        )

    async def _handle_task(self, message):
        payload_str = json.dumps(message.payload)
        prompt = (
            f"Message from {message.from_agent} [{message.message_type}]:\n{payload_str}\n\n"
            "Decide what action to take next and use your tools to assign tasks, update milestones, "
            "or broadcast messages as appropriate."
        )
        await self._run_tool_loop(prompt)

    async def _tool_assign_task(self, to_agent: str, task_description: str, phase: str, priority: int = 3) -> str:
        await self.send_message(
            to=to_agent,
            topic=Topic.MANAGEMENT,
            message_type=MessageType.TASK_ASSIGNMENT,
            payload={"task": task_description, "phase": phase},
            priority=priority,
        )
        return f"Task assigned to {to_agent}: {task_description}"

    async def _tool_check_project_status(self) -> str:
        snap = self.project_state.snapshot()
        return json.dumps(snap)

    async def _tool_update_milestone(self, phase: str, status: str) -> str:
        await self.project_state.set_milestone(phase, status)
        snap = self.project_state.snapshot()
        complete = all(v == "complete" for v in snap["milestones"].values())
        if complete:
            self.project_state.project_complete = True
            await self.send_message(
                to=Topic.ALL,
                topic=Topic.ALL,
                message_type=MessageType.BROADCAST,
                payload={"message": "project_complete"},
                priority=1,
            )
        return f"Milestone {phase} → {status}"

    async def _tool_broadcast_message(self, message: str) -> str:
        await self.send_message(
            to=Topic.ALL,
            topic=Topic.ALL,
            message_type=MessageType.BROADCAST,
            payload={"message": message},
        )
        return f"Broadcast: {message}"
