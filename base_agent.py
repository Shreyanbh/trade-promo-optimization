import asyncio
import json
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from typing import Any

import anthropic

from agentic_ai.communication.message_bus import MessageBus
from agentic_ai.communication.message_schema import (
    AgentStatus, Message, MessageType, Topic
)
from agentic_ai.state.project_state import ProjectState
from src.config.settings import ANTHROPIC_MODEL
from src.utils.logger import get_logger


class BaseAgent(ABC):
    def __init__(
        self,
        agent_id: str,
        role: str,
        topics: list[str],
        message_bus: MessageBus,
        project_state: ProjectState,
        anthropic_client: anthropic.Anthropic,
        tools: list[dict],
    ):
        self.agent_id = agent_id
        self.role = role
        self.message_bus = message_bus
        self.project_state = project_state
        self.client = anthropic_client
        self.tools = tools
        self.system_prompt: str = ""
        self.activity_log: list[dict] = []
        self.status = AgentStatus.IDLE
        self._conversation: list[dict] = []

        self.inbox: asyncio.Queue = message_bus.register(agent_id, topics)
        self.log = get_logger(f"agent.{agent_id}")

    async def run(self) -> None:
        self.log.info(f"[{self.agent_id}] started — {self.role}")
        await self.project_state.set_agent_status(self.agent_id, AgentStatus.IDLE)
        while not self.project_state.project_complete:
            try:
                msg: Message = await asyncio.wait_for(self.inbox.get(), timeout=2.0)
                await self._process_message(msg)
            except asyncio.TimeoutError:
                continue
        self.log.info(f"[{self.agent_id}] shutting down — project complete")

    async def _process_message(self, message: Message) -> None:
        self.status = AgentStatus.WORKING
        await self.project_state.set_agent_status(self.agent_id, AgentStatus.WORKING)
        self._log("received_message", {
            "from": message.from_agent,
            "type": message.message_type,
            "topic": message.topic,
        })
        await self._handle_task(message)
        self.status = AgentStatus.IDLE
        await self.project_state.set_agent_status(self.agent_id, AgentStatus.IDLE)

    @abstractmethod
    async def _handle_task(self, message: Message) -> None:
        """Each agent implements its own decision logic here."""

    async def send_message(
        self,
        to: str,
        topic: str,
        message_type: str,
        payload: dict,
        priority: int = 3,
        correlation_id: str = "",
    ) -> None:
        msg = Message(
            from_agent=self.agent_id,
            to_agent=to,
            topic=topic,
            message_type=message_type,
            payload=payload,
            priority=priority,
            correlation_id=correlation_id,
        )
        await self.message_bus.publish(msg)
        self._log("sent_message", {"to": to, "type": message_type, "topic": topic})

    async def _call_llm(
        self,
        user_content: str,
        reset_conversation: bool = False,
    ) -> anthropic.types.Message:
        if reset_conversation:
            self._conversation = []
        self._conversation.append({"role": "user", "content": user_content})

        response = self.client.messages.create(
            model=ANTHROPIC_MODEL,
            max_tokens=4096,
            system=self.system_prompt,
            tools=self.tools if self.tools else [],
            messages=self._conversation,
        )
        self._conversation.append({"role": "assistant", "content": response.content})
        return response

    async def _run_tool_loop(self, initial_prompt: str) -> str:
        """
        Runs the standard Anthropic tool_use loop:
        call LLM → execute any tool_use blocks → re-call → repeat until end_turn.
        Returns the final text response.
        """
        response = await self._call_llm(initial_prompt, reset_conversation=True)

        while response.stop_reason == "tool_use":
            tool_results = []
            for block in response.content:
                if block.type == "tool_use":
                    result = await self._execute_tool(block.name, block.input)
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": str(result),
                    })

            self._conversation.append({"role": "user", "content": tool_results})
            response = self.client.messages.create(
                model=ANTHROPIC_MODEL,
                max_tokens=4096,
                system=self.system_prompt,
                tools=self.tools if self.tools else [],
                messages=self._conversation,
            )
            self._conversation.append({"role": "assistant", "content": response.content})

        for block in response.content:
            if hasattr(block, "text"):
                return block.text
        return ""

    async def _execute_tool(self, tool_name: str, tool_input: dict) -> Any:
        method = getattr(self, f"_tool_{tool_name}", None)
        if method is None:
            return f"Error: tool '{tool_name}' not found on {self.agent_id}"
        try:
            self._log("tool_call", {"tool": tool_name, "input": tool_input})
            result = await method(**tool_input) if asyncio.iscoroutinefunction(method) else method(**tool_input)
            return result
        except Exception as exc:
            self.log.error(f"Tool '{tool_name}' failed: {exc}")
            return f"Error: {exc}"

    def _log(self, action: str, detail: dict) -> None:
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "agent_id":  self.agent_id,
            "action":    action,
            "detail":    detail,
        }
        self.activity_log.append(entry)
        asyncio.get_event_loop().create_task(
            self.project_state.log_activity(self.agent_id, action, detail)
        )

    def get_status(self) -> dict:
        last = self.activity_log[-1] if self.activity_log else {}
        return {
            "agent_id":     self.agent_id,
            "role":         self.role,
            "status":       self.status.value,
            "last_action":  last.get("action", "—"),
            "message_count": len(self.activity_log),
        }
