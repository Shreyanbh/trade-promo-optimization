import json
import os
from agentic_ai.agents.base_agent import BaseAgent
from agentic_ai.communication.message_schema import MessageType, Topic
from agentic_ai.tools.data_tools import run_data_quality_checks, optimize_dataset_storage
from src.config.settings import PATHS


class DataEngineer2Agent(BaseAgent):
    def __init__(self, message_bus, project_state, anthropic_client):
        tools = [
            {
                "name": "run_data_quality_checks",
                "description": "Run quality checks (nulls, duplicates) on a dataset",
                "input_schema": {
                    "type": "object",
                    "properties": {"dataset_path": {"type": "string"}},
                    "required": ["dataset_path"],
                },
            },
            {
                "name": "optimize_dataset_storage",
                "description": "Convert CSV dataset to Parquet for storage efficiency",
                "input_schema": {
                    "type": "object",
                    "properties": {"dataset_path": {"type": "string"}},
                    "required": ["dataset_path"],
                },
            },
            {
                "name": "archive_raw_data",
                "description": "Archive raw data files to data/raw/",
                "input_schema": {
                    "type": "object",
                    "properties": {"source_path": {"type": "string"}},
                    "required": ["source_path"],
                },
            },
            {
                "name": "notify_de_lead",
                "description": "Send completion status to DE Lead",
                "input_schema": {
                    "type": "object",
                    "properties": {"summary": {"type": "string"}},
                    "required": ["summary"],
                },
            },
        ]
        super().__init__(
            agent_id="data_engineer_2",
            role="Data Engineer 2",
            topics=[Topic.DATA_ENGINEERING, Topic.PIPELINE],
            message_bus=message_bus,
            project_state=project_state,
            anthropic_client=anthropic_client,
            tools=tools,
        )
        self.system_prompt = (
            "You are Data Engineer 2. You specialize in data quality, archiving, and storage optimization. "
            "When assigned Phase 1 tasks:\n"
            "1. Run quality checks on all processed datasets\n"
            "2. Optimize storage (Parquet conversion) where applicable\n"
            "3. Archive raw data copies\n"
            "4. Notify DE Lead when complete."
        )

    async def _handle_task(self, message):
        payload_str = json.dumps(message.payload)
        processed_dir = PATHS["processed_data"]
        prompt = (
            f"Message from {message.from_agent} [{message.message_type}]:\n{payload_str}\n"
            f"Processed data directory: {processed_dir}\n"
            "Run quality checks on available processed datasets, optimize storage, then notify DE Lead."
        )
        await self._run_tool_loop(prompt)

    async def _tool_run_data_quality_checks(self, dataset_path: str) -> str:
        return run_data_quality_checks(dataset_path)

    async def _tool_optimize_dataset_storage(self, dataset_path: str) -> str:
        return optimize_dataset_storage(dataset_path)

    async def _tool_archive_raw_data(self, source_path: str) -> str:
        import shutil
        os.makedirs(PATHS["raw_data"], exist_ok=True)
        if os.path.exists(source_path):
            dest = os.path.join(PATHS["raw_data"], os.path.basename(source_path))
            shutil.copy2(source_path, dest)
            return json.dumps({"status": "ok", "archived": dest})
        return json.dumps({"status": "skipped", "reason": "source not found"})

    async def _tool_notify_de_lead(self, summary: str) -> str:
        await self.send_message(
            to="de_lead",
            topic=Topic.DATA_ENGINEERING,
            message_type=MessageType.STATUS_UPDATE,
            payload={"summary": summary, "agent": self.agent_id},
        )
        return f"DE Lead notified: {summary}"
