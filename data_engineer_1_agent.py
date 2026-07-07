import json
from agentic_ai.agents.base_agent import BaseAgent
from agentic_ai.communication.message_schema import MessageType, Topic
from agentic_ai.tools.data_tools import ingest_data, validate_schema, run_etl_pipeline
from src.config.settings import PATHS
import os


class DataEngineer1Agent(BaseAgent):
    def __init__(self, message_bus, project_state, anthropic_client):
        tools = [
            {
                "name": "ingest_data",
                "description": "Ingest raw data and generate synthetic dataset if no source provided",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "source_path":    {"type": "string"},
                        "dataset_name":   {"type": "string"},
                    },
                },
            },
            {
                "name": "validate_schema",
                "description": "Validate a dataset against a named schema",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "dataset_path": {"type": "string"},
                        "schema_name":  {"type": "string"},
                    },
                    "required": ["dataset_path", "schema_name"],
                },
            },
            {
                "name": "run_etl_pipeline",
                "description": "Run ETL cleaning pipeline on a dataset",
                "input_schema": {
                    "type": "object",
                    "properties": {"dataset_name": {"type": "string"}},
                    "required": ["dataset_name"],
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
            agent_id="data_engineer_1",
            role="Data Engineer 1",
            topics=[Topic.DATA_ENGINEERING, Topic.PIPELINE],
            message_bus=message_bus,
            project_state=project_state,
            anthropic_client=anthropic_client,
            tools=tools,
        )
        self.system_prompt = (
            "You are Data Engineer 1. Your responsibilities are data ingestion and ETL pipelines. "
            "When assigned Phase 1 tasks:\n"
            "1. Call ingest_data (generates synthetic data if no real source)\n"
            "2. Call validate_schema for each dataset\n"
            "3. Call run_etl_pipeline to clean each dataset\n"
            "4. Register datasets in project state\n"
            "5. Notify DE Lead when complete."
        )

    async def _handle_task(self, message):
        payload_str = json.dumps(message.payload)
        prompt = (
            f"Message from {message.from_agent} [{message.message_type}]:\n{payload_str}\n"
            "Execute your data engineering tasks using the available tools, then notify DE Lead."
        )
        await self._run_tool_loop(prompt)

    async def _tool_ingest_data(self, source_path: str = None, dataset_name: str = "synthetic") -> str:
        result = ingest_data(source_path, dataset_name)
        result_dict = json.loads(result)
        if "datasets" in result_dict:
            for name, path in result_dict["datasets"].items():
                await self.project_state.register_dataset(name, path)
        return result

    async def _tool_validate_schema(self, dataset_path: str, schema_name: str) -> str:
        return validate_schema(dataset_path, schema_name)

    async def _tool_run_etl_pipeline(self, dataset_name: str) -> str:
        return run_etl_pipeline(dataset_name)

    async def _tool_notify_de_lead(self, summary: str) -> str:
        await self.send_message(
            to="de_lead",
            topic=Topic.DATA_ENGINEERING,
            message_type=MessageType.STATUS_UPDATE,
            payload={"summary": summary, "agent": self.agent_id},
        )
        return f"DE Lead notified: {summary}"
