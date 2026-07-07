import json
import os
from agentic_ai.agents.base_agent import BaseAgent
from agentic_ai.communication.message_schema import MessageType, Topic
from agentic_ai.tools.data_tools import run_eda
from agentic_ai.tools.model_tools import engineer_features, compute_clv_tool, compute_promo_sensitivity_tool
from src.config.settings import PATHS


class DataScientist1Agent(BaseAgent):
    def __init__(self, message_bus, project_state, anthropic_client):
        tools = [
            {
                "name": "run_eda",
                "description": "Run exploratory data analysis on a dataset",
                "input_schema": {
                    "type": "object",
                    "properties": {"dataset_path": {"type": "string"}},
                    "required": ["dataset_path"],
                },
            },
            {
                "name": "engineer_features",
                "description": "Build the full feature matrix from all datasets",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "customers_path":    {"type": "string"},
                        "transactions_path": {"type": "string"},
                        "products_path":     {"type": "string"},
                        "promos_path":       {"type": "string"},
                    },
                    "required": ["customers_path", "transactions_path", "products_path", "promos_path"],
                },
            },
            {
                "name": "compute_clv",
                "description": "Compute Customer Lifetime Value scores",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "customers_path":    {"type": "string"},
                        "transactions_path": {"type": "string"},
                    },
                    "required": ["customers_path", "transactions_path"],
                },
            },
            {
                "name": "compute_promo_sensitivity",
                "description": "Compute promo sensitivity scores",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "transactions_path": {"type": "string"},
                        "promos_path":       {"type": "string"},
                    },
                    "required": ["transactions_path", "promos_path"],
                },
            },
            {
                "name": "notify_ds_lead",
                "description": "Notify DS Lead when a task is complete",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "summary":      {"type": "string"},
                        "output_path":  {"type": "string"},
                        "request_type": {"type": "string", "enum": ["status_update", "approval_request"]},
                    },
                    "required": ["summary", "request_type"],
                },
            },
        ]
        super().__init__(
            agent_id="data_scientist_1",
            role="Data Scientist 1 — EDA & Feature Engineering",
            topics=[Topic.DATA_SCIENCE, Topic.PIPELINE],
            message_bus=message_bus,
            project_state=project_state,
            anthropic_client=anthropic_client,
            tools=tools,
        )
        self.system_prompt = (
            "You are Data Scientist 1, specializing in EDA and feature engineering. "
            "Phase 2: Run EDA on transactions and customers datasets, then notify DS Lead for approval. "
            "Phase 3: Compute CLV, promo sensitivity, and build the full feature matrix, then notify DS Lead. "
            "Always pass the correct file paths from the data/processed/ directory."
        )

    async def _handle_task(self, message):
        processed = PATHS["processed_data"]
        payload_str = json.dumps(message.payload)
        prompt = (
            f"Message from {message.from_agent} [{message.message_type}]:\n{payload_str}\n"
            f"Processed data directory: {processed}\n"
            "Execute your data science tasks using the tools, then notify DS Lead."
        )
        await self._run_tool_loop(prompt)

    async def _tool_run_eda(self, dataset_path: str) -> str:
        return run_eda(dataset_path)

    async def _tool_engineer_features(
        self, customers_path: str, transactions_path: str, products_path: str, promos_path: str
    ) -> str:
        result = engineer_features(customers_path, transactions_path, products_path, promos_path)
        result_dict = json.loads(result)
        if result_dict.get("status") == "ok":
            await self.project_state.register_dataset("feature_matrix", result_dict["path"])
        return result

    async def _tool_compute_clv(self, customers_path: str, transactions_path: str) -> str:
        return compute_clv_tool(customers_path, transactions_path)

    async def _tool_compute_promo_sensitivity(self, transactions_path: str, promos_path: str) -> str:
        return compute_promo_sensitivity_tool(transactions_path, promos_path)

    async def _tool_notify_ds_lead(self, summary: str, request_type: str, output_path: str = "") -> str:
        msg_type = MessageType.APPROVAL_REQUEST if request_type == "approval_request" else MessageType.STATUS_UPDATE
        await self.send_message(
            to="ds_lead",
            topic=Topic.DATA_SCIENCE,
            message_type=msg_type,
            payload={"summary": summary, "output_path": output_path, "agent": self.agent_id},
        )
        return f"DS Lead notified: {summary}"
