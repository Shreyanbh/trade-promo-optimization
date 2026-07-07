import json
import os
from agentic_ai.agents.base_agent import BaseAgent
from agentic_ai.communication.message_schema import MessageType, Topic
from agentic_ai.tools.model_tools import (
    train_segmentation_model, train_recommender_model,
    evaluate_model_tool, export_model_tool,
)
from src.config.settings import PATHS


class DataScientist2Agent(BaseAgent):
    def __init__(self, message_bus, project_state, anthropic_client):
        tools = [
            {
                "name": "train_segmentation_model",
                "description": "Train a customer segmentation model (kmeans or dbscan)",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "feature_matrix_path": {"type": "string"},
                        "algorithm":           {"type": "string", "enum": ["kmeans", "dbscan"]},
                    },
                    "required": ["feature_matrix_path"],
                },
            },
            {
                "name": "train_recommender_model",
                "description": "Train the collaborative filtering recommender model",
                "input_schema": {
                    "type": "object",
                    "properties": {"transactions_path": {"type": "string"}},
                    "required": ["transactions_path"],
                },
            },
            {
                "name": "evaluate_model",
                "description": "Evaluate a trained model's performance",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "model_type":          {"type": "string", "enum": ["segmentation", "recommender"]},
                        "feature_matrix_path": {"type": "string"},
                        "transactions_path":   {"type": "string"},
                        "k":                   {"type": "integer"},
                    },
                    "required": ["model_type"],
                },
            },
            {
                "name": "export_model",
                "description": "Serialize and save a trained model",
                "input_schema": {
                    "type": "object",
                    "properties": {"model_type": {"type": "string"}},
                    "required": ["model_type"],
                },
            },
            {
                "name": "notify_ds_lead",
                "description": "Send metrics/completion to DS Lead for approval",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "summary":     {"type": "string"},
                        "metrics":     {"type": "object"},
                        "model_name":  {"type": "string"},
                        "request_type":{"type": "string"},
                    },
                    "required": ["summary"],
                },
            },
        ]
        super().__init__(
            agent_id="data_scientist_2",
            role="Data Scientist 2 — Modelling & Evaluation",
            topics=[Topic.DATA_SCIENCE, Topic.PIPELINE],
            message_bus=message_bus,
            project_state=project_state,
            anthropic_client=anthropic_client,
            tools=tools,
        )
        self.system_prompt = (
            "You are Data Scientist 2, specializing in model development and evaluation. "
            "Phase 4: Train segmentation (KMeans), evaluate it, train recommender (ALS/NMF), evaluate it, "
            "then request DS Lead approval with metrics. "
            "Phase 5: Export approved models to outputs/models/. "
            "Use the feature_matrix.parquet and transactions.parquet from data/processed/."
        )

    async def _handle_task(self, message):
        processed = PATHS["processed_data"]
        payload_str = json.dumps(message.payload)
        prompt = (
            f"Message from {message.from_agent} [{message.message_type}]:\n{payload_str}\n"
            f"Processed data: {processed}\n"
            "Execute your modelling tasks using the tools, then notify DS Lead."
        )
        await self._run_tool_loop(prompt)

    async def _tool_train_segmentation_model(self, feature_matrix_path: str, algorithm: str = "kmeans") -> str:
        return train_segmentation_model(feature_matrix_path, algorithm)

    async def _tool_train_recommender_model(self, transactions_path: str) -> str:
        return train_recommender_model(transactions_path)

    async def _tool_evaluate_model(
        self,
        model_type: str,
        feature_matrix_path: str = None,
        transactions_path: str = None,
        k: int = 10,
    ) -> str:
        return evaluate_model_tool(model_type, feature_matrix_path, transactions_path, k)

    async def _tool_export_model(self, model_type: str) -> str:
        result = export_model_tool(model_type)
        result_dict = json.loads(result)
        if result_dict.get("status") == "ok":
            await self.project_state.register_model(model_type, {"path": result_dict["path"]})
        return result

    async def _tool_notify_ds_lead(
        self, summary: str, metrics: dict = None, model_name: str = "", request_type: str = "status_update"
    ) -> str:
        msg_type = MessageType.APPROVAL_REQUEST if request_type == "approval_request" else MessageType.STATUS_UPDATE
        await self.send_message(
            to="ds_lead",
            topic=Topic.DATA_SCIENCE,
            message_type=msg_type,
            payload={"summary": summary, "metrics": metrics or {}, "model_name": model_name},
        )
        return f"DS Lead notified: {summary}"
