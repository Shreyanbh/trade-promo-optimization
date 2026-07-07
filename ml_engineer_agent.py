import json
import os
from agentic_ai.agents.base_agent import BaseAgent
from agentic_ai.communication.message_schema import MessageType, Topic
from src.config.settings import PATHS


class MLEngineerAgent(BaseAgent):
    def __init__(self, message_bus, project_state, anthropic_client):
        tools = [
            {
                "name": "wrap_model_api",
                "description": "Create a REST API spec JSON for a trained model and save it to outputs/models/",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "model_path": {"type": "string"},
                        "model_type": {"type": "string"},
                    },
                    "required": ["model_path", "model_type"],
                },
            },
            {
                "name": "setup_model_monitoring",
                "description": "Save a monitoring configuration JSON for the model with drift and retrain thresholds",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "model_type": {"type": "string"},
                    },
                    "required": ["model_type"],
                },
            },
            {
                "name": "version_model",
                "description": "Log a model version entry to the model versions registry",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "model_type": {"type": "string"},
                        "version":    {"type": "string"},
                    },
                    "required": ["model_type", "version"],
                },
            },
            {
                "name": "notify_pm",
                "description": "Send a STATUS_UPDATE to the project manager",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "summary": {"type": "string"},
                    },
                    "required": ["summary"],
                },
            },
        ]
        super().__init__(
            agent_id="ml_engineer",
            role="ML Engineer — Model Deployment",
            topics=[Topic.PIPELINE, Topic.MANAGEMENT],
            message_bus=message_bus,
            project_state=project_state,
            anthropic_client=anthropic_client,
            tools=tools,
        )
        self.system_prompt = (
            "You are the ML Engineer responsible for model deployment and serving infrastructure. "
            "After models are exported, you wrap them in REST API specs, set up monitoring configs, "
            "and log their versions to the model registry. "
            "You ensure models are production-ready before notifying the Project Manager. "
            "Always wrap the API, set up monitoring, version the model, then notify PM."
        )

    async def _handle_task(self, message):
        models_dir = PATHS["models"]
        payload_str = json.dumps(message.payload)
        prompt = (
            f"Message from {message.from_agent} [{message.message_type}]:\n{payload_str}\n"
            f"Models directory: {models_dir}\n"
            "Wrap the model API, set up monitoring, version the model, then notify Project Manager."
        )
        await self._run_tool_loop(prompt)

    def _tool_wrap_model_api(self, model_path: str, model_type: str) -> str:
        os.makedirs(PATHS["models"], exist_ok=True)
        api_spec = {
            "api_version": "v1",
            "model_type": model_type,
            "model_path": model_path,
            "endpoint": f"/api/v1/predict/{model_type}",
            "method": "POST",
            "request_schema": {
                "type": "object",
                "properties": {
                    "customer_id": {"type": "string", "description": "Unique customer identifier"},
                    "features":    {"type": "object", "description": "Feature vector for prediction"},
                },
                "required": ["customer_id"],
            },
            "response_schema": {
                "type": "object",
                "properties": {
                    "customer_id":  {"type": "string"},
                    "prediction":   {"type": "object"},
                    "model_type":   {"type": "string"},
                    "model_version": {"type": "string"},
                },
            },
            "auth": "Bearer token required",
            "rate_limit": "1000 req/min",
            "timeout_ms": 500,
        }
        spec_path = os.path.join(PATHS["models"], f"{model_type}_api_spec.json")
        with open(spec_path, "w") as f:
            json.dump(api_spec, f, indent=2)
        return json.dumps({
            "status": "ok",
            "model_type": model_type,
            "api_spec_path": spec_path,
            "endpoint": api_spec["endpoint"],
        })

    def _tool_setup_model_monitoring(self, model_type: str) -> str:
        os.makedirs(PATHS["models"], exist_ok=True)
        monitoring_config = {
            "model_type": model_type,
            "drift_threshold": 0.10,
            "retrain_trigger": "drift_score > drift_threshold OR accuracy_drop > 0.05",
            "check_interval_days": 7,
            "metrics_to_monitor": [
                "prediction_distribution",
                "feature_drift",
                "request_latency_p99_ms",
                "error_rate",
            ],
            "alerting": {
                "email": "ml-team@company.com",
                "slack_channel": "#ml-alerts",
                "severity_threshold": "WARNING",
            },
            "data_retention_days": 90,
        }
        config_path = os.path.join(PATHS["models"], f"{model_type}_monitoring.json")
        with open(config_path, "w") as f:
            json.dump(monitoring_config, f, indent=2)
        return json.dumps({
            "status": "ok",
            "model_type": model_type,
            "monitoring_config_path": config_path,
            "drift_threshold": monitoring_config["drift_threshold"],
            "check_interval_days": monitoring_config["check_interval_days"],
        })

    def _tool_version_model(self, model_type: str, version: str) -> str:
        from datetime import datetime, timezone
        os.makedirs(PATHS["models"], exist_ok=True)
        versions_path = os.path.join(PATHS["models"], "model_versions.json")

        if os.path.exists(versions_path):
            with open(versions_path, "r") as f:
                versions_registry = json.load(f)
        else:
            versions_registry = {}

        if model_type not in versions_registry:
            versions_registry[model_type] = []

        version_entry = {
            "version": version,
            "registered_at": datetime.now(timezone.utc).isoformat(),
            "model_path": os.path.join(PATHS["models"], f"{model_type}_model.pkl"),
            "api_spec_path": os.path.join(PATHS["models"], f"{model_type}_api_spec.json"),
            "status": "active",
        }
        versions_registry[model_type].append(version_entry)

        with open(versions_path, "w") as f:
            json.dump(versions_registry, f, indent=2)

        return json.dumps({
            "status": "ok",
            "model_type": model_type,
            "version": version,
            "versions_registry_path": versions_path,
            "total_versions": len(versions_registry[model_type]),
        })

    async def _tool_notify_pm(self, summary: str) -> str:
        await self.send_message(
            to="project_manager",
            topic=Topic.MANAGEMENT,
            message_type=MessageType.STATUS_UPDATE,
            payload={"summary": summary, "agent": self.agent_id},
        )
        return json.dumps({
            "status": "ok",
            "notified": "project_manager",
            "summary": summary,
        })
