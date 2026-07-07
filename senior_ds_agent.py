import json
import os
from agentic_ai.agents.base_agent import BaseAgent
from agentic_ai.communication.message_schema import MessageType, Topic
from src.config.settings import PATHS


class SeniorDataScientistAgent(BaseAgent):
    def __init__(self, message_bus, project_state, anthropic_client):
        tools = [
            {
                "name": "validate_model_quality",
                "description": "Validate model quality metrics — checks silhouette score and returns approval decision with recommendations",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "model_type": {"type": "string"},
                        "metrics":    {"type": "object"},
                    },
                    "required": ["model_type", "metrics"],
                },
            },
            {
                "name": "check_segment_bias",
                "description": "Check if any cluster is under-represented (<5% of population) indicating potential bias",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "cluster_profiles_summary": {"type": "string"},
                    },
                    "required": ["cluster_profiles_summary"],
                },
            },
            {
                "name": "run_cross_validation",
                "description": "Run simulated cross-validation on the dataset for the given model type",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "dataset_path": {"type": "string"},
                        "model_type":   {"type": "string"},
                    },
                    "required": ["dataset_path", "model_type"],
                },
            },
            {
                "name": "notify_ds_lead",
                "description": "Send approval response to DS Lead after validation is complete",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "summary":  {"type": "string"},
                        "approved": {"type": "boolean"},
                    },
                    "required": ["summary", "approved"],
                },
            },
        ]
        super().__init__(
            agent_id="senior_data_scientist",
            role="Senior Data Scientist — Model Validation",
            topics=[Topic.DATA_SCIENCE, Topic.MANAGEMENT],
            message_bus=message_bus,
            project_state=project_state,
            anthropic_client=anthropic_client,
            tools=tools,
        )
        self.system_prompt = (
            "You are the Senior Data Scientist responsible for model validation. "
            "Your job is to validate model quality, check for bias and overfitting, "
            "and assess statistical soundness. You review models produced by DS2 before "
            "the DS Lead grants final approval. You ensure robustness across all customer segments. "
            "Always run validate_model_quality and check_segment_bias before notifying DS Lead. "
            "If silhouette score is below 0.1 or any segment is under-represented, reject and explain."
        )

    async def _handle_task(self, message):
        processed = PATHS["processed_data"]
        payload_str = json.dumps(message.payload)
        prompt = (
            f"Message from {message.from_agent} [{message.message_type}]:\n{payload_str}\n"
            f"Processed data directory: {processed}\n"
            "Validate the model quality, check for segment bias, optionally run cross-validation, "
            "then notify DS Lead with your approval decision."
        )
        await self._run_tool_loop(prompt)

    def _tool_validate_model_quality(self, model_type: str, metrics: dict) -> str:
        recommendations = []
        approved = True

        silhouette = metrics.get("silhouette_score", metrics.get("silhouette", None))
        if silhouette is not None:
            if silhouette < 0.1:
                approved = False
                recommendations.append(
                    f"Silhouette score {silhouette:.4f} is below the minimum threshold of 0.1. "
                    "Consider adjusting n_clusters or switching algorithm."
                )
            elif silhouette < 0.3:
                recommendations.append(
                    f"Silhouette score {silhouette:.4f} is acceptable but could be improved. "
                    "Try tuning hyperparameters."
                )
        else:
            recommendations.append("No silhouette score provided — cannot fully validate clustering quality.")

        inertia = metrics.get("inertia", None)
        if inertia is not None and inertia > 1_000_000:
            recommendations.append(
                "High inertia detected. Consider normalising features before clustering."
            )

        precision = metrics.get("precision_at_k", metrics.get("precision", None))
        if precision is not None and precision < 0.05:
            approved = False
            recommendations.append(
                f"Recommender precision@k {precision:.4f} is very low. "
                "Check collaborative filtering data sparsity."
            )

        return json.dumps({
            "status": "ok",
            "model_type": model_type,
            "approved": approved,
            "metrics_received": metrics,
            "recommendations": recommendations,
            "verdict": "APPROVED" if approved else "REJECTED",
        })

    def _tool_check_segment_bias(self, cluster_profiles_summary: str) -> str:
        lines = cluster_profiles_summary.strip().splitlines()
        biased_segments = []
        parsed_segments = []

        for line in lines:
            # Expect lines like "Cluster 0: size=1200, pct=24.5"
            # or any format containing a percentage number
            parts = line.split()
            pct_value = None
            segment_name = parts[0] if parts else "unknown"

            for part in parts:
                part_clean = part.replace("pct=", "").replace("%", "").rstrip(",")
                try:
                    pct_value = float(part_clean)
                    if 0 < pct_value <= 100:
                        break
                except ValueError:
                    continue

            if pct_value is not None:
                parsed_segments.append({"segment": segment_name, "pct": pct_value})
                if pct_value < 5.0:
                    biased_segments.append({
                        "segment": segment_name,
                        "pct": pct_value,
                        "issue": "Cluster represents less than 5% of population — may be too small for reliable insights.",
                    })

        bias_detected = len(biased_segments) > 0
        return json.dumps({
            "status": "ok",
            "bias_detected": bias_detected,
            "segments_parsed": parsed_segments,
            "biased_segments": biased_segments,
            "recommendation": (
                "Consider merging small clusters or adjusting min_samples (DBSCAN) / n_clusters (KMeans)."
                if bias_detected else "All segments are adequately sized."
            ),
        })

    def _tool_run_cross_validation(self, dataset_path: str, model_type: str) -> str:
        import random
        # Simulated CV — in production this would load the dataset and run k-fold CV
        random.seed(abs(hash(dataset_path + model_type)) % (2 ** 31))
        cv_scores = [round(random.uniform(0.30, 0.65), 4) for _ in range(5)]
        mean_cv = round(sum(cv_scores) / len(cv_scores), 4)
        std_cv = round((sum((s - mean_cv) ** 2 for s in cv_scores) / len(cv_scores)) ** 0.5, 4)
        return json.dumps({
            "status": "ok",
            "dataset_path": dataset_path,
            "model_type": model_type,
            "cv_folds": 5,
            "cv_scores": cv_scores,
            "cv_score_mean": mean_cv,
            "cv_score_std": std_cv,
            "interpretation": (
                "Good CV stability." if std_cv < 0.05
                else "High variance across folds — possible overfitting or small dataset."
            ),
        })

    async def _tool_notify_ds_lead(self, summary: str, approved: bool) -> str:
        await self.send_message(
            to="ds_lead",
            topic=Topic.DATA_SCIENCE,
            message_type=MessageType.APPROVAL_RESPONSE,
            payload={
                "summary": summary,
                "approved": approved,
                "agent": self.agent_id,
            },
        )
        return json.dumps({
            "status": "ok",
            "notified": "ds_lead",
            "approved": approved,
            "summary": summary,
        })
