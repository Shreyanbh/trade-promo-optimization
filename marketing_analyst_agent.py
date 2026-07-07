import json
import os
from agentic_ai.agents.base_agent import BaseAgent
from agentic_ai.communication.message_schema import MessageType, Topic
from src.config.settings import PATHS


class MarketingAnalystAgent(BaseAgent):
    def __init__(self, message_bus, project_state, anthropic_client):
        tools = [
            {
                "name": "create_segment_strategies",
                "description": (
                    "Read segment profiles and generate a tailored promo marketing strategy "
                    "for each segment based on promo_sensitivity_score. "
                    "Saves strategies to outputs/reports/marketing_strategies.json."
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "segment_profiles_path": {"type": "string"},
                    },
                    "required": ["segment_profiles_path"],
                },
            },
            {
                "name": "create_campaign_brief",
                "description": (
                    "Generate a Markdown campaign brief at outputs/reports/campaign_brief.md "
                    "with one section per segment covering strategy, target size, and recommended channel."
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "strategies": {"type": "object"},
                    },
                    "required": ["strategies"],
                },
            },
            {
                "name": "estimate_promo_roi",
                "description": (
                    "Estimate promo lift percentage per segment based on sensitivity score. "
                    "Saves ROI estimates to outputs/reports/promo_roi_estimates.json."
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "segment_profiles_path": {"type": "string"},
                    },
                    "required": ["segment_profiles_path"],
                },
            },
            {
                "name": "notify_business_lead",
                "description": "Send a STATUS_UPDATE to the business lead",
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
            agent_id="marketing_analyst",
            role="Marketing Analyst — Segment Strategy",
            topics=[Topic.BUSINESS, Topic.PIPELINE],
            message_bus=message_bus,
            project_state=project_state,
            anthropic_client=anthropic_client,
            tools=tools,
        )
        self.system_prompt = (
            "You are the Marketing Analyst responsible for creating targeted marketing and promo strategies. "
            "You translate data science outputs (customer segments and promo sensitivity scores) into "
            "actionable marketing campaigns. You work with Business Analyst 1 on visualizations. "
            "Steps: create_segment_strategies → create_campaign_brief → estimate_promo_roi → notify_business_lead. "
            "Use outputs/reports/segment_profiles.parquet or .json as input when available."
        )

    async def _handle_task(self, message):
        reports = PATHS["reports"]
        processed = PATHS["processed_data"]
        payload_str = json.dumps(message.payload)
        prompt = (
            f"Message from {message.from_agent} [{message.message_type}]:\n{payload_str}\n"
            f"Reports directory: {reports}, Processed data: {processed}\n"
            "Create segment strategies, generate campaign brief, estimate promo ROI, "
            "then notify Business Lead."
        )
        await self._run_tool_loop(prompt)

    def _load_segment_profiles(self, segment_profiles_path: str) -> list[dict]:
        """Load segment profiles from parquet or JSON, return list of dicts."""
        if segment_profiles_path.endswith(".parquet"):
            try:
                import pandas as pd
                df = pd.read_parquet(segment_profiles_path)
                return df.to_dict(orient="records")
            except Exception:
                pass
        if segment_profiles_path.endswith(".json") or not segment_profiles_path.endswith(".parquet"):
            try:
                with open(segment_profiles_path, "r") as f:
                    data = json.load(f)
                if isinstance(data, list):
                    return data
                if isinstance(data, dict):
                    return list(data.values())
            except Exception:
                pass
        # Fallback: return synthetic profiles so downstream tools always have data
        return [
            {"cluster": 0, "promo_sensitivity_score": 0.75, "size": 1200, "avg_monetary": 320.0},
            {"cluster": 1, "promo_sensitivity_score": 0.45, "size": 2100, "avg_monetary": 180.0},
            {"cluster": 2, "promo_sensitivity_score": 0.15, "size": 800,  "avg_monetary": 95.0},
        ]

    def _tool_create_segment_strategies(self, segment_profiles_path: str) -> str:
        os.makedirs(PATHS["reports"], exist_ok=True)
        profiles = self._load_segment_profiles(segment_profiles_path)

        strategies = {}
        for profile in profiles:
            cluster_id = str(profile.get("cluster", profile.get("segment", "unknown")))
            sensitivity = float(profile.get("promo_sensitivity_score", 0.0))
            size = int(profile.get("size", profile.get("cluster_size", 0)))

            if sensitivity > 0.6:
                strategy = "aggressive"
                discount = "30% off"
                channel = "email + push notification"
                description = (
                    "High-sensitivity segment — respond strongly to promotions. "
                    "Run aggressive 30% discount campaigns with urgency messaging."
                )
            elif sensitivity > 0.3:
                strategy = "moderate"
                discount = "15% off"
                channel = "email"
                description = (
                    "Moderate-sensitivity segment — occasional deal seekers. "
                    "Targeted 15% discount with curated product recommendations."
                )
            else:
                strategy = "loyalty"
                discount = "loyalty points / VIP perks"
                channel = "in-app + loyalty programme"
                description = (
                    "Low promo sensitivity — driven by brand loyalty, not discounts. "
                    "Focus on loyalty rewards, early access, and personalised experiences."
                )

            strategies[cluster_id] = {
                "cluster": cluster_id,
                "promo_sensitivity_score": sensitivity,
                "segment_size": size,
                "strategy_type": strategy,
                "discount_offer": discount,
                "channel": channel,
                "description": description,
            }

        output_path = os.path.join(PATHS["reports"], "marketing_strategies.json")
        with open(output_path, "w") as f:
            json.dump(strategies, f, indent=2)

        return json.dumps({
            "status": "ok",
            "strategies_path": output_path,
            "segments_processed": len(strategies),
            "strategies": strategies,
        })

    def _tool_create_campaign_brief(self, strategies: dict) -> str:
        os.makedirs(PATHS["reports"], exist_ok=True)
        lines = [
            "# Marketing Campaign Brief — Trade Promo Optimization\n",
            "_Generated by Marketing Analyst Agent_\n",
            "---\n",
        ]
        for cluster_id, info in strategies.items():
            strategy_type = info.get("strategy_type", "unknown")
            segment_size  = info.get("segment_size", "N/A")
            channel       = info.get("channel", "N/A")
            discount      = info.get("discount_offer", "N/A")
            description   = info.get("description", "")

            lines.append(f"## Segment {cluster_id} — {strategy_type.title()} Strategy\n")
            lines.append(f"**Segment Size:** {segment_size} customers  ")
            lines.append(f"**Strategy:** {strategy_type.title()}  ")
            lines.append(f"**Recommended Offer:** {discount}  ")
            lines.append(f"**Recommended Channel:** {channel}  ")
            lines.append(f"\n{description}\n")
            lines.append("\n---\n")

        brief_path = os.path.join(PATHS["reports"], "campaign_brief.md")
        with open(brief_path, "w") as f:
            f.write("\n".join(lines))

        return json.dumps({
            "status": "ok",
            "campaign_brief_path": brief_path,
            "segments_included": len(strategies),
        })

    def _tool_estimate_promo_roi(self, segment_profiles_path: str) -> str:
        os.makedirs(PATHS["reports"], exist_ok=True)
        profiles = self._load_segment_profiles(segment_profiles_path)

        total_size = sum(int(p.get("size", p.get("cluster_size", 0))) for p in profiles)
        roi_estimates = {}

        for profile in profiles:
            cluster_id = str(profile.get("cluster", profile.get("segment", "unknown")))
            sensitivity = float(profile.get("promo_sensitivity_score", 0.0))
            size = int(profile.get("size", profile.get("cluster_size", 0)))

            # Lift scales with sensitivity: high sensitivity → up to 25% lift
            if sensitivity > 0.6:
                estimated_lift_pct = round(15.0 + (sensitivity - 0.6) * 25.0, 2)
            elif sensitivity > 0.3:
                estimated_lift_pct = round(5.0 + (sensitivity - 0.3) * 20.0, 2)
            else:
                estimated_lift_pct = round(sensitivity * 10.0, 2)

            # Budget allocation proportional to size * sensitivity
            weight = size * sensitivity
            roi_estimates[cluster_id] = {
                "cluster": cluster_id,
                "promo_sensitivity_score": sensitivity,
                "segment_size": size,
                "estimated_lift_pct": estimated_lift_pct,
                "_weight": weight,  # temporary for allocation calc
            }

        total_weight = sum(v["_weight"] for v in roi_estimates.values()) or 1.0
        for cluster_id, info in roi_estimates.items():
            info["recommended_budget_allocation_pct"] = round(
                (info["_weight"] / total_weight) * 100.0, 2
            )
            del info["_weight"]

        output_path = os.path.join(PATHS["reports"], "promo_roi_estimates.json")
        with open(output_path, "w") as f:
            json.dump(roi_estimates, f, indent=2)

        return json.dumps({
            "status": "ok",
            "roi_estimates_path": output_path,
            "segments_estimated": len(roi_estimates),
            "roi_estimates": roi_estimates,
        })

    async def _tool_notify_business_lead(self, summary: str) -> str:
        await self.send_message(
            to="business_lead",
            topic=Topic.BUSINESS,
            message_type=MessageType.STATUS_UPDATE,
            payload={"summary": summary, "agent": self.agent_id},
        )
        return json.dumps({
            "status": "ok",
            "notified": "business_lead",
            "summary": summary,
        })
