import json
import os
from agentic_ai.agents.base_agent import BaseAgent
from agentic_ai.communication.message_schema import MessageType, Topic
from src.config.settings import PATHS


class FinanceAnalystAgent(BaseAgent):
    def __init__(self, message_bus, project_state, anthropic_client):
        tools = [
            {
                "name": "calculate_revenue_impact",
                "description": (
                    "Read segment profiles and compute total revenue per segment "
                    "(size * avg_monetary, or size * 150 as proxy). "
                    "Saves results to outputs/reports/revenue_impact.json."
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
                "name": "calculate_roi",
                "description": (
                    "Read promo_roi_estimates.json, compute net_benefit and overall ROI%. "
                    "Assumes promo cost = 5% of revenue. "
                    "Saves to outputs/reports/financial_summary.json."
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "promo_roi_path": {"type": "string"},
                    },
                    "required": ["promo_roi_path"],
                },
            },
            {
                "name": "generate_financial_report",
                "description": (
                    "Generate outputs/reports/financial_report.md with an executive summary table "
                    "combining revenue impact and ROI data."
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "revenue_path": {"type": "string"},
                        "roi_path":     {"type": "string"},
                    },
                    "required": ["revenue_path", "roi_path"],
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
            agent_id="finance_analyst",
            role="Finance Analyst — ROI & Revenue Impact",
            topics=[Topic.BUSINESS, Topic.MANAGEMENT],
            message_bus=message_bus,
            project_state=project_state,
            anthropic_client=anthropic_client,
            tools=tools,
        )
        self.system_prompt = (
            "You are the Finance Analyst responsible for computing financial impact of promo strategies. "
            "You calculate ROI, budget requirements, and revenue uplift projections, then present "
            "a clear financial business case to stakeholders. "
            "Steps: calculate_revenue_impact → calculate_roi → generate_financial_report → notify_business_lead. "
            "Use outputs/reports/segment_profiles.json (or .parquet) and "
            "outputs/reports/promo_roi_estimates.json as inputs."
        )

    async def _handle_task(self, message):
        reports = PATHS["reports"]
        payload_str = json.dumps(message.payload)
        prompt = (
            f"Message from {message.from_agent} [{message.message_type}]:\n{payload_str}\n"
            f"Reports directory: {reports}\n"
            "Calculate revenue impact, compute ROI, generate financial report, "
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
        try:
            with open(segment_profiles_path, "r") as f:
                data = json.load(f)
            if isinstance(data, list):
                return data
            if isinstance(data, dict):
                return list(data.values())
        except Exception:
            pass
        # Fallback synthetic profiles
        return [
            {"cluster": 0, "size": 1200, "avg_monetary": 320.0, "promo_sensitivity_score": 0.75},
            {"cluster": 1, "size": 2100, "avg_monetary": 180.0, "promo_sensitivity_score": 0.45},
            {"cluster": 2, "size": 800,  "avg_monetary": 95.0,  "promo_sensitivity_score": 0.15},
        ]

    def _tool_calculate_revenue_impact(self, segment_profiles_path: str) -> str:
        os.makedirs(PATHS["reports"], exist_ok=True)
        profiles = self._load_segment_profiles(segment_profiles_path)

        revenue_impact = {}
        grand_total = 0.0

        for profile in profiles:
            cluster_id = str(profile.get("cluster", profile.get("segment", "unknown")))
            size = int(profile.get("size", profile.get("cluster_size", 0)))
            avg_monetary = float(profile.get("avg_monetary", profile.get("avg_spend", 150.0)))
            if avg_monetary <= 0:
                avg_monetary = 150.0  # proxy when not available

            total_revenue = round(size * avg_monetary, 2)
            grand_total += total_revenue

            revenue_impact[cluster_id] = {
                "cluster": cluster_id,
                "segment_size": size,
                "avg_monetary": avg_monetary,
                "total_revenue": total_revenue,
            }

        # Add revenue share % after computing grand total
        for info in revenue_impact.values():
            info["revenue_share_pct"] = round(
                (info["total_revenue"] / grand_total * 100) if grand_total > 0 else 0.0, 2
            )

        output = {
            "segments": revenue_impact,
            "grand_total_revenue": round(grand_total, 2),
        }
        output_path = os.path.join(PATHS["reports"], "revenue_impact.json")
        with open(output_path, "w") as f:
            json.dump(output, f, indent=2)

        return json.dumps({
            "status": "ok",
            "revenue_impact_path": output_path,
            "grand_total_revenue": round(grand_total, 2),
            "segments_computed": len(revenue_impact),
        })

    def _tool_calculate_roi(self, promo_roi_path: str) -> str:
        os.makedirs(PATHS["reports"], exist_ok=True)

        try:
            with open(promo_roi_path, "r") as f:
                roi_estimates = json.load(f)
        except Exception:
            roi_estimates = {
                "0": {"estimated_lift_pct": 20.0, "segment_size": 1200, "recommended_budget_allocation_pct": 50.0},
                "1": {"estimated_lift_pct": 10.0, "segment_size": 2100, "recommended_budget_allocation_pct": 35.0},
                "2": {"estimated_lift_pct": 2.0,  "segment_size": 800,  "recommended_budget_allocation_pct": 15.0},
            }

        # Load revenue data if available to use actual revenue figures
        revenue_path = os.path.join(PATHS["reports"], "revenue_impact.json")
        revenue_by_segment = {}
        try:
            with open(revenue_path, "r") as f:
                rev_data = json.load(f)
            for seg_id, info in rev_data.get("segments", {}).items():
                revenue_by_segment[seg_id] = info.get("total_revenue", 0.0)
        except Exception:
            pass

        financial_summary = {}
        total_revenue = 0.0
        total_promo_cost = 0.0
        total_net_benefit = 0.0

        for cluster_id, info in roi_estimates.items():
            lift_pct = float(info.get("estimated_lift_pct", 0.0)) / 100.0
            size = int(info.get("segment_size", info.get("size", 0)))
            # Use actual revenue if available, else proxy $150 per customer
            base_revenue = revenue_by_segment.get(cluster_id, size * 150.0)
            incremental_revenue = round(base_revenue * lift_pct, 2)
            promo_cost = round(base_revenue * 0.05, 2)  # 5% of revenue as promo cost
            net_benefit = round(incremental_revenue - promo_cost, 2)

            total_revenue += base_revenue
            total_promo_cost += promo_cost
            total_net_benefit += net_benefit

            financial_summary[cluster_id] = {
                "cluster": cluster_id,
                "base_revenue": round(base_revenue, 2),
                "estimated_lift_pct": info.get("estimated_lift_pct", 0.0),
                "incremental_revenue": incremental_revenue,
                "promo_cost": promo_cost,
                "net_benefit": net_benefit,
            }

        overall_roi_pct = round(
            (total_net_benefit / total_promo_cost * 100) if total_promo_cost > 0 else 0.0, 2
        )

        summary_output = {
            "segments": financial_summary,
            "totals": {
                "total_base_revenue": round(total_revenue, 2),
                "total_promo_cost": round(total_promo_cost, 2),
                "total_net_benefit": round(total_net_benefit, 2),
                "overall_roi_pct": overall_roi_pct,
            },
        }
        output_path = os.path.join(PATHS["reports"], "financial_summary.json")
        with open(output_path, "w") as f:
            json.dump(summary_output, f, indent=2)

        return json.dumps({
            "status": "ok",
            "financial_summary_path": output_path,
            "overall_roi_pct": overall_roi_pct,
            "total_net_benefit": round(total_net_benefit, 2),
            "total_promo_cost": round(total_promo_cost, 2),
        })

    def _tool_generate_financial_report(self, revenue_path: str, roi_path: str) -> str:
        os.makedirs(PATHS["reports"], exist_ok=True)

        # Load revenue data
        try:
            with open(revenue_path, "r") as f:
                rev_data = json.load(f)
        except Exception:
            rev_data = {"segments": {}, "grand_total_revenue": 0.0}

        # Load ROI/financial summary data
        try:
            with open(roi_path, "r") as f:
                roi_data = json.load(f)
        except Exception:
            roi_data = {"segments": {}, "totals": {}}

        totals = roi_data.get("totals", {})
        grand_total_revenue = rev_data.get("grand_total_revenue", totals.get("total_base_revenue", 0.0))
        overall_roi_pct = totals.get("overall_roi_pct", 0.0)
        total_net_benefit = totals.get("total_net_benefit", 0.0)
        total_promo_cost = totals.get("total_promo_cost", 0.0)

        lines = [
            "# Financial Report — Trade Promo Optimization\n",
            "_Generated by Finance Analyst Agent_\n",
            "---\n",
            "## Executive Summary\n",
            f"| Metric | Value |",
            f"|--------|-------|",
            f"| Total Base Revenue | ${grand_total_revenue:,.2f} |",
            f"| Total Promo Cost (5% of revenue) | ${total_promo_cost:,.2f} |",
            f"| Total Net Benefit | ${total_net_benefit:,.2f} |",
            f"| **Overall ROI** | **{overall_roi_pct:.1f}%** |",
            "",
            "---\n",
            "## Segment-Level Breakdown\n",
            "| Segment | Base Revenue | Lift % | Incremental Revenue | Promo Cost | Net Benefit |",
            "|---------|-------------|--------|---------------------|------------|-------------|",
        ]

        roi_segments = roi_data.get("segments", {})
        for cluster_id, info in roi_segments.items():
            base_rev = info.get("base_revenue", 0.0)
            lift = info.get("estimated_lift_pct", 0.0)
            incr = info.get("incremental_revenue", 0.0)
            cost = info.get("promo_cost", 0.0)
            net  = info.get("net_benefit", 0.0)
            lines.append(
                f"| Segment {cluster_id} | ${base_rev:,.2f} | {lift:.1f}% "
                f"| ${incr:,.2f} | ${cost:,.2f} | ${net:,.2f} |"
            )

        lines += [
            "",
            "---\n",
            "## Revenue Share by Segment\n",
            "| Segment | Customers | Revenue Share |",
            "|---------|-----------|---------------|",
        ]

        rev_segments = rev_data.get("segments", {})
        for cluster_id, info in rev_segments.items():
            size = info.get("segment_size", 0)
            share = info.get("revenue_share_pct", 0.0)
            lines.append(f"| Segment {cluster_id} | {size:,} | {share:.1f}% |")

        lines += [
            "",
            "---\n",
            "## Recommendation\n",
            f"The promo programme is projected to deliver an overall ROI of **{overall_roi_pct:.1f}%** "
            f"with a net benefit of **${total_net_benefit:,.2f}** against a promo investment of "
            f"**${total_promo_cost:,.2f}**. High-sensitivity segments should receive priority budget allocation.",
            "",
            "_Finance Analyst Agent_",
        ]

        report_path = os.path.join(PATHS["reports"], "financial_report.md")
        with open(report_path, "w") as f:
            f.write("\n".join(lines))

        return json.dumps({
            "status": "ok",
            "financial_report_path": report_path,
            "overall_roi_pct": overall_roi_pct,
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
