import json
import os
from agentic_ai.agents.base_agent import BaseAgent
from agentic_ai.communication.message_schema import MessageType, Topic
from agentic_ai.tools.report_tools import (
    generate_visualization, create_segment_report, export_dashboard_data
)
from src.config.settings import PATHS


class BusinessAnalyst1Agent(BaseAgent):
    def __init__(self, message_bus, project_state, anthropic_client):
        tools = [
            {
                "name": "generate_visualization",
                "description": "Generate a chart from dataset and save to outputs/visualizations/",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "data_path":  {"type": "string"},
                        "chart_type": {"type": "string", "enum": ["bar", "hist", "scatter"]},
                        "x_col":      {"type": "string"},
                        "y_col":      {"type": "string"},
                        "title":      {"type": "string"},
                    },
                    "required": ["data_path", "title"],
                },
            },
            {
                "name": "create_segment_report",
                "description": "Generate a Markdown segment profile report",
                "input_schema": {
                    "type": "object",
                    "properties": {"segment_summary_path": {"type": "string"}},
                    "required": ["segment_summary_path"],
                },
            },
            {
                "name": "export_dashboard_data",
                "description": "Write all dashboard-ready data files",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "segment_profiles_path":  {"type": "string"},
                        "recommendations_path":   {"type": "string"},
                    },
                },
            },
            {
                "name": "notify_business_lead",
                "description": "Notify Business Lead when reports are ready",
                "input_schema": {
                    "type": "object",
                    "properties": {"summary": {"type": "string"}},
                    "required": ["summary"],
                },
            },
        ]
        super().__init__(
            agent_id="business_analyst_1",
            role="Business Analyst 1 — Reporting & Visualization",
            topics=[Topic.BUSINESS, Topic.PIPELINE],
            message_bus=message_bus,
            project_state=project_state,
            anthropic_client=anthropic_client,
            tools=tools,
        )
        self.system_prompt = (
            "You are Business Analyst 1, specializing in reporting and visualization. "
            "Phase 5: Generate visualizations for customer segments and promo effectiveness, "
            "create a segment profile Markdown report, export dashboard-ready data files, "
            "then notify Business Lead that reports are ready."
        )

    async def _handle_task(self, message):
        processed = PATHS["processed_data"]
        reports = PATHS["reports"]
        payload_str = json.dumps(message.payload)
        prompt = (
            f"Message from {message.from_agent} [{message.message_type}]:\n{payload_str}\n"
            f"Processed data: {processed}, Reports: {reports}\n"
            "Create visualizations and reports, export dashboard data, then notify Business Lead."
        )
        await self._run_tool_loop(prompt)

    async def _tool_generate_visualization(
        self, data_path: str, title: str, chart_type: str = "bar",
        x_col: str = None, y_col: str = None
    ) -> str:
        return generate_visualization(data_path, chart_type, x_col, y_col, title)

    async def _tool_create_segment_report(self, segment_summary_path: str) -> str:
        return create_segment_report(segment_summary_path)

    async def _tool_export_dashboard_data(
        self, segment_profiles_path: str = None, recommendations_path: str = None
    ) -> str:
        snap = self.project_state.snapshot()
        kpis = snap.get("kpis", {})
        return export_dashboard_data(segment_profiles_path, recommendations_path, kpis, snap)

    async def _tool_notify_business_lead(self, summary: str) -> str:
        await self.send_message(
            to="business_lead",
            topic=Topic.BUSINESS,
            message_type=MessageType.REPORT_READY,
            payload={"summary": summary, "agent": self.agent_id},
        )
        return f"Business Lead notified: {summary}"
