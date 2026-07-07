import json
import os
from agentic_ai.agents.base_agent import BaseAgent
from agentic_ai.communication.message_schema import MessageType, Topic
from src.config.settings import PATHS


class ProductManagerAgent(BaseAgent):
    def __init__(self, message_bus, project_state, anthropic_client):
        tools = [
            {
                "name": "create_product_roadmap",
                "description": (
                    "Create a phased product roadmap JSON based on the current project state snapshot. "
                    "Q1=data pipeline, Q2=segmentation, Q3=recommendations, Q4=real-time scoring. "
                    "Saves to outputs/reports/product_roadmap.json."
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "project_state_snapshot": {"type": "object"},
                    },
                    "required": ["project_state_snapshot"],
                },
            },
            {
                "name": "prioritize_features",
                "description": (
                    "Return a ranked priority list of 6 product features based on business impact. "
                    "Saves to outputs/reports/feature_priorities.json."
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "insights_summary": {"type": "string"},
                    },
                    "required": ["insights_summary"],
                },
            },
            {
                "name": "create_sprint_plan",
                "description": (
                    "Generate a 2-week sprint plan Markdown document for the given project phase. "
                    "Saves to outputs/reports/sprint_plan_{phase}.md."
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "phase": {"type": "string"},
                    },
                    "required": ["phase"],
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
            agent_id="product_manager_pm",
            role="Product Manager — Roadmap & Prioritization",
            topics=[Topic.MANAGEMENT, Topic.BUSINESS],
            message_bus=message_bus,
            project_state=project_state,
            anthropic_client=anthropic_client,
            tools=tools,
        )
        self.system_prompt = (
            "You are the Product Manager responsible for the promo optimization platform roadmap. "
            "You prioritize features based on business impact and technical feasibility, and translate "
            "project outcomes into product requirements and sprint plans. "
            "Steps: create_product_roadmap → prioritize_features → create_sprint_plan → notify_pm. "
            "Use the project state snapshot to inform roadmap decisions."
        )

    async def _handle_task(self, message):
        reports = PATHS["reports"]
        payload_str = json.dumps(message.payload)
        snap = self.project_state.snapshot()
        prompt = (
            f"Message from {message.from_agent} [{message.message_type}]:\n{payload_str}\n"
            f"Reports directory: {reports}\n"
            f"Project state: {json.dumps(snap)}\n"
            "Create the product roadmap, prioritize features, generate a sprint plan, "
            "then notify Project Manager."
        )
        await self._run_tool_loop(prompt)

    def _tool_create_product_roadmap(self, project_state_snapshot: dict) -> str:
        os.makedirs(PATHS["reports"], exist_ok=True)

        # Determine completion status from snapshot for context
        phases_complete = project_state_snapshot.get("phases_complete", [])
        current_phase = project_state_snapshot.get("current_phase", "unknown")
        datasets_registered = list(project_state_snapshot.get("datasets", {}).keys())
        models_registered = list(project_state_snapshot.get("models", {}).keys())

        roadmap = {
            "product": "Trade Promo Optimization Platform",
            "version": "1.0",
            "current_project_phase": current_phase,
            "phases_complete": phases_complete,
            "quarters": {
                "Q1": {
                    "theme": "Data Pipeline & Infrastructure",
                    "goal": "Reliable, automated data ingestion and processing pipeline",
                    "epics": [
                        "Automated data ingestion from POS and CRM systems",
                        "Data quality monitoring and alerting",
                        "Scalable feature engineering pipeline",
                        "CI/CD for data pipeline",
                    ],
                    "kpis": ["Pipeline uptime > 99%", "Data freshness < 24h", "Zero data loss"],
                    "status": "complete" if "phase_1" in phases_complete else "in_progress",
                    "datasets_available": datasets_registered,
                },
                "Q2": {
                    "theme": "Customer Segmentation",
                    "goal": "Production-grade customer segmentation with automated refresh",
                    "epics": [
                        "KMeans and DBSCAN segmentation models",
                        "Automated model retraining pipeline",
                        "Segment drift monitoring",
                        "Segment profile dashboard",
                    ],
                    "kpis": [
                        "Silhouette score > 0.3",
                        "Segment stability > 80% month-over-month",
                        "Model retraining < 2h",
                    ],
                    "status": "complete" if "phase_4" in phases_complete else "planned",
                    "models_available": models_registered,
                },
                "Q3": {
                    "theme": "Personalised Recommendations",
                    "goal": "Real-time product recommendations per customer segment",
                    "epics": [
                        "Collaborative filtering recommender model",
                        "REST API for real-time recommendations",
                        "A/B testing framework for recommendation strategies",
                        "Campaign management integration",
                    ],
                    "kpis": [
                        "Precision@10 > 0.15",
                        "Recommendation latency < 200ms",
                        "Click-through rate uplift > 10%",
                    ],
                    "status": "planned",
                },
                "Q4": {
                    "theme": "Real-Time Promo Scoring",
                    "goal": "Sub-100ms promo eligibility scoring at point-of-sale",
                    "epics": [
                        "Real-time feature serving with feature store",
                        "Online scoring microservice",
                        "Promo budget optimisation engine",
                        "Executive ROI dashboard",
                    ],
                    "kpis": [
                        "Scoring latency P99 < 100ms",
                        "Promo ROI > 15%",
                        "Budget utilisation accuracy > 95%",
                    ],
                    "status": "backlog",
                },
            },
        }

        output_path = os.path.join(PATHS["reports"], "product_roadmap.json")
        with open(output_path, "w") as f:
            json.dump(roadmap, f, indent=2)

        return json.dumps({
            "status": "ok",
            "roadmap_path": output_path,
            "quarters_defined": list(roadmap["quarters"].keys()),
            "current_project_phase": current_phase,
        })

    def _tool_prioritize_features(self, insights_summary: str) -> str:
        os.makedirs(PATHS["reports"], exist_ok=True)

        # Static priority list ranked by business impact x technical feasibility
        feature_priorities = [
            {
                "rank": 1,
                "feature": "Real-time customer segment lookup API",
                "impact": "high",
                "effort": "medium",
                "rationale": (
                    "Enables downstream systems (POS, CRM, marketing tools) to consume "
                    "segment data instantly. Highest leverage for revenue impact."
                ),
                "quarter": "Q3",
            },
            {
                "rank": 2,
                "feature": "Automated segment retraining pipeline",
                "impact": "high",
                "effort": "medium",
                "rationale": (
                    "Segments drift over time. Automated retraining maintains accuracy "
                    "without manual intervention, protecting long-term ROI."
                ),
                "quarter": "Q2",
            },
            {
                "rank": 3,
                "feature": "Promo ROI dashboard (executive-facing)",
                "impact": "high",
                "effort": "low",
                "rationale": (
                    "Stakeholders need visibility into promo performance. "
                    "Low effort for high perceived business value."
                ),
                "quarter": "Q4",
            },
            {
                "rank": 4,
                "feature": "A/B testing framework for promo strategies",
                "impact": "medium",
                "effort": "medium",
                "rationale": (
                    "Required to validate that model-driven promos outperform baseline. "
                    "Enables iterative improvement of recommendation quality."
                ),
                "quarter": "Q3",
            },
            {
                "rank": 5,
                "feature": "Feature store for real-time scoring",
                "impact": "high",
                "effort": "high",
                "rationale": (
                    "Critical for sub-100ms scoring at POS. High effort but unlocks "
                    "the full real-time promo scoring capability in Q4."
                ),
                "quarter": "Q4",
            },
            {
                "rank": 6,
                "feature": "Segment drift monitoring & alerting",
                "impact": "medium",
                "effort": "low",
                "rationale": (
                    "Ensures data quality and model reliability over time. "
                    "Low effort and prevents silent model degradation."
                ),
                "quarter": "Q2",
            },
        ]

        output = {
            "insights_summary": insights_summary,
            "prioritization_method": "Impact x Feasibility matrix",
            "features": feature_priorities,
        }

        output_path = os.path.join(PATHS["reports"], "feature_priorities.json")
        with open(output_path, "w") as f:
            json.dump(output, f, indent=2)

        return json.dumps({
            "status": "ok",
            "feature_priorities_path": output_path,
            "features_ranked": len(feature_priorities),
            "top_feature": feature_priorities[0]["feature"],
        })

    def _tool_create_sprint_plan(self, phase: str) -> str:
        os.makedirs(PATHS["reports"], exist_ok=True)

        phase_lower = phase.lower().replace(" ", "_")

        sprint_tasks = {
            "data_pipeline": [
                ("Week 1", [
                    "Set up raw data ingestion from POS CSV exports",
                    "Implement data validation and schema checks",
                    "Build processed data transformation scripts",
                    "Write unit tests for pipeline components",
                ]),
                ("Week 2", [
                    "Integrate automated pipeline scheduling (Airflow / cron)",
                    "Add data quality monitoring and alerting",
                    "Document pipeline architecture and data dictionary",
                    "Demo to stakeholders and gather feedback",
                ]),
            ],
            "segmentation": [
                ("Week 1", [
                    "Train KMeans segmentation model on feature matrix",
                    "Tune n_clusters using elbow method and silhouette analysis",
                    "Run Senior DS validation and bias checks",
                    "Export model and generate segment profiles",
                ]),
                ("Week 2", [
                    "Build automated retraining script with scheduling",
                    "Create segment profile visualisations",
                    "Set up model monitoring config",
                    "Stakeholder review of segments and sign-off",
                ]),
            ],
            "recommendations": [
                ("Week 1", [
                    "Train collaborative filtering recommender (NMF/ALS)",
                    "Evaluate Precision@10 and coverage metrics",
                    "Wrap model in REST API spec",
                    "Integrate recommendations with segment profiles",
                ]),
                ("Week 2", [
                    "Deploy recommendation API to staging environment",
                    "Build A/B test framework scaffold",
                    "Create campaign brief from recommendation outputs",
                    "UAT with marketing team and iterate",
                ]),
            ],
            "real_time_scoring": [
                ("Week 1", [
                    "Design feature store schema for online serving",
                    "Build real-time feature computation service",
                    "Implement promo scoring microservice",
                    "Load test scoring API (target: P99 < 100ms)",
                ]),
                ("Week 2", [
                    "Integrate scoring API with POS system (sandbox)",
                    "Build executive ROI dashboard (Streamlit / BI tool)",
                    "Complete end-to-end integration testing",
                    "Go-live readiness review and sign-off",
                ]),
            ],
        }

        # Fallback sprint for unknown phases
        tasks = sprint_tasks.get(phase_lower, [
            ("Week 1", [
                "Define acceptance criteria and success metrics",
                "Break down epics into user stories",
                "Assign tasks to team members",
                "Set up tracking in project management tool",
            ]),
            ("Week 2", [
                "Execute planned tasks",
                "Daily stand-ups and blocker resolution",
                "Mid-sprint demo to stakeholders",
                "Sprint review and retrospective",
            ]),
        ])

        lines = [
            f"# Sprint Plan — {phase.title()}\n",
            "_Generated by Product Manager Agent_\n",
            f"**Phase:** {phase.title()}  ",
            "**Sprint Duration:** 2 weeks  ",
            "**Goal:** Complete all planned deliverables for this phase and achieve stakeholder sign-off.\n",
            "---\n",
        ]

        for week_label, week_tasks in tasks:
            lines.append(f"## {week_label}\n")
            lines.append("| # | Task | Owner | Status |")
            lines.append("|---|------|-------|--------|")
            for i, task in enumerate(week_tasks, 1):
                lines.append(f"| {i} | {task} | TBD | To Do |")
            lines.append("")

        lines += [
            "---\n",
            "## Definition of Done\n",
            "- [ ] All tasks completed and reviewed",
            "- [ ] Unit tests passing",
            "- [ ] Output artefacts saved to outputs/",
            "- [ ] Stakeholder sign-off received",
            "",
            "_Product Manager Agent_",
        ]

        plan_path = os.path.join(PATHS["reports"], f"sprint_plan_{phase_lower}.md")
        with open(plan_path, "w") as f:
            f.write("\n".join(lines))

        return json.dumps({
            "status": "ok",
            "sprint_plan_path": plan_path,
            "phase": phase,
            "weeks_planned": len(tasks),
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
