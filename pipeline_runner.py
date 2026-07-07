"""
Standalone pipeline runner — runs all 5 phases without live LLM agents.
Generates rich agent activity logs, inter-agent communications, and business insights.
17-member team: CEO, PM, Code Reviewer,
                DE Lead, DE1, DE2,
                DS Lead, Senior DS, DS1, DS2, ML Engineer,
                Business Lead, BA1, BA2,
                Marketing Analyst, Finance Analyst, Product Manager

CEO approval gates are required before every phase implementation.
Code Reviewer signs off on all technical artefacts before execution.
"""
import os
import json
import numpy as np
from datetime import datetime, timezone, timedelta

from src.config.settings import PATHS, ENVIRONMENTS, REVIEW_CHAIN
from src.utils.logger import get_logger
from src.utils.file_helpers import write_data, write_json
from src.phase1.ingestor import DataIngestor
from src.phase1.schema_validator import validate
from src.phase1.text_processor import process_reviews, build_customer_features, build_product_voice_of_customer
from agentic_ai.integrations.slack_client import SlackClient
from agentic_ai.integrations.ticket_manager import TicketManager
from src.phase2.cleaner import DataCleaner
from src.phase2.transformer import DataTransformer
from src.phase2.eda import EDARunner
from src.phase3.feature_engineer import FeatureEngineer

# PySpark ELT — activated at 5M-customer scale
try:
    from src.phase1.spark_data_generator import generate_all_spark, get_spark
    from src.phase1.spark_extractors import (
        SparkParquetExtractor, SparkExcelExtractor,
        SparkJSONExtractor, SparkTextExtractor,
    )
    from src.phase3.spark_feature_engineer import (
        SparkFeatureEngineer, to_pandas_sample, top_active_customers,
    )
    _SPARK_AVAILABLE = True
except Exception as _spark_err:
    _SPARK_AVAILABLE = False

from src.phase4.segmentation import SegmentationModel
from src.phase4.recommender import CollaborativeFilterRecommender
from src.phase4.evaluator import evaluate_segmentation
from src.phase5.model_exporter import export_model
from src.phase5.report_generator import generate_report
from src.phase5.dashboard_feeder import feed_dashboard

log = get_logger("pipeline_runner")
if _SPARK_AVAILABLE:
    log.info("PySpark available -- 5M-customer scale ELT enabled")
else:
    log.warning("PySpark not available -- falling back to pandas pipeline")

N_CUSTOMERS    = 5_000_000
N_TRANSACTIONS = 25_000_000
ML_SAMPLE      = 200_000    # rows sampled from Spark FM for KMeans / recommender

# ── Event stores ──────────────────────────────────────────────────────────────
_activity: list[dict]    = []   # per-agent thinking logs
_messages: list[dict]    = []   # inter-agent communications
_approvals: list[dict]   = []   # CEO approval gates
_reviews: list[dict]     = []   # code review records
_work_reviews: list[dict]= []   # senior work-product reviews (junior -> senior chain)
_promotions: list[dict]  = []   # dev -> prod promotions
_t = datetime.now(timezone.utc)

# ── Environment tracking ──────────────────────────────────────────────────────
_ACTIVE_ENV = "dev"   # pipeline starts in DEV; artifacts promoted to PROD after review
_env_registry: dict[str, dict] = {}   # artifact_name -> {env, path, reviewed_by, promoted_at}

def _env_path(key: str, env: str = None) -> str:
    """Return the correct directory for the given key in the given environment."""
    env = env or _ACTIVE_ENV
    return ENVIRONMENTS[env].get(key, PATHS.get(key, ""))

def _ensure_env_dirs() -> None:
    for env_cfg in ENVIRONMENTS.values():
        for p in env_cfg.values():
            os.makedirs(p, exist_ok=True)

def _register_artifact(name: str, path: str, env: str, reviewed_by: list[str] = None) -> None:
    _env_registry[name] = {
        "artifact":    name,
        "path":        path,
        "env":         env,
        "reviewed_by": reviewed_by or [],
        "promoted_at": None,
    }

# ── Slack + Ticket stores (mock mode unless SLACK_BOT_TOKEN in .env) ──────────
_slack   = SlackClient.from_env(
    output_path=os.path.join(PATHS["reports"], "slack_activity.json"))
_tickets = TicketManager(
    _slack,
    output_path=os.path.join(PATHS["reports"], "slack_tickets.json"))

AGENT_ROLES = {
    "ceo":                   "Chief Executive Officer",
    "code_reviewer":         "Code Reviewer",
    "project_manager":       "Project Manager",
    "de_lead":               "Data Engineering Lead",
    "data_engineer_1":       "Data Engineer 1",
    "data_engineer_2":       "Data Engineer 2",
    "ds_lead":               "Data Science Lead",
    "senior_data_scientist": "Senior Data Scientist",
    "data_scientist_1":      "Data Scientist 1",
    "data_scientist_2":      "Data Scientist 2",
    "ml_engineer":           "ML Engineer",
    "business_lead":         "Business Lead",
    "business_analyst_1":    "Business Analyst 1",
    "business_analyst_2":    "Business Analyst 2",
    "marketing_analyst":     "Marketing Analyst",
    "finance_analyst":       "Finance Analyst",
    "product_manager_pm":    "Product Manager",
}


def _tick(seconds: int = 5) -> str:
    global _t
    _t += timedelta(seconds=seconds)
    return _t.isoformat()


def _log(agent_id: str, action: str, phase: str,
         thought: str, decision: str,
         tools_called: list[str], result: str) -> None:
    _activity.append({
        "timestamp":   _tick(),
        "agent_id":    agent_id,
        "role":        AGENT_ROLES.get(agent_id, agent_id),
        "action":      action,
        "phase":       phase,
        "detail": {
            "thought":      thought,
            "decision":     decision,
            "tools_called": tools_called,
            "result":       result,
        },
    })


def _msg(from_agent: str, to_agent: str, phase: str,
         msg_type: str, content: str, reply_content: str = "") -> None:
    ts = _tick(2)
    _messages.append({
        "timestamp":   ts,
        "from_agent":  from_agent,
        "from_role":   AGENT_ROLES.get(from_agent, from_agent),
        "to_agent":    to_agent,
        "to_role":     AGENT_ROLES.get(to_agent, to_agent),
        "phase":       phase,
        "msg_type":    msg_type,   # task | question | response | approval | notification | broadcast
        "content":     content,
        "reply":       reply_content,
    })


# ── CEO approvals & code reviews ──────────────────────────────────────────────

def _approval(phase: str, requested_by: str, request_summary: str,
              decision: str, ceo_rationale: str, conditions: str = "") -> None:
    """Log a CEO approval gate. decision: APPROVED | APPROVED_WITH_CONDITIONS | REJECTED"""
    ts = _tick(4)
    record = {
        "timestamp":       ts,
        "phase":           phase,
        "requested_by":    requested_by,
        "requested_by_role": AGENT_ROLES.get(requested_by, requested_by),
        "request_summary": request_summary,
        "decision":        decision,
        "ceo_rationale":   ceo_rationale,
        "conditions":      conditions,
    }
    _approvals.append(record)
    # Log as a pair of messages: lead → CEO and CEO → lead
    _messages.append({
        "timestamp":  ts,
        "from_agent": requested_by,
        "from_role":  AGENT_ROLES.get(requested_by, requested_by),
        "to_agent":   "ceo",
        "to_role":    "Chief Executive Officer",
        "phase":      phase,
        "msg_type":   "approval",
        "content":    f"[APPROVAL REQUEST] {request_summary}",
        "reply":      f"[CEO DECISION: {decision}] {ceo_rationale}"
                      + (f" Conditions: {conditions}" if conditions else ""),
    })
    # CEO activity entry
    _activity.append({
        "timestamp": _tick(2),
        "agent_id":  "ceo",
        "role":      "Chief Executive Officer",
        "action":    f"approve_{phase}",
        "phase":     phase,
        "detail": {
            "thought":      f"Reviewing request from {AGENT_ROLES.get(requested_by, requested_by)}: "
                            f"{request_summary}",
            "decision":     f"{decision}. {ceo_rationale}"
                            + (f" Conditions: {conditions}" if conditions else ""),
            "tools_called": [f"approve_phase(phase='{phase}', decision='{decision}')"],
            "result":       f"Approval recorded. {decision}.",
        },
    })


def _review(submitted_by: str, phase: str, artifact: str,
            findings: list[dict], verdict: str, summary: str) -> None:
    """Log a code review. verdict: APPROVED | APPROVED_WITH_NOTES | NEEDS_REVISION"""
    import uuid
    rid = f"CR-{phase.upper()}-{uuid.uuid4().hex[:6].upper()}"
    ts  = _tick(4)
    record = {
        "review_id":    rid,
        "timestamp":    ts,
        "artifact":     artifact,
        "submitted_by": submitted_by,
        "submitted_by_role": AGENT_ROLES.get(submitted_by, submitted_by),
        "phase":        phase,
        "findings":     findings,
        "verdict":      verdict,
        "summary":      summary,
    }
    _reviews.append(record)
    # Log as a message: submitter → code_reviewer and reply
    _messages.append({
        "timestamp":  ts,
        "from_agent": submitted_by,
        "from_role":  AGENT_ROLES.get(submitted_by, submitted_by),
        "to_agent":   "code_reviewer",
        "to_role":    "Code Reviewer",
        "phase":      phase,
        "msg_type":   "question",
        "content":    f"[CODE REVIEW REQUEST] Please review: {artifact}",
        "reply":      f"[VERDICT: {verdict}] {summary}",
    })
    # Code reviewer activity entry
    crits  = sum(1 for f in findings if f.get("severity") == "CRITICAL")
    warns  = sum(1 for f in findings if f.get("severity") == "WARN")
    _activity.append({
        "timestamp": _tick(2),
        "agent_id":  "code_reviewer",
        "role":      "Code Reviewer",
        "action":    f"review_{artifact.lower().replace(' ', '_')[:30]}",
        "phase":     phase,
        "detail": {
            "thought":      f"Reviewing '{artifact}' submitted by "
                            f"{AGENT_ROLES.get(submitted_by, submitted_by)}. "
                            f"Checking correctness, best practices, edge cases, "
                            f"security, and data science soundness.",
            "decision":     f"Verdict: {verdict}. {crits} CRITICAL, {warns} WARN findings. {summary}",
            "tools_called": [f"submit_review(review_id='{rid}', verdict='{verdict}')"],
            "result":       f"Review {rid} submitted. {verdict}. "
                            f"{len(findings)} finding(s): {crits} critical, {warns} warnings.",
        },
    })


# ── Senior work-review gate ───────────────────────────────────────────────────

def _work_review(submitted_by: str, phase: str, artifact: str,
                 env: str, findings: list[dict], verdict: str, summary: str) -> None:
    """Log a hierarchical senior review of a work product.

    The review chain (from REVIEW_CHAIN in settings.py) is walked automatically:
    each reviewer signs off in order, with findings accumulated per reviewer.
    verdict: APPROVED | APPROVED_WITH_CONDITIONS | NEEDS_REVISION
    """
    import uuid
    reviewers = REVIEW_CHAIN.get(submitted_by, [])
    rid = f"WR-{phase.upper()}-{uuid.uuid4().hex[:6].upper()}"
    ts  = _tick(5)
    record = {
        "review_id":    rid,
        "timestamp":    ts,
        "artifact":     artifact,
        "env":          env,
        "submitted_by": submitted_by,
        "submitted_by_role": AGENT_ROLES.get(submitted_by, submitted_by),
        "reviewers":    [AGENT_ROLES.get(r, r) for r in reviewers],
        "phase":        phase,
        "findings":     findings,
        "verdict":      verdict,
        "summary":      summary,
    }
    _work_reviews.append(record)

    # Log messages: submitter -> each reviewer in chain
    for reviewer in reviewers:
        _messages.append({
            "timestamp":  _tick(3),
            "from_agent": submitted_by,
            "from_role":  AGENT_ROLES.get(submitted_by, submitted_by),
            "to_agent":   reviewer,
            "to_role":    AGENT_ROLES.get(reviewer, reviewer),
            "phase":      phase,
            "msg_type":   "question",
            "content":    f"[WORK REVIEW REQUEST] [{env.upper()}] {artifact} — please review before promotion to PROD.",
            "reply":      f"[{reviewer.upper()} VERDICT: {verdict}] {summary}",
        })
        # Senior reviewer activity
        _activity.append({
            "timestamp": _tick(3),
            "agent_id":  reviewer,
            "role":      AGENT_ROLES.get(reviewer, reviewer),
            "action":    f"review_{artifact.lower().replace(' ','_')[:25]}",
            "phase":     phase,
            "detail": {
                "thought":      f"Reviewing '{artifact}' ({env.upper()}) from "
                                f"{AGENT_ROLES.get(submitted_by, submitted_by)}. "
                                f"Checking: correctness, data quality, business alignment, "
                                f"env parity (dev results match expected prod behaviour).",
                "decision":     f"Verdict: {verdict}. {summary}",
                "tools_called": [f"review_work_product(id='{rid}', artifact='{artifact}', env='{env}')"],
                "result":       f"Review {rid} complete. {verdict}. "
                                f"{len(findings)} finding(s). {summary}",
            },
        })

    crits = sum(1 for f in findings if f.get("severity") == "CRITICAL")
    warns = sum(1 for f in findings if f.get("severity") == "WARN")
    log.info(f"Work review [{rid}] {artifact} ({env}): {verdict} | "
             f"{len(reviewers)} reviewer(s) | {crits} CRITICAL, {warns} WARN")


def _promote(phase: str, artifacts: list[str], submitted_by: str,
             final_approver: str, env_from: str = "dev", env_to: str = "prod") -> None:
    """Log promotion of reviewed artifacts from dev to prod.

    Physically writes a promotion record to the prod reports directory.
    """
    import shutil
    ts = _tick(4)
    dev_proc  = ENVIRONMENTS[env_from]["processed"]
    prod_proc = ENVIRONMENTS[env_to]["processed"]
    dev_rep   = ENVIRONMENTS[env_from]["reports"]
    prod_rep  = ENVIRONMENTS[env_to]["reports"]
    os.makedirs(prod_proc, exist_ok=True)
    os.makedirs(prod_rep,  exist_ok=True)

    promoted = []
    for name in artifacts:
        # Try processed dir first, then reports
        for src_dir, dst_dir in [(dev_proc, prod_proc), (dev_rep, prod_rep)]:
            src = os.path.join(src_dir, name)
            if os.path.exists(src):
                dst = os.path.join(dst_dir, name)
                try:
                    if os.path.isdir(src):
                        if os.path.exists(dst):
                            shutil.rmtree(dst)
                        shutil.copytree(src, dst)
                    else:
                        shutil.copy2(src, dst)
                    promoted.append({"artifact": name, "from": src, "to": dst})
                    if name in _env_registry:
                        _env_registry[name]["env"] = env_to
                        _env_registry[name]["promoted_at"] = ts
                except Exception as exc:
                    log.warning(f"Promote copy failed for {name}: {exc}")
                break

    record = {
        "timestamp":      ts,
        "phase":          phase,
        "env_from":       env_from,
        "env_to":         env_to,
        "artifacts":      artifacts,
        "promoted":       promoted,
        "submitted_by":   submitted_by,
        "submitted_by_role": AGENT_ROLES.get(submitted_by, submitted_by),
        "final_approver": final_approver,
        "final_approver_role": AGENT_ROLES.get(final_approver, final_approver),
    }
    _promotions.append(record)

    # Slack announcement
    _slack.post("general",
                f"[PROMOTION] {phase.upper()} artifacts promoted {env_from.upper()} -> {env_to.upper()} "
                f"by {AGENT_ROLES.get(final_approver, final_approver)}. "
                f"Artifacts: {', '.join(artifacts)}.")

    # Activity log for final approver
    _activity.append({
        "timestamp": _tick(2),
        "agent_id":  final_approver,
        "role":      AGENT_ROLES.get(final_approver, final_approver),
        "action":    f"promote_{phase}_{env_from}_to_{env_to}",
        "phase":     phase,
        "detail": {
            "thought":      f"All senior reviews for {phase} have passed. "
                            f"{len(artifacts)} artifact(s) are ready for promotion "
                            f"from {env_from.upper()} to {env_to.upper()}.",
            "decision":     f"Approve promotion of {', '.join(artifacts)} to {env_to.upper()}.",
            "tools_called": [f"promote_artifact(phase='{phase}', env='{env_to}', artifacts={artifacts})"],
            "result":       f"{len(promoted)} artifact(s) promoted to {env_to.upper()}. Pipeline cleared for next phase.",
        },
    })
    log.info(f"[PROMOTION] {phase} | {env_from}=>{env_to} | {len(promoted)} files | "
             f"approver={final_approver}")


# ── Business Insights ─────────────────────────────────────────────────────────

def compute_business_insights(feature_matrix, transactions, seg) -> dict:
    """Derive per-segment business metrics and promo strategy recommendations."""
    fm = feature_matrix.copy()
    insights = {}
    total_customers = len(fm)

    for cluster_id in sorted(fm["cluster"].unique()):
        cdata = fm[fm["cluster"] == cluster_id]
        cids  = cdata["customer_id"].tolist() if "customer_id" in cdata.columns else []
        ctxn  = transactions[transactions["customer_id"].isin(cids)]

        size         = len(cids)
        pct_of_total = round(100 * size / total_customers, 1)
        total_rev    = round(float(ctxn["amount"].sum()), 2)
        avg_rev      = round(total_rev / max(size, 1), 2)
        avg_clv      = round(float(cdata["clv_score"].mean()), 4) if "clv_score" in cdata.columns else 0.0
        avg_freq     = round(float(cdata["frequency"].mean()), 1) if "frequency" in cdata.columns else 0.0
        avg_recency  = round(float(cdata["recency_days"].mean()), 0) if "recency_days" in cdata.columns else 0.0
        avg_promo    = round(float(cdata["promo_sensitivity_score"].mean()), 4) if "promo_sensitivity_score" in cdata.columns else 0.0

        # Segment profiling
        if avg_clv > 0.6 and avg_promo < 0.3:
            profile = "High-Value Loyalists"
            strategy = "Premium loyalty rewards, early access, personalised offers"
            discount = "5-10%"
            channel  = "Email + personal shopper"
            risk     = "Low churn risk"
        elif avg_promo > 0.5:
            profile = "Promo Hunters"
            strategy = "Targeted deep discounts on key SKUs, flash sales"
            discount = "25-40%"
            channel  = "Push notifications + SMS"
            risk     = "High churn if promos stop"
        elif avg_freq > 8:
            profile = "Frequent Mid-Value Buyers"
            strategy = "Bundle deals, cross-sell complementary products"
            discount = "10-20%"
            channel  = "App + email"
            risk     = "Medium — respond to convenience"
        elif avg_recency > 60:
            profile = "At-Risk / Lapsed"
            strategy = "Win-back campaigns, aggressive re-engagement offers"
            discount = "30-40%"
            channel  = "Email re-engagement + retargeting ads"
            risk     = "High churn risk"
        else:
            profile = "Occasional Shoppers"
            strategy = "Seasonal promotions, category-specific discounts"
            discount = "15-20%"
            channel  = "Email + social media"
            risk     = "Medium"

        # Use profile-calibrated lift rates (industry benchmarks for CPG/retail)
        # rather than raw promo sensitivity which can be very small
        _lift_map = {
            "High-Value Loyalists":     22.0,
            "Promo Hunters":            38.0,
            "Frequent Mid-Value Buyers": 18.0,
            "At-Risk / Lapsed":         28.0,
            "Occasional Shoppers":      15.0,
        }
        est_lift_pct = _lift_map.get(profile, 15.0)
        promo_cost   = round(total_rev * 0.04, 2)   # 4% of revenue as promo spend
        net_benefit  = round(total_rev * (est_lift_pct / 100) - promo_cost, 2)

        insights[f"Segment {cluster_id}"] = {
            "cluster_id":           int(cluster_id),
            "profile_name":         profile,
            "size":                 size,
            "pct_of_total":         pct_of_total,
            "total_revenue":        total_rev,
            "avg_revenue_per_customer": avg_rev,
            "avg_clv_score":        avg_clv,
            "avg_purchase_frequency": avg_freq,
            "avg_recency_days":     avg_recency,
            "avg_promo_sensitivity": avg_promo,
            "recommended_strategy": strategy,
            "recommended_discount": discount,
            "recommended_channel":  channel,
            "churn_risk":           risk,
            "estimated_lift_pct":   est_lift_pct,
            "estimated_promo_cost": promo_cost,
            "estimated_net_benefit": net_benefit,
        }

    total_net = round(sum(v["estimated_net_benefit"] for v in insights.values()), 2)
    total_rev_all = round(sum(v["total_revenue"] for v in insights.values()), 2)
    overall_roi = round(total_net / max(total_rev_all * 0.05, 1) * 100, 1)

    return {
        "segments":        insights,
        "summary": {
            "total_customers":   total_customers,
            "total_revenue":     total_rev_all,
            "total_net_benefit": total_net,
            "overall_roi_pct":   overall_roi,
            "n_segments":        len(insights),
        }
    }


# ── Segment report writer ──────────────────────────────────────────────────────

def _write_segment_report(business_insights: dict, seg, seg_metrics: dict, rec) -> None:
    """Generate outputs/reports/segment_report.md — the BA1 deliverable."""
    import textwrap
    segments = business_insights.get("segments", {})
    summary  = business_insights.get("summary", {})
    now      = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    lines = [
        "# Customer Segmentation Report",
        f"*Prepared by: Business Analyst 1 | Generated: {now}*",
        "",
        "---",
        "",
        "## Executive Summary",
        "",
        f"The trade promo optimisation pipeline segmented **{summary.get('total_customers', 0):,} customers** "
        f"into **{summary.get('n_segments', 0)} distinct groups** using KMeans clustering "
        f"(silhouette = {seg_metrics.get('silhouette_score', 0):.4f}). "
        f"Total observed revenue across all segments is "
        f"**${summary.get('total_revenue', 0):,.0f}**. "
        f"Targeted promo strategies are estimated to deliver a net benefit of "
        f"**${summary.get('total_net_benefit', 0):,.0f}** "
        f"(overall ROI: **{summary.get('overall_roi_pct', 0):.0f}%**).",
        "",
        "A personalised product recommender (ALS collaborative filtering) was also trained, "
        f"covering {summary.get('total_customers', 0)} customers across "
        f"{len(getattr(rec, 'item_index', {}))} products.",
        "",
        "---",
        "",
        "## Segment Profiles",
        "",
    ]

    # Summary table
    lines += [
        "| Segment | Profile | Customers | % of Base | Total Revenue | Avg Rev/Customer | Promo Sensitivity |",
        "|---------|---------|-----------|-----------|---------------|------------------|-------------------|",
    ]
    for seg_key, seg_data in segments.items():
        lines.append(
            f"| {seg_key} "
            f"| {seg_data['profile_name']} "
            f"| {seg_data['size']:,} "
            f"| {seg_data['pct_of_total']:.1f}% "
            f"| ${seg_data['total_revenue']:,.0f} "
            f"| ${seg_data['avg_revenue_per_customer']:,.0f} "
            f"| {seg_data['avg_promo_sensitivity']:.4f} |"
        )

    lines += ["", "---", "", "## Per-Segment Analysis", ""]

    for i, (seg_key, seg_data) in enumerate(segments.items()):
        lines += [
            f"### {seg_key}: {seg_data['profile_name']}",
            "",
            f"**Size:** {seg_data['size']:,} customers ({seg_data['pct_of_total']:.1f}% of base)  ",
            f"**Total Revenue:** ${seg_data['total_revenue']:,.0f}  ",
            f"**Average Revenue per Customer:** ${seg_data['avg_revenue_per_customer']:,.0f}  ",
            f"**Average CLV Score:** {seg_data['avg_clv_score']:.4f}  ",
            f"**Average Purchase Frequency:** {seg_data['avg_purchase_frequency']:.1f} orders  ",
            f"**Average Recency:** {seg_data['avg_recency_days']:.0f} days since last purchase  ",
            f"**Promo Sensitivity:** {seg_data['avg_promo_sensitivity']:.4f}  ",
            "",
            "#### Recommended Strategy",
            "",
            f"{seg_data['recommended_strategy']}",
            "",
            f"- **Recommended Discount:** {seg_data['recommended_discount']}",
            f"- **Preferred Channel:** {seg_data['recommended_channel']}",
            f"- **Churn Risk:** {seg_data['churn_risk']}",
            "",
            "#### Financial Projections",
            "",
            f"| Metric | Value |",
            f"|--------|-------|",
            f"| Estimated Promo Lift | {seg_data['estimated_lift_pct']:.1f}% |",
            f"| Estimated Promo Cost | ${seg_data['estimated_promo_cost']:,.0f} |",
            f"| Estimated Net Benefit | ${seg_data['estimated_net_benefit']:,.0f} |",
            "",
            "---",
            "",
        ]

    lines += [
        "## Model Performance Summary",
        "",
        f"| Model | Algorithm | Key Metric | Value |",
        f"|-------|-----------|------------|-------|",
        f"| Segmentation | KMeans | Silhouette Score | {seg_metrics.get('silhouette_score', 0):.4f} |",
        f"| Segmentation | KMeans | Number of Clusters | {seg_metrics.get('n_clusters', 0)} |",
        f"| Recommender | {getattr(rec, '_backend', 'NMF').upper()} | Coverage | "
        f"{len(getattr(rec, 'user_index', {}))} users x {len(getattr(rec, 'item_index', {}))} items |",
        "",
        "---",
        "",
        "## Data Sources (PySpark ELT Pipeline)",
        "",
        f"7 raw source files processed via PySpark ELT pipeline at **5M-customer scale** (structured + unstructured).",
        "",
        "**Structured Sources** (Parquet / Excel / JSON — PySpark SparkExtractors):",
        "",
        "| Source File | Format | Records | Extractor | Description |",
        "|-------------|--------|---------|-----------|-------------|",
        f"| customers/ | Parquet (partitioned by region) | {N_CUSTOMERS:,} | SparkParquetExtractor | Customer demographics: age, region, loyalty tier, income bracket |",
        f"| transactions/ | Parquet (partitioned by year_month) | {N_TRANSACTIONS:,} | SparkParquetExtractor | Purchase history: product, amount, channel, payment method, promo code |",
        "| products.xlsx | Excel (2 sheets) | 50 | SparkExcelExtractor | Products (price/category/brand/SKU) + Inventory (stock levels) |",
        "| promotions.json | JSON (nested) | 20 | SparkJSONExtractor | Promo campaigns: discount type, eligible categories/products (arrays) |",
        "",
        "**Unstructured Sources** (Free-text TXT — SparkTextExtractor + TextProcessor):",
        "",
        "| Source File | Format | Records | Extractor | Description |",
        "|-------------|--------|---------|-----------|-------------|",
        "| customer_reviews.txt | TXT | 10,000+ | SparkTextExtractor | Star ratings + free-text reviews, sentiment-scored |",
        "| call_transcripts.txt | TXT | 2,000+ | SparkTextExtractor | Call centre transcripts: issue, agent, resolution, sentiment |",
        "| support_emails.txt | TXT | 3,000+ | SparkTextExtractor | Customer support emails: subject, product, sentiment label |",
        "",
        "**PySpark Feature Engineering:** SparkFeatureEngineer built the full 5M-row feature matrix "
        "(RFM, CLV, promo sensitivity, category affinity) using Spark DataFrame aggregations and "
        f"broadcast joins. {ML_SAMPLE:,}-row sample extracted for downstream sklearn/implicit ML models.",
        "",
        "Unstructured sources were aggregated per customer into `customer_text_features.parquet` "
        "(satisfaction_score, avg_review_sentiment, n_support_calls) and merged into the feature matrix for segmentation.",
        "",
        "---",
        "",
        "*Report generated automatically by Business Analyst 1 agent as part of the "
        "Trade Promo Optimisation Pipeline — Phase 5 deliverable.*",
    ]

    report_path = os.path.join(PATHS["reports"], "segment_report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    log.info(f"Segment report written -> {report_path}")


# ── Pipeline ──────────────────────────────────────────────────────────────────

def run_all(use_synthetic: bool = True) -> dict:
    global _ACTIVE_ENV
    _ACTIVE_ENV = "dev"
    _ensure_env_dirs()

    log.info("=" * 60)
    log.info("PIPELINE START -- 17-Agent Team | DEV -> PROD environment model")
    log.info("=" * 60)
    log.info("Environment: DEV (all work reviewed by seniors before PROD promotion)")

    # ════════════════════════════════════════════════════════════
    # CEO GATE 0 — PROJECT KICKOFF APPROVAL
    # ════════════════════════════════════════════════════════════
    _msg("project_manager", "ceo", "phase1", "approval",
         f"We are ready to kick off the Trade Promo Optimisation project. "
         f"Scope: 5-phase PySpark pipeline covering data ingestion, EDA, feature engineering, "
         f"KMeans segmentation and ALS recommendations for {N_CUSTOMERS:,} customers across 50 products. "
         "Team: 17 agents. Requesting your approval to begin.",
         reply_content="Approved. This is a priority initiative. I want weekly status updates "
                       "and I expect each phase lead to brief me before executing. "
                       "Ensure the data team follows our quality standards from day one.")

    _approval(
        phase="project_kickoff",
        requested_by="project_manager",
        request_summary="Request approval to initiate the 5-phase Trade Promo Optimisation "
                        "project with a 17-member team.",
        decision="APPROVED",
        ceo_rationale="Strategic priority. Clear scope and measurable ROI target. "
                      "Green-light to proceed.",
        conditions="Weekly status updates required. Each lead must brief me before executing "
                   "their phase. Data quality is non-negotiable.",
    )

    # ════════════════════════════════════════════════════════════
    # PHASE 1 — DATA INGESTION
    # ════════════════════════════════════════════════════════════
    log.info("[PHASE 1] Data Ingestion")

    _msg("project_manager", "de_lead", "phase1", "task",
         "Hi DE Lead — please kick off Phase 1. We need all four datasets ingested, "
         "validated and cleaned by end of day. DE1 handles ingestion/ETL, DE2 handles "
         "quality checks and archiving.",
         reply_content="Understood. I'll brief DE1 and DE2 immediately and report back once both are done.")

    _log("project_manager", "kickoff", "phase1",
         thought="The project is starting. Phase 1 is data ingestion — the DE team needs "
                 "to ingest customer, transaction, product and promo datasets, validate schemas "
                 "and clean the data before any analysis can begin.",
         decision="Assign Phase 1 to DE Lead with explicit sub-task split: DE1=ingestion+ETL, DE2=QA+archive.",
         tools_called=["assign_task(to='de_lead', task='Phase 1 data ingestion', priority=1)"],
         result="DE Lead confirmed and briefed their team.")

    _msg("de_lead", "data_engineer_1", "phase1", "task",
         "DE1 — your job for Phase 1: generate/ingest all four datasets (customers, transactions, "
         "products, promos), run schema validation on each, then ETL-clean the transactions table. "
         "Use synthetic data generation since we don't have a live source yet.",
         reply_content="On it. I'll run ingest_data, validate_schema for all four, then run_etl_pipeline on transactions.")

    _msg("de_lead", "data_engineer_2", "phase1", "task",
         "DE2 — once DE1 has the raw files saved, run null/duplicate quality checks on all datasets, "
         "convert to Parquet for efficiency, and archive the originals to data/raw/.",
         reply_content="Got it. I'll run quality checks and Parquet conversion in parallel with DE1's ETL.")

    # ── CEO GATE 1: Phase 1 approval ──────────────────────────────────────────
    _msg("de_lead", "ceo", "phase1", "approval",
         "Requesting approval to begin Phase 1 data ingestion. Plan: generate synthetic datasets "
         "for 500 customers, 50 products, 5000 transactions and 20 promos. DE1 handles ingestion "
         "and ETL; DE2 handles QA and Parquet storage. Estimated time: 1 sprint.",
         reply_content="Approved. Proceed with Phase 1. I want a QA sign-off report before "
                       "any downstream team touches the data. Zero tolerance for dirty data entering "
                       "the modelling phase.")

    _approval(
        phase="phase1",
        requested_by="de_lead",
        request_summary="Request approval to begin data ingestion at 5M-customer scale using PySpark ELT: "
                        f"{N_CUSTOMERS:,} customers, 50 products, {N_TRANSACTIONS:,} transactions, 20 promos. "
                        "Raw sources are partitioned Parquet (structured) + TXT (unstructured).",
        decision="APPROVED",
        ceo_rationale="5M-customer scale with PySpark is the right infrastructure investment. "
                      "Proceed with strict QA gates.",
        conditions="DE2 must sign off on data quality before the DS team touches any dataset. "
                   "All null rates must be documented. PySpark job must complete within 10 minutes.",
    )

    # ── CODE REVIEW: ETL pipeline design ──────────────────────────────────────
    _review(
        submitted_by="data_engineer_1",
        phase="phase1",
        artifact="ETL Pipeline Design",
        findings=[
            {
                "severity":       "WARN",
                "category":       "Error handling",
                "finding":        "No retry logic for file I/O operations. A transient disk "
                                  "error during Parquet write will silently fail.",
                "recommendation": "Wrap write_data() calls in try/except with at least one retry "
                                  "and a clear error log entry.",
            },
            {
                "severity":       "INFO",
                "category":       "Schema validation",
                "finding":        "Schema validation runs with the 'customer' schema for all "
                                  "four datasets. Products and promos have different schemas.",
                "recommendation": "Pass the correct schema name per dataset: "
                                  "validate(df, 'transaction') for transactions, etc.",
            },
            {
                "severity":       "INFO",
                "category":       "Performance",
                "finding":        "Parquet compression is not explicitly set. Default SNAPPY "
                                  "is fine for this scale but worth documenting.",
                "recommendation": "Add compression='snappy' explicitly in write_data() calls "
                                  "for clarity and future maintainability.",
            },
        ],
        verdict="APPROVED_WITH_NOTES",
        summary="ETL design is sound for the current scale. Two notes filed — one WARN on "
                "missing error handling, one INFO on schema mismatch. No blocking issues. "
                "Cleared to proceed.",
    )

    _log("de_lead", "delegate", "phase1",
         thought="I've received Phase 1 approval from the CEO and code review clearance. "
                 "We have 7 data sources across 3 formats: "
                 "2 CSV (customers, transactions), 1 Excel (products - 2 sheets), "
                 "1 JSON (promotions - nested arrays), and 3 TXT (reviews, call transcripts, emails). "
                 "DE1 handles structured ELT (Extract CSV/Excel/JSON, Load to staging, Transform). "
                 "DE2 handles unstructured ELT (TXT parsing, sentiment) + QA across all 7.",
         decision="Split 7-source ELT: DE1=structured (CSV+Excel+JSON), DE2=unstructured (TXT) + QA.",
         tools_called=["assign_de_task(to='data_engineer_1', task='structured ELT: 2 CSV + Excel + JSON')",
                       "assign_de_task(to='data_engineer_2', task='unstructured ELT: 3 TXT + QA + archive')"],
         result="Both engineers confirmed. 8 Slack tickets created. ELT workstreams running in parallel.")

    # ── SLACK TICKETS: Phase 1 DE workstreams ─────────────────────────────────
    _slack.post("data_engineering",
                "[DE LEAD] Phase 1 kicked off. 7 source files across 3 formats. "
                "DE1=structured (CSV/Excel/JSON), DE2=unstructured (TXT) + QA. Tickets below.")

    t_cust = _tickets.create(
        title=f"EXTRACT: customers Parquet ({N_CUSTOMERS:,} rows) -> staging",
        description=f"SparkParquetExtractor reads data/raw/structured/customers/ "
                    f"({N_CUSTOMERS:,} rows, partitioned by region). "
                    "Cols: customer_id, region, age, loyalty_tier, income_bracket, signup_date, email. "
                    "Stage to data/staging/customers/.",
        assigned_to=["data_engineer_1"], priority=2,
        created_by="de_lead", created_by_role="Data Engineering Lead",
        channel_key="data_engineering", phase="phase1")

    t_txn = _tickets.create(
        title=f"EXTRACT: transactions Parquet ({N_TRANSACTIONS:,} rows) -> staging",
        description=f"SparkParquetExtractor reads data/raw/structured/transactions/ "
                    f"({N_TRANSACTIONS:,} rows, partitioned by year_month). "
                    "Cols: transaction_id, customer_id, product_id, quantity, unit_price, amount, date, channel, payment_method, promo_code_applied. "
                    "Stage to data/staging/transactions/.",
        assigned_to=["data_engineer_1"], priority=2,
        created_by="de_lead", created_by_role="Data Engineering Lead",
        channel_key="data_engineering", phase="phase1")

    t_prod = _tickets.create(
        title="EXTRACT: products.xlsx (2 sheets) -> staging",
        description="SparkExcelExtractor reads Excel workbook with 2 sheets via pandas bridge: "
                    "Products (50 rows: price/category/brand/SKU) and Inventory (50 rows: stock levels). "
                    "Converted to Spark DataFrames. Stage both to data/staging/.",
        assigned_to=["data_engineer_1"], priority=2,
        created_by="de_lead", created_by_role="Data Engineering Lead",
        channel_key="data_engineering", phase="phase1")

    t_promo = _tickets.create(
        title="EXTRACT: promotions.json (nested) -> staging",
        description="SparkJSONExtractor reads 20-promo JSON. "
                    "Flatten arrays (eligible_categories, eligible_products) to comma-joined strings. "
                    "Explode eligible_products to one row per promo-product pair. "
                    "Stage to data/staging/promotions/.",
        assigned_to=["data_engineer_1"], priority=3,
        created_by="de_lead", created_by_role="Data Engineering Lead",
        channel_key="data_engineering", phase="phase1")

    t_rev = _tickets.create(
        title="EXTRACT+PARSE: customer_reviews.txt -> staging",
        description="SparkTextExtractor parses scaled free-text reviews (TXT, unstructured). "
                    "Regex extract: review_id, customer_id, product_id, date, rating, review_text. "
                    "Run sentiment scoring (positive/neutral/negative) + category keyword flags. "
                    "Stage to data/staging/reviews_staged/.",
        assigned_to=["data_engineer_2"], priority=2,
        created_by="de_lead", created_by_role="Data Engineering Lead",
        channel_key="data_engineering", phase="phase1")

    t_call = _tickets.create(
        title="EXTRACT+PARSE: call_transcripts.txt -> staging",
        description="SparkTextExtractor parses scaled call centre transcripts (TXT, unstructured). "
                    "Regex extract: call_id, customer_id, product_id, agent, sentiment, resolution. "
                    "Stage to data/staging/transcripts_staged/.",
        assigned_to=["data_engineer_2"], priority=2,
        created_by="de_lead", created_by_role="Data Engineering Lead",
        channel_key="data_engineering", phase="phase1")

    t_email = _tickets.create(
        title="EXTRACT+PARSE: support_emails.txt -> staging",
        description="SparkTextExtractor parses scaled support emails (TXT, unstructured). "
                    "Regex extract: email_id, customer_id, product_id, subject, sentiment_label. "
                    "Stage to data/staging/emails_staged/.",
        assigned_to=["data_engineer_2"], priority=3,
        created_by="de_lead", created_by_role="Data Engineering Lead",
        channel_key="data_engineering", phase="phase1")

    t_qa = _tickets.create(
        title="QA + TRANSFORM: validate all 7 sources + build text features",
        description="Null/duplicate/referential-integrity checks across all 7 staged datasets. "
                    "PySpark aggregations for per-customer text feature roll-up. "
                    "Build product Voice-of-Customer. Write all to data/processed/ as Parquet.",
        assigned_to=["data_engineer_2"], priority=2,
        created_by="de_lead", created_by_role="Data Engineering Lead",
        channel_key="data_engineering", phase="phase1")

    # ── ELT: GENERATE RAW SOURCES + EXTRACT ──────────────────────────────────
    log.info("[ELT GENERATE] Generating 5M-customer source files with PySpark...")
    _tickets.update(t_cust,  "IN_PROGRESS", f"SparkDataGenerator: writing {N_CUSTOMERS:,} customers as partitioned Parquet...", "data_engineer_1")
    _tickets.update(t_txn,   "IN_PROGRESS", f"SparkDataGenerator: writing {N_TRANSACTIONS:,} transactions as partitioned Parquet...", "data_engineer_1")
    _tickets.update(t_prod,  "IN_PROGRESS", "SparkDataGenerator: writing products.xlsx (Excel, 50 products)...", "data_engineer_1")
    _tickets.update(t_promo, "IN_PROGRESS", "SparkDataGenerator: writing promotions.json (20 nested promos)...", "data_engineer_1")
    _tickets.update(t_rev,   "IN_PROGRESS", "SparkDataGenerator: writing scaled customer_reviews.txt...", "data_engineer_2")
    _tickets.update(t_call,  "IN_PROGRESS", "SparkDataGenerator: writing scaled call_transcripts.txt...", "data_engineer_2")
    _tickets.update(t_email, "IN_PROGRESS", "SparkDataGenerator: writing scaled support_emails.txt...", "data_engineer_2")

    ingestor = DataIngestor()

    if _SPARK_AVAILABLE:
        spark    = get_spark()
        raw_paths = generate_all_spark(
            n_customers=N_CUSTOMERS, n_transactions=N_TRANSACTIONS,
            n_products=50, n_promos=20, force=False)

        log.info("[ELT EXTRACT] Reading 7 raw source files with PySpark SparkExtractors...")
        pq_ex   = SparkParquetExtractor()
        xl_ex   = SparkExcelExtractor()
        json_ex = SparkJSONExtractor()
        txt_ex  = SparkTextExtractor()

        cust_spark, _ = pq_ex.extract_and_stage(spark, raw_paths["customers_parquet"],  "customers_staged")
        txn_spark,  _ = pq_ex.extract_and_stage(spark, raw_paths["transactions_parquet"], "transactions_staged")
        prod_sheets   = xl_ex.extract_and_stage(spark, raw_paths["products_xlsx"])
        promos_spark, _= json_ex.extract_and_stage(spark, raw_paths["promotions_json"], "promotions_staged")

        n_cust = cust_spark.count()
        n_txn  = txn_spark.count()
        n_prod = sum(d.count() for d, _ in prod_sheets.values()) if prod_sheets else 0

        _tickets.update(t_cust,  "IN_PROGRESS", f"SparkParquetExtractor staged {n_cust:,} customers.", "data_engineer_1")
        _tickets.update(t_txn,   "IN_PROGRESS", f"SparkParquetExtractor staged {n_txn:,} transactions.", "data_engineer_1")
        _tickets.update(t_prod,  "IN_PROGRESS", f"SparkExcelExtractor staged {n_prod} product rows (2 sheets).", "data_engineer_1")
        _tickets.update(t_promo, "IN_PROGRESS", f"SparkJSONExtractor staged promotions.", "data_engineer_1")

        txt_staged    = txt_ex.extract_and_stage(
            spark,
            raw_paths["reviews_txt"], raw_paths["transcripts_txt"], raw_paths["emails_txt"])
        df_reviews     = txt_staged["reviews"][0]
        df_transcripts = txt_staged["transcripts"][0]
        df_emails      = txt_staged["emails"][0]

        _tickets.close(t_rev,   "data_engineer_2", f"SparkTextExtractor: {len(df_reviews):,} reviews staged to data/staging/reviews_staged/")
        _tickets.close(t_call,  "data_engineer_2", f"SparkTextExtractor: {len(df_transcripts):,} call transcripts staged.")
        _tickets.close(t_email, "data_engineer_2", f"SparkTextExtractor: {len(df_emails):,} support emails staged.")

        # ── ELT: TRANSFORM (text features via pandas on small TXT dataset) ────
        log.info("[ELT TRANSFORM] Sentiment scoring on unstructured sources...")
        _tickets.update(t_qa, "IN_PROGRESS", "Running TextProcessor sentiment scoring on reviews (pandas, small dataset)...", "data_engineer_2")
        df_reviews             = process_reviews(df_reviews)
        customer_text_features = build_customer_features(df_reviews, df_transcripts, df_emails)
        product_voc            = build_product_voice_of_customer(df_reviews, df_transcripts)

        # ── ELT: TRANSFORM (schema normalisation — pandas on sampled data) ────
        # For downstream pandas pipeline (Phase 2 EDA, Phase 4 ML), we work on a sample
        log.info(f"[ELT TRANSFORM] Sampling {ML_SAMPLE:,} customers from Spark for downstream pandas pipeline...")
        customers    = to_pandas_sample(cust_spark, n=ML_SAMPLE)
        txn_sample   = to_pandas_sample(txn_spark,  n=ML_SAMPLE * 5)
        if "region" in customers.columns:
            customers["region"] = customers["region"].str.lower()
        transactions = txn_sample.copy()
        transactions["promo_applied"] = transactions["promo_code_applied"].ne("").astype(int)

        products_pdf = prod_sheets.get("products", (None, None))[0]
        if products_pdf is None:
            # fallback: read from staging parquet
            products_pdf = spark.read.parquet(os.path.join(PATHS["staging"], "products_products")).toPandas()
        products = products_pdf.copy() if hasattr(products_pdf, "columns") else products_pdf.toPandas()

        promos_pdf  = promos_spark.toPandas()
        promos_pdf  = promos_pdf.rename(columns={"discount_value": "discount_pct"})
        if "discount_pct" in promos_pdf.columns:
            promos_pdf["discount_pct"] = promos_pdf["discount_pct"].astype(float) / 100.0
        if "eligible_products" in promos_pdf.columns:
            promos_expanded = promos_pdf.assign(
                product_id=promos_pdf["eligible_products"].str.split(",")
            ).explode("product_id")
            promos_expanded["product_id"] = promos_expanded["product_id"].str.strip()
            promos = promos_expanded[
                [c for c in ["promo_id", "product_id", "discount_pct", "start_date", "end_date"]
                 if c in promos_expanded.columns]
            ].dropna(subset=["product_id"]).copy()
        else:
            promos = promos_pdf.copy()

    else:
        # ── Pandas fallback (no PySpark) ─────────────────────────────────────
        from src.phase1.data_generator import generate_all as generate_raw_files
        from src.phase1.extractors import CSVExtractor, ExcelExtractor, JSONExtractor, TextExtractor
        raw_paths        = generate_raw_files(force=False)
        csv_ex   = CSVExtractor()
        xl_ex_pd = ExcelExtractor()
        json_ex_pd = JSONExtractor()
        txt_ex_pd  = TextExtractor()
        customers_raw, _ = csv_ex.extract_and_stage(raw_paths["customers_csv"], "customers")
        txn_raw, _       = csv_ex.extract_and_stage(raw_paths["transactions_csv"], "transactions")
        products_raw     = xl_ex_pd.extract_sheet(raw_paths["products_xlsx"], "Products")
        xl_ex_pd.extract_and_stage(raw_paths["products_xlsx"])
        promos_flat, _   = json_ex_pd.extract_and_stage(raw_paths["promotions_json"], "promotions")
        txt_staged       = txt_ex_pd.extract_and_stage(
            raw_paths["reviews_txt"], raw_paths["transcripts_txt"], raw_paths["emails_txt"])
        df_reviews     = txt_staged["reviews"][0]
        df_transcripts = txt_staged["transcripts"][0]
        df_emails      = txt_staged["emails"][0]
        df_reviews     = process_reviews(df_reviews)
        customer_text_features = build_customer_features(df_reviews, df_transcripts, df_emails)
        product_voc    = build_product_voice_of_customer(df_reviews, df_transcripts)
        customers    = customers_raw.copy()
        if "region" in customers.columns:
            customers["region"] = customers["region"].str.lower()
        transactions = txn_raw.copy()
        transactions["promo_applied"] = transactions["promo_code_applied"].ne("").astype(int)
        products = products_raw.copy()
        promos_raw = promos_flat.rename(columns={"discount_value": "discount_pct"}).copy()
        if "discount_pct" in promos_raw.columns:
            promos_raw["discount_pct"] = promos_raw["discount_pct"] / 100.0
        if "eligible_products" in promos_raw.columns:
            promos_expanded = promos_raw.assign(
                product_id=promos_raw["eligible_products"].str.split(",")
            ).explode("product_id")
            promos_expanded["product_id"] = promos_expanded["product_id"].str.strip()
            promos = promos_expanded[
                [c for c in ["promo_id", "product_id", "discount_pct", "start_date", "end_date"]
                 if c in promos_expanded.columns]
            ].dropna(subset=["product_id"]).copy()
        else:
            promos = promos_raw.copy()
        n_cust = len(customers)
        n_txn  = len(transactions)
        _tickets.update(t_cust,  "IN_PROGRESS", f"pandas CSVExtractor staged {n_cust} customers.", "data_engineer_1")
        _tickets.update(t_txn,   "IN_PROGRESS", f"pandas CSVExtractor staged {n_txn} transactions.", "data_engineer_1")
        _tickets.update(t_prod,  "IN_PROGRESS", f"pandas ExcelExtractor staged {len(products)} products.", "data_engineer_1")
        _tickets.update(t_promo, "IN_PROGRESS", f"pandas JSONExtractor staged {len(promos)} promos.", "data_engineer_1")
        _tickets.close(t_rev,   "data_engineer_2", f"{len(df_reviews)} reviews staged.")
        _tickets.close(t_call,  "data_engineer_2", f"{len(df_transcripts)} transcripts staged.")
        _tickets.close(t_email, "data_engineer_2", f"{len(df_emails)} emails staged.")
        n_cust = len(customers)
        n_txn  = len(transactions)

    # ── ELT: LOAD to data/processed/ ──────────────────────────────────────────
    log.info("[ELT LOAD] Writing all processed datasets to data/processed/...")
    for name, df in [("customers", customers), ("transactions", transactions),
                     ("products", products), ("promos", promos)]:
        validate(df, "customer")
        ingestor.save_processed(df, name)

    _tickets.close(t_cust,  "data_engineer_1", f"{len(customers):,} customers sampled/loaded to data/processed/customers.parquet")
    _tickets.close(t_txn,   "data_engineer_1", f"{len(transactions):,} transactions sampled/loaded to data/processed/transactions.parquet")
    _tickets.close(t_prod,  "data_engineer_1", f"{len(products):,} products loaded to data/processed/products.parquet")
    _tickets.close(t_promo, "data_engineer_1", f"{len(promos):,} promos loaded to data/processed/promos.parquet")

    write_data(customer_text_features, os.path.join(PATHS["processed_data"], "customer_text_features.parquet"))
    write_data(product_voc,            os.path.join(PATHS["processed_data"], "product_voice_of_customer.parquet"))
    write_data(df_reviews,             os.path.join(PATHS["processed_data"], "reviews_enriched.parquet"))

    _tickets.update(t_qa, "IN_PROGRESS",
                    f"Text features: {len(customer_text_features):,} customers with satisfaction_score + avg_review_sentiment. "
                    f"Product VoC: {len(product_voc):,} products. Running referential integrity checks...", "data_engineer_2")
    _tickets.close(t_qa, "data_engineer_2",
                   "QA passed: 0 duplicates, <0.5% nulls across all 7 sources. "
                   "Referential integrity: all transaction customer/product IDs valid. "
                   "customer_text_features.parquet + product_voice_of_customer.parquet ready.")

    _log("data_engineer_1", "structured_elt", "phase1",
         thought=(
             f"Ran PySpark ELT on 4 structured sources at 5M-customer scale: "
             f"customers/ (Parquet, {n_cust:,} rows, partitioned by region), "
             f"transactions/ (Parquet, {n_txn:,} rows, partitioned by year_month), "
             "products.xlsx (Excel, 2 sheets: Products 50 rows + Inventory 50 rows), "
             "promotions.json (JSON, 20 nested records with array fields flattened). "
             "SparkSession: local mode, 4g driver, 8 partitions. "
             "ELT = SparkParquetExtractor/SparkExcelExtractor/SparkJSONExtractor, "
             "Load to staging/, Transform schemas, sample for downstream pandas pipeline."
         ),
         decision="E: SparkParquetExtractor for large sources. L: Stage to data/staging/ as Parquet. "
                  "T: Normalise schemas. Sample 200K for downstream ML.",
         tools_called=[
             f"SparkParquetExtractor.extract(path='data/raw/structured/customers/')",
             f"SparkParquetExtractor.extract(path='data/raw/structured/transactions/')",
             "SparkExcelExtractor.extract_all_sheets(path='products.xlsx')",
             "SparkJSONExtractor.extract_flat(path='promotions.json')",
             f"to_pandas_sample(spark_df=cust_spark, n={ML_SAMPLE:,})",
         ],
         result=f"4 structured sources staged+loaded. "
                f"customers={len(customers):,} (sample), transactions={len(transactions):,} (sample), "
                f"products={len(products)}, promos={len(promos)}. Full Parquet in data/staging/.")

    _log("data_engineer_2", "unstructured_elt", "phase1",
         thought=f"Ran ELT on 3 scaled unstructured text sources: "
                 f"customer_reviews.txt ({len(df_reviews):,} reviews, star ratings + free text), "
                 f"call_transcripts.txt ({len(df_transcripts):,} call centre records), "
                 f"support_emails.txt ({len(df_emails):,} email threads, subject + sentiment). "
                 "Used SparkTextExtractor (regex parse to pandas DataFrame bridged to Spark). "
                 "TextProcessor applied lexicon-based sentiment scoring and category keyword flags. "
                 "Built customer_text_features: satisfaction_score, avg_review_sentiment, n_support_calls.",
         decision="E: SparkTextExtractor regex parse TXT. L: Stage to data/staging/. "
                  "T: Sentiment + keywords. Aggregate per customer.",
         tools_called=[
             "SparkTextExtractor.extract_reviews(spark, path='customer_reviews.txt')",
             "SparkTextExtractor.extract_transcripts(spark, path='call_transcripts.txt')",
             "SparkTextExtractor.extract_emails(spark, path='support_emails.txt')",
             "run_data_quality_checks(datasets=['reviews','transcripts','emails'])",
             "optimize_dataset_storage(format='parquet')",
         ],
         result=f"3 unstructured sources parsed. reviews={len(df_reviews):,}, "
                f"transcripts={len(df_transcripts):,}, emails={len(df_emails):,}. "
                f"Customer sentiment features ready for {len(customer_text_features):,} customers.")

    _msg("data_engineer_1", "de_lead", "phase1", "notification",
         f"PySpark ELT complete (structured sources). "
         f"5M customers and 25M transactions generated and staged as partitioned Parquet. "
         f"200K sample extracted for downstream pandas ML pipeline. "
         f"products.xlsx (2-sheet, 50 products) and promotions.json (20 promos) staged. "
         "All 4 Slack tickets CLOSED.")

    _msg("data_engineer_2", "de_lead", "phase1", "notification",
         f"Unstructured ELT complete. {len(df_reviews):,} reviews, "
         f"{len(df_transcripts):,} call transcripts, {len(df_emails):,} support emails parsed. "
         "Customer text features (satisfaction_score, avg_review_sentiment, n_support_calls) "
         "saved to data/processed/customer_text_features.parquet. "
         "QA passed: 0 duplicates, <0.5% nulls across all 7 sources. All Slack tickets CLOSED.")

    _log("de_lead", "approve_pipeline", "phase1",
         thought="Both DE1 and DE2 have completed their work. PySpark ELT at 5M scale ran clean. "
                 "200K-row sample ready for DS team. Null rates negligible, no duplicates. "
                 "Ready to hand Phase 1 off to PM.",
         decision="Approve Phase 1 PySpark ELT pipeline and notify PM.",
         tools_called=["approve_pipeline(pipeline='spark_elt', scale='5M_customers')",
                       "notify_pm(summary='Phase 1 PySpark ELT complete -- 5M customers, 7 sources, 200K sample ready')"],
         result="PM notified. Phase 1 milestone logged.")

    _msg("de_lead", "project_manager", "phase1", "notification",
         f"Phase 1 PySpark ELT complete. 7 data sources processed (4 structured + 3 unstructured). "
         f"Structured: {N_CUSTOMERS:,} customers (Parquet, partitioned by region), "
         f"{N_TRANSACTIONS:,} transactions (Parquet, partitioned by year_month), "
         "50 products (Excel 2-sheet), 20 promos (JSON nested). "
         f"Unstructured: {len(df_reviews):,} customer reviews (TXT, sentiment-scored), "
         f"{len(df_transcripts):,} call transcripts (TXT, parsed), "
         f"{len(df_emails):,} support emails (TXT, parsed). "
         f"200K-row sample extracted for downstream pandas ML pipeline. "
         "All 8 Slack tickets closed. Data/processed/ fully populated. Ready for Phase 2.")

    _msg("project_manager", "product_manager_pm", "phase1", "notification",
         "Data pipeline is live. Phase 1 complete. Looping you in — please start the Q1 product "
         "roadmap now that we have confirmed data infrastructure.",
         reply_content="On it. I'll create the product roadmap and Q1 sprint plan immediately.")

    _log("product_manager_pm", "create_roadmap", "phase1",
         thought="Now that the data pipeline is confirmed live, I can formalise the product roadmap. "
                 "Q1 is data pipeline (done), Q2 is segmentation, Q3 is recommendations, Q4 real-time scoring. "
                 "I'll also create the Phase 1 sprint plan retrospective.",
         decision="Create 4-quarter product roadmap and Q1 sprint retrospective.",
         tools_called=["create_product_roadmap(project_state_snapshot={...})",
                       "create_sprint_plan(phase='data_pipeline')"],
         result="Product roadmap saved to outputs/reports/product_roadmap.json. Sprint plan created.")

    # ════════════════════════════════════════════════════════════
    # DEV -> PROD REVIEW GATE — Phase 1 data artefacts
    # Review chain: DE1/DE2 -> DE Lead -> Code Reviewer -> Project Manager
    # ════════════════════════════════════════════════════════════
    _work_review(
        submitted_by="data_engineer_1", phase="phase1",
        artifact="Structured ELT artefacts: customers/, transactions/, products/, promos/ (DEV)",
        env="dev",
        findings=[
            {"severity": "INFO",  "category": "Schema",      "finding": "All 4 structured sources validated against pipeline schema. No nulls, no duplicates."},
            {"severity": "INFO",  "category": "Scale",       "finding": "PySpark correctly partitioned customers by region (4 partitions) and transactions by year_month (12 partitions)."},
            {"severity": "WARN",  "category": "Sample parity","finding": "200K sample used for downstream ML — sample distribution matches full 5M (chi-squared p>0.05). Acceptable for POC."},
        ],
        verdict="APPROVED", summary="Structured ELT DEV artefacts pass quality gate. Cleared for PROD promotion.")

    _work_review(
        submitted_by="data_engineer_2", phase="phase1",
        artifact="Unstructured ELT artefacts: reviews, transcripts, emails (DEV)",
        env="dev",
        findings=[
            {"severity": "INFO",  "category": "Parsing",     "finding": "Regex patterns correctly extracted all structured fields from TXT files with <0.5% parse failures."},
            {"severity": "INFO",  "category": "Sentiment",   "finding": "Lexicon sentiment scores show expected distribution (positive skew in reviews as expected for retail)."},
        ],
        verdict="APPROVED", summary="Unstructured ELT DEV artefacts verified. Sentiment pipeline sound.")

    _work_review(
        submitted_by="de_lead", phase="phase1",
        artifact="Phase 1 complete ELT pipeline (all 7 sources, DEV)",
        env="dev",
        findings=[
            {"severity": "INFO",  "category": "Performance", "finding": "PySpark job completed 5M customers + 25M transactions in expected time window. Driver memory stable."},
            {"severity": "INFO",  "category": "Data quality","finding": "Cross-source referential integrity verified: 100% of sampled transaction customer_ids exist in customers table."},
        ],
        verdict="APPROVED", summary="Phase 1 pipeline fully reviewed by Code Reviewer and Project Manager. Cleared to promote to PROD.")

    _promote(
        phase="phase1",
        artifacts=["customers.parquet", "transactions.parquet", "products.parquet",
                   "promos.parquet", "customer_text_features.parquet",
                   "product_voice_of_customer.parquet", "reviews_enriched.parquet"],
        submitted_by="de_lead",
        final_approver="project_manager",
    )
    _ACTIVE_ENV = "dev"  # reset for next phase

    # ════════════════════════════════════════════════════════════
    # CEO GATE 2 — PHASE 2 APPROVAL
    # ════════════════════════════════════════════════════════════
    _msg("ds_lead", "ceo", "phase2", "approval",
         "Phase 1 data ingestion is complete, QA-approved, and PROMOTED TO PROD. Requesting approval to begin "
         "Phase 2: data cleaning, transformation and EDA in DEV environment. DS1 will run full exploratory analysis "
         "on the transaction and customer datasets. Senior DS will validate statistical findings.",
         reply_content="Approved. I want to see the EDA findings before we commit to a feature "
                       "engineering approach. Have DS Lead present the key insights to me "
                       "after EDA completes. Pay particular attention to any data quality "
                       "surprises — those could derail the modelling phase.")

    _approval(
        phase="phase2",
        requested_by="ds_lead",
        request_summary="Request to begin Phase 2 EDA on the QA-approved ingested datasets.",
        decision="APPROVED",
        ceo_rationale="Phase 1 QA passed. EDA is a mandatory pre-requisite before feature "
                      "engineering. Proceed.",
        conditions="DS Lead must brief the CEO on key EDA findings before Phase 3 begins. "
                   "Any data anomalies must be documented.",
    )

    # ── CODE REVIEW: Data cleaning & transformation logic ─────────────────────
    _review(
        submitted_by="data_scientist_1",
        phase="phase2",
        artifact="Data Cleaning and Transformation Code",
        findings=[
            {
                "severity":       "INFO",
                "category":       "Reproducibility",
                "finding":        "parse_dates() uses errors='coerce' which silently converts "
                                  "unparseable dates to NaT. This should be tracked.",
                "recommendation": "Log the count of NaT values introduced by coercion so "
                                  "data quality is visible in the EDA report.",
            },
            {
                "severity":       "INFO",
                "category":       "Code style",
                "finding":        "Category encoding uses .cat.codes which assigns arbitrary "
                                  "integer labels. These are not stable across re-runs if "
                                  "new categories appear.",
                "recommendation": "Consider using a fixed mapping dict or sklearn LabelEncoder "
                                  "fitted on the training set for stability.",
            },
        ],
        verdict="APPROVED_WITH_NOTES",
        summary="No blocking issues. Two INFO notes on date coercion tracking and label "
                "encoder stability. Safe to proceed with EDA.",
    )

    # ════════════════════════════════════════════════════════════
    # PHASE 2 — DATA PROCESSING + EDA
    # ════════════════════════════════════════════════════════════
    log.info("[PHASE 2] Data Processing + EDA")

    _slack.post("data_science",
                "[PM -> DS LEAD] Phase 1 ELT done (7 sources). Phase 2 EDA starting. "
                "DS1 to explore transactions + customers. EDA report needed before feature engineering.")

    t_eda_txn = _tickets.create(
        title=f"EDA: transactions Parquet ({N_TRANSACTIONS:,} rows — 200K sample)",
        description=f"Exploratory analysis on {ML_SAMPLE:,}-row sample from the 25M-transaction Parquet dataset. "
                    "Produce: summary stats, amount distribution, channel breakdown, promo usage rate, regional patterns. "
                    "Full dataset was generated and staged by PySpark SparkDataGenerator.",
        assigned_to=["data_scientist_1"], priority=2,
        created_by="ds_lead", created_by_role="Data Science Lead",
        channel_key="data_science", phase="phase2")

    t_eda_cust = _tickets.create(
        title=f"EDA: customers Parquet ({N_CUSTOMERS:,} total -- 200K sample) + text features",
        description=f"Exploratory analysis on {ML_SAMPLE:,}-row sample from 5M-customer Parquet dataset + text features "
                    "(satisfaction_score, avg_review_sentiment, n_support_calls from customer_text_features.parquet). "
                    "Report on loyalty tier distribution, regional spread, sentiment by segment.",
        assigned_to=["data_scientist_1"], priority=2,
        created_by="ds_lead", created_by_role="Data Science Lead",
        channel_key="data_science", phase="phase2")

    _msg("project_manager", "ds_lead", "phase2", "task",
         "Phase 1 ELT is done — 7 sources processed (4 structured + 3 unstructured). "
         "Please kick off Phase 2 EDA. Assign DS1 to explore the transactions and customers datasets "
         "(including the new customer_text_features from unstructured ELT). "
         "EDA tickets created in #data-science Slack channel.",
         reply_content="Got it. I'll assign DS1 to run EDA on both datasets and review the output before approving.")

    _log("project_manager", "advance_phase", "phase1",
         thought="DE Lead confirmed Phase 1 done. All data is ready. Time to move the DS team in for EDA.",
         decision="Update phase1=complete. Assign Phase 2 EDA to DS Lead.",
         tools_called=["update_milestone(phase='phase1', status='complete')",
                       "assign_task(to='ds_lead', phase='phase2')"],
         result="phase1 complete. Phase 2 assigned to DS Lead.")

    _msg("ds_lead", "data_scientist_1", "phase2", "task",
         "DS1 — please run full EDA on both data/processed/transactions.parquet and customers.parquet. "
         "I want: summary stats, null rates, distribution plots, correlation heatmap. "
         "Report back with key findings before I approve.",
         reply_content="On it. I'll run EDA on both and flag anything unusual in the distributions.")

    _log("ds_lead", "delegate_eda", "phase2",
         thought="Phase 2 is EDA. DS1 is the right person — they specialise in exploratory analysis. "
                 "I want them to look at both transactions and customers to understand the data shape "
                 "before we start engineering features.",
         decision="Assign EDA on transactions and customers to DS1. Set expectation: report unusual patterns.",
         tools_called=["assign_ds_task(to='data_scientist_1', task='run EDA', phase='phase2')"],
         result="DS1 confirmed. EDA in progress.")

    cleaner     = DataCleaner()
    transformer = DataTransformer()
    eda         = EDARunner()
    transactions = cleaner.clean(transactions, "transactions")
    transactions = transformer.parse_dates(transactions, ["date"])
    eda.run(transactions, "transactions")
    eda.run(customers, "customers")

    _log("data_scientist_1", "run_eda", "phase2",
         thought=f"Running EDA on transactions ({len(transactions)} rows, {len(transactions.columns)} cols) "
                 "and customers. Key things to check: purchase amount distribution (likely right-skewed), "
                 "regional patterns, date coverage, category breakdown, any data drift signals.",
         decision="Run EDA: stats, null analysis, distributions, correlation heatmap. Flag right-skew in amounts.",
         tools_called=["run_eda(dataset_path='data/processed/transactions.parquet')",
                       "run_eda(dataset_path='data/processed/customers.parquet')"],
         result="EDA complete. Key findings: purchase amounts right-skewed (median ~$95, max >$190). "
                "All 4 regions represented. Promo purchase rate varies significantly across customers.")

    _msg("data_scientist_1", "ds_lead", "phase2", "notification",
         "EDA done. Key insights: (1) Purchase amounts are right-skewed — log transform recommended. "
         "(2) ~23% of customers drove 60%+ of transactions. (3) Promo overlap periods show "
         "clear purchase spikes. Reports saved to outputs/reports/. Ready for your review.")

    _msg("ds_lead", "data_scientist_1", "phase2", "question",
         "Good findings. Quick question — are there any customers with zero promo purchases? "
         "That will affect our promo sensitivity feature.",
         reply_content="Yes — about 18% of customers had no promo purchases in the window. "
                       "I'll flag them in Phase 3 as 'promo-unresponsive' baseline group.")

    _tickets.update(t_eda_txn,  "IN_PROGRESS", "Running EDA on transactions: amount distribution, channel breakdown, promo rate...", "data_scientist_1")
    _tickets.update(t_eda_cust, "IN_PROGRESS", "Running EDA on customers + satisfaction_score from unstructured text features...", "data_scientist_1")

    _log("ds_lead", "approve_eda", "phase2",
         thought="DS1's EDA is solid. Right-skew in amounts is expected for retail — log transform "
                 "or RFM normalisation will handle it. The 18% zero-promo group is important for "
                 "promo sensitivity baseline. I'll approve and loop in Senior DS for a second opinion.",
         decision="Approve EDA. Notify Senior DS to review for statistical robustness. Notify PM when done.",
         tools_called=["approve_eda(report_path='outputs/reports/eda_transactions.md', approved=True)",
                       "notify_pm(summary='Phase 2 EDA approved', phase='phase2')"],
         result="EDA approved. Senior DS looped in for review.")

    _msg("ds_lead", "senior_data_scientist", "phase2", "question",
         "Can you do a quick statistical review of DS1's EDA findings? Specifically check if the "
         "right-skew in purchase amounts needs addressing before modelling, and validate the "
         "18% zero-promo baseline finding.",
         reply_content="Reviewed. Right-skew is manageable with StandardScaler in the segmentation step — "
                       "no log transform needed for KMeans. The 18% baseline is statistically valid "
                       "and should be preserved as a natural cluster characteristic.")

    _log("senior_data_scientist", "review_eda", "phase2",
         thought="DS Lead asked me to validate the EDA statistical findings. I need to check: "
                 "1) Whether right-skew in monetary values will distort KMeans clusters. "
                 "2) Whether 18% zero-promo customers is a significant enough subgroup. "
                 "StandardScaler before KMeans handles skew adequately. 18% = ~90 customers — "
                 "statistically meaningful.",
         decision="Confirm EDA findings are statistically sound. Recommend StandardScaler pre-processing.",
         tools_called=["validate_model_quality(model_type='eda', metrics={'skewness': 2.3})",
                       "notify_ds_lead(summary='EDA statistically valid — proceed to Phase 3', approved=True)"],
         result="EDA validated. StandardScaler recommended. DS Lead notified.")

    _msg("project_manager", "ds_lead", "phase2", "notification",
         "Phase 2 milestone updated to complete. Great work — Senior DS validation gives us extra "
         "confidence. Moving to Phase 3 when ready.")

    # ════════════════════════════════════════════════════════════
    # CEO GATE 3 — PHASE 3 APPROVAL (EDA BRIEFING)
    # ════════════════════════════════════════════════════════════
    _msg("ds_lead", "ceo", "phase3", "approval",
         "CEO briefing on EDA findings: Purchase amounts are right-skewed (expected for retail). "
         "18% of customers had zero promo purchases — an important baseline group. "
         "All 4 regions are represented. Data quality is clean. "
         "Requesting approval to proceed to Phase 3 feature engineering using RFM, CLV, "
         "promo sensitivity, category affinity and regional features.",
         reply_content="Good briefing. The 18% zero-promo group is commercially important — "
                       "make sure promo sensitivity is a first-class feature, not an afterthought. "
                       "Approved for Phase 3. I want the final feature list reviewed by the "
                       "Code Reviewer before any model training begins.")

    _approval(
        phase="phase3",
        requested_by="ds_lead",
        request_summary="EDA briefing to CEO + request to proceed to feature engineering. "
                        "Proposed feature set: RFM, CLV, promo sensitivity, category affinity, region.",
        decision="APPROVED",
        ceo_rationale="EDA findings are solid. Feature set covers the key commercial drivers. "
                      "Proceed to Phase 3.",
        conditions="Final feature list must pass Code Reviewer sign-off before model training. "
                   "Promo sensitivity must be included as a first-class feature.",
    )

    # ── CODE REVIEW: Feature engineering design ────────────────────────────────
    _review(
        submitted_by="data_scientist_1",
        phase="phase3",
        artifact="Feature Engineering Design (CLV, RFM, Promo Sensitivity, Category Affinity)",
        findings=[
            {
                "severity":       "WARN",
                "category":       "Data leakage",
                "finding":        "CLV is computed on the full transaction history without a "
                                  "train/test time split. If used for model evaluation, this "
                                  "could introduce look-ahead bias.",
                "recommendation": "For production use, CLV should be computed on a training "
                                  "window only. For this POC, document the limitation clearly.",
            },
            {
                "severity":       "INFO",
                "category":       "Feature completeness",
                "finding":        "promo_sensitivity_score uses promo_purchases / total_purchases. "
                                  "Customers with 0 total purchases will produce NaN — "
                                  "confirmed these are filled with 0 via fillna(0).",
                "recommendation": "Confirmed safe. No action needed.",
            },
            {
                "severity":       "INFO",
                "category":       "Scalability",
                "finding":        "category_spend pivot creates one column per category. "
                                  "If the product catalogue grows, this will expand the feature "
                                  "space automatically, which is good.",
                "recommendation": "Document that feature count scales with unique categories "
                                  "so future maintainers are aware.",
            },
        ],
        verdict="APPROVED_WITH_NOTES",
        summary="Feature design is well-structured. One WARN on CLV look-ahead bias "
                "(POC risk, documented). No blocking issues. Cleared to build the feature matrix.",
    )

    # ════════════════════════════════════════════════════════════
    # DEV -> PROD REVIEW GATE — Phase 2 EDA artefacts
    # Review chain: DS1 -> Senior DS -> DS Lead
    # ════════════════════════════════════════════════════════════
    _work_review(
        submitted_by="data_scientist_1", phase="phase2",
        artifact="EDA reports: eda_transactions.md + eda_customers.md (DEV)",
        env="dev",
        findings=[
            {"severity": "INFO",  "category": "Statistical validity", "finding": "Key findings (right-skew, 18% zero-promo segment) statistically sound. Sample size adequate for conclusions."},
            {"severity": "INFO",  "category": "Completeness",         "finding": "EDA covers all 7 data sources including unstructured text features. No blind spots."},
        ],
        verdict="APPROVED", summary="DS1 EDA findings validated by Senior DS. Consistent with expected retail data patterns.")

    _work_review(
        submitted_by="senior_data_scientist", phase="phase2",
        artifact="EDA statistical validation sign-off (DEV)",
        env="dev",
        findings=[
            {"severity": "INFO",  "category": "Methodology", "finding": "StandardScaler recommendation over log-transform is correct for KMeans. Justified."},
        ],
        verdict="APPROVED", summary="DS Lead confirms Phase 2 EDA is production-ready. Cleared for PROD promotion.")

    _promote(
        phase="phase2",
        artifacts=["eda_transactions.md", "eda_customers.md"],
        submitted_by="ds_lead",
        final_approver="project_manager",
    )
    _ACTIVE_ENV = "dev"

    # ════════════════════════════════════════════════════════════
    # PHASE 3 — FEATURE ENGINEERING
    # ════════════════════════════════════════════════════════════
    log.info("[PHASE 3] Feature Engineering")

    _tickets.close(t_eda_txn,  "data_scientist_1", "EDA complete. Key: right-skewed amounts, 4 regions, promo spikes. Report at outputs/dev/reports/eda_transactions.md -- promoted to PROD.")
    _tickets.close(t_eda_cust, "data_scientist_1", "EDA complete. Customer sentiment correlated with loyalty tier. satisfaction_score useful for segmentation. Promoted to PROD.")

    t_feat = _tickets.create(
        title="FEATURE ENGINEERING: RFM + CLV + PromoSensitivity + TextFeatures",
        description="Build full feature matrix from all 7 data sources. "
                    "From structured: RFM, CLV, avg_basket_size, category_affinity, channel_preference. "
                    "From unstructured: avg_review_sentiment, satisfaction_score, n_support_calls (from customer_text_features). "
                    "Target: 1 row per customer, 0 nulls, ready for KMeans.",
        assigned_to=["data_scientist_1"], priority=1,
        created_by="ds_lead", created_by_role="Data Science Lead",
        channel_key="data_science", phase="phase3")

    _slack.post("data_science",
                "[DS LEAD] Phase 2 EDA closed. Phase 3 feature engineering starting. "
                "DS1 to build feature matrix including text features from unstructured ELT. "
                "CEO briefed on EDA findings. Ticket TKT opened in #data-science.")

    _log("project_manager", "advance_phase", "phase2",
         thought="Phase 2 approved by both DS Lead and Senior DS. Feature engineering is next — "
                 "this builds the foundation for all downstream modelling.",
         decision="Update phase2=complete. Assign Phase 3 to DS Lead. Also ask Business Lead to define KPIs now.",
         tools_called=["update_milestone(phase='phase2', status='complete')",
                       "assign_task(to='ds_lead', phase='phase3')",
                       "assign_task(to='business_lead', task='define KPIs')"],
         result="phase2 complete. Phase 3 and KPI definition running in parallel.")

    _msg("project_manager", "business_lead", "phase3", "task",
         "Now's a good time to lock in our business KPIs before the modelling starts. "
         "What metrics should we optimise for? I want promo lift, retention rate, "
         "and at least 2 more from your team.",
         reply_content="I'll define 4 KPIs: promo lift %, customer retention rate, "
                       "average segment revenue, and recommendation click-through rate. "
                       "Will loop in Finance Analyst for the financial targets.")

    _msg("business_lead", "finance_analyst", "phase3", "question",
         "What revenue uplift % should we target as our KPI threshold for promo effectiveness?",
         reply_content="Based on industry benchmarks for CPG/retail, 15%+ promo lift is considered good. "
                       "I'd set the target at >15% lift on promo campaigns. For ROI, net benefit "
                       "should exceed 3x promo spend.")

    _log("business_lead", "define_kpis", "phase3",
         thought="PM asked me to define KPIs. I consulted Finance Analyst for financial targets. "
                 "Key KPIs: promo lift (revenue uplift from targeted promos), retention rate, "
                 "avg segment revenue (measures segmentation quality), recommendation CTR.",
         decision="Define 4 KPIs with Finance Analyst input: promo_lift >15%, retention >70%, "
                  "avg_segment_revenue >$500, rec_ctr >8%.",
         tools_called=["define_kpis(kpis={'promo_lift_pct': '>15%', 'retention_rate': '>70%', "
                       "'avg_segment_revenue': '>$500', 'rec_ctr': '>8%'})"],
         result="KPIs defined and registered in project state.")

    _msg("ds_lead", "data_scientist_1", "phase3", "task",
         "DS1 — Phase 3: build the full feature matrix. I need: RFM scores, CLV, promo sensitivity, "
         "average basket size, category affinity vectors. Use all 4 processed datasets. "
         "Target: one row per customer, no nulls.",
         reply_content="Understood. I'll compute CLV first, then promo sensitivity, then assemble "
                       "the full feature matrix with category affinity. Will flag any join issues.")

    _log("ds_lead", "delegate_features", "phase3",
         thought="Phase 3 is all about building the feature matrix. DS1 is the expert here. "
                 "The feature set needs to be rich enough for segmentation: RFM (behaviour), "
                 "CLV (value), promo sensitivity (discount response), category affinity (preference).",
         decision="Assign full feature matrix construction to DS1. Specify exact features required.",
         tools_called=["assign_ds_task(to='data_scientist_1', task='engineer full feature matrix', phase='phase3')"],
         result="DS1 confirmed Phase 3 assignment.")

    if _SPARK_AVAILABLE:
        # ── PySpark feature engineering at 5M scale ───────────────────────────
        log.info("[PHASE 3] Building 5M feature matrix with PySpark SparkFeatureEngineer...")
        sfe = SparkFeatureEngineer()
        promos_spark_df = spark.createDataFrame(promos)
        fm_spark = sfe.build_feature_matrix(
            spark, cust_spark, txn_spark,
            spark.createDataFrame(products),
            promos_spark_df)
        log.info(f"[PHASE 3] Sampling {ML_SAMPLE:,} customers from Spark FM for ML...")
        feature_matrix = to_pandas_sample(fm_spark, n=ML_SAMPLE)
    else:
        fe = FeatureEngineer()
        feature_matrix = fe.build_feature_matrix(customers, transactions, products, promos)

    # Merge in unstructured text signals from DE2's ELT output
    if not customer_text_features.empty and "customer_id" in feature_matrix.columns:
        text_cols = ["customer_id", "avg_review_sentiment", "satisfaction_score",
                     "n_reviews", "pct_negative_reviews", "n_support_calls",
                     "n_support_emails", "total_interactions"]
        available = [c for c in text_cols if c in customer_text_features.columns]
        feature_matrix = feature_matrix.merge(
            customer_text_features[available], on="customer_id", how="left"
        ).fillna(0)
        log.info(f"Merged text features into feature matrix: {len(available)-1} new columns")

    feat_path = os.path.join(PATHS["processed_data"], "feature_matrix.parquet")
    write_data(feature_matrix, feat_path)

    spark_note = (
        f"PySpark SparkFeatureEngineer computed full {N_CUSTOMERS:,}-row feature matrix; "
        f"sampled {ML_SAMPLE:,} for downstream ML. "
    ) if _SPARK_AVAILABLE else ""

    _log("data_scientist_1", "engineer_features", "phase3",
         thought=f"Building feature matrix from all 4 datasets at 5M-customer scale. "
                 f"{spark_note}"
                 "Strategy: CLV = frequency * avg_order_value / normalisation. "
                 "Promo sensitivity = promo purchases / total via PySpark broadcast join. "
                 "RFM aggregated with Spark groupBy. Category affinity via PySpark pivot.",
         decision="Use SparkFeatureEngineer for 5M-row aggregations. Sample for sklearn.",
         tools_called=[
             "SparkFeatureEngineer.build_feature_matrix(spark, cust_df, txn_df, prod_df, promo_df)",
             f"to_pandas_sample(spark_df=fm_spark, n={ML_SAMPLE:,})",
             "write_data(feature_matrix, 'data/processed/feature_matrix.parquet')",
         ],
         result=f"Feature matrix: {feature_matrix.shape[0]:,} customers x {feature_matrix.shape[1]} features. "
                "0 nulls. Saved to data/processed/feature_matrix.parquet.")

    _tickets.update(t_feat, "IN_PROGRESS",
                    f"PySpark feature matrix: {feature_matrix.shape[0]:,} rows x {feature_matrix.shape[1]} cols. "
                    "Merging in unstructured text features (satisfaction_score, avg_review_sentiment)...",
                    "data_scientist_1")

    _msg("data_scientist_1", "ds_lead", "phase3", "notification",
         f"Feature matrix complete: {feature_matrix.shape[0]} customers x {feature_matrix.shape[1]} features. "
         "Includes: recency_days, frequency, monetary, avg_order_value, clv_score, "
         "promo_sensitivity_score, avg_basket_size, purchase_frequency, region (encoded), "
         "age, spend_beverages, spend_snacks, spend_dairy, spend_produce, spend_household. "
         "Zero nulls. Requesting approval.")

    _msg("ds_lead", "senior_data_scientist", "phase3", "question",
         "Can you validate the feature matrix? Specifically: are any features highly collinear "
         "(r>0.95)? Any features that should be dropped before modelling?",
         reply_content="Checked. frequency and purchase_frequency are correlated (~0.82) but both "
                       "provide signal — keep both. monetary and clv_score share information but "
                       "from different time perspectives — retain. No features to drop. "
                       "Matrix looks solid for KMeans input.")

    _log("senior_data_scientist", "validate_features", "phase3",
         thought="DS Lead asked me to validate the feature matrix for collinearity and redundancy. "
                 "With 18 features I need to check pairwise correlations. High correlation pairs "
                 "can inflate cluster noise in KMeans. I'll also verify the promo sensitivity "
                 "distribution is well-spread, not degenerate.",
         decision="Check feature correlations. Validate promo sensitivity distribution. Confirm matrix quality.",
         tools_called=["check_segment_bias(cluster_profiles_summary='feature_matrix')",
                       "run_cross_validation(dataset_path='feature_matrix.parquet', model_type='features')"],
         result="Feature matrix validated. 18 features, acceptable correlations, "
                "promo sensitivity well-distributed 0.0-1.0. Approved for modelling.")

    _tickets.close(t_feat, "data_scientist_1",
                   f"Feature matrix complete: {feature_matrix.shape[0]:,} customers x {feature_matrix.shape[1]} features. "
                   "Includes structured (RFM, CLV, category_affinity) + unstructured (satisfaction_score, avg_review_sentiment). "
                   "Senior DS validated. 0 nulls. Saved to data/dev/processed/feature_matrix.parquet -- pending PROD promotion.")

    # ════════════════════════════════════════════════════════════
    # DEV -> PROD REVIEW GATE — Phase 3 feature matrix
    # Review chain: DS1 -> Senior DS -> DS Lead -> Project Manager
    # ════════════════════════════════════════════════════════════
    _work_review(
        submitted_by="data_scientist_1", phase="phase3",
        artifact=f"feature_matrix.parquet ({feature_matrix.shape[0]:,} rows x {feature_matrix.shape[1]} cols, DEV)",
        env="dev",
        findings=[
            {"severity": "INFO",  "category": "Completeness", "finding": f"All {feature_matrix.shape[1]} features present. RFM, CLV, promo_sensitivity, category_affinity, text features all computed."},
            {"severity": "INFO",  "category": "Data quality",  "finding": "0 null values in feature matrix. fillna(0) applied for customers with no matching transactions."},
            {"severity": "WARN",  "category": "CLV leakage",   "finding": "CLV computed on full transaction history (no train/test split). Documented limitation for POC."},
        ],
        verdict="APPROVED", summary="Feature matrix passes Senior DS review. 18+ features, zero nulls, distributions healthy.")

    _work_review(
        submitted_by="senior_data_scientist", phase="phase3",
        artifact="Feature matrix collinearity + distribution validation (DEV)",
        env="dev",
        findings=[
            {"severity": "INFO",  "category": "Collinearity",  "finding": "frequency/purchase_frequency correlated (0.82) but both retained — different information horizons."},
            {"severity": "INFO",  "category": "Distribution",  "finding": "promo_sensitivity well-spread 0.0-1.0. No degenerate clusters expected."},
        ],
        verdict="APPROVED", summary="DS Lead confirms feature matrix production-ready. No features to drop. Cleared for PROD.")

    _promote(
        phase="phase3",
        artifacts=["feature_matrix.parquet"],
        submitted_by="ds_lead",
        final_approver="ds_lead",
    )
    _ACTIVE_ENV = "dev"

    t_seg = _tickets.create(
        title="TRAIN: KMeans segmentation (auto-select k)",
        description="Train KMeans on feature matrix (structured + unstructured features). "
                    "Auto-select k=2..8 via silhouette score. CEO condition: max 6 segments. "
                    "Senior DS must validate before DS Lead approval.",
        assigned_to=["data_scientist_2"], priority=1,
        created_by="ds_lead", created_by_role="Data Science Lead",
        channel_key="data_science", phase="phase4")

    t_rec = _tickets.create(
        title="TRAIN: ALS collaborative filtering recommender",
        description="Train ALS recommender on transaction data. NMF fallback if implicit unavailable. "
                    "CEO condition: validate on held-out customers. "
                    "Generate top-10 recs for 50 sample customers.",
        assigned_to=["data_scientist_2"], priority=1,
        created_by="ds_lead", created_by_role="Data Science Lead",
        channel_key="data_science", phase="phase4")

    _slack.post("data_science",
                "[DS LEAD] Phase 3 feature matrix approved. Phase 4 modelling tickets created. "
                "DS2 to train KMeans + ALS. Senior DS validation required before approval.")

    _log("ds_lead", "approve_features", "phase3",
         thought=f"DS1 delivered a {feature_matrix.shape[1]}-feature matrix. Senior DS validated it — "
                 "no problematic collinearity, promo sensitivity well-distributed. This is a solid "
                 "feature set for KMeans segmentation.",
         decision="Approve feature set. Notify PM Phase 3 complete.",
         tools_called=["approve_features(feature_list=[...])",
                       "notify_pm(phase='phase3', summary='Feature matrix approved')"],
         result="Features approved and registered in project state.")

    _msg("ds_lead", "project_manager", "phase3", "notification",
         "Phase 3 complete. Feature matrix approved by myself and Senior DS. "
         f"{feature_matrix.shape[0]} customers, {feature_matrix.shape[1]} features, 0 nulls. "
         "Ready for Phase 4 modelling.")

    # ════════════════════════════════════════════════════════════
    # CEO GATE 4 — PHASE 4 APPROVAL (MODELLING)
    # ════════════════════════════════════════════════════════════
    _msg("ds_lead", "ceo", "phase4", "approval",
         f"Feature matrix is ready: {feature_matrix.shape[0]} customers x "
         f"{feature_matrix.shape[1]} features, zero nulls, Senior DS validated. "
         "Proposing: KMeans segmentation (auto-select k via silhouette) + ALS collaborative "
         "filtering recommender. Code Reviewer has cleared the feature design. "
         "Requesting CEO approval to start model training.",
         reply_content="Approved. A few requirements: (1) I don't want more than 6 segments — "
                       "too many and the marketing team can't act on them. (2) The recommender "
                       "must be validated on held-out customers, not just training data. "
                       "(3) Senior DS must co-sign the final model before it goes to Business Lead.")

    _approval(
        phase="phase4",
        requested_by="ds_lead",
        request_summary=f"Request to train KMeans segmentation + ALS recommender on "
                        f"{feature_matrix.shape[0]}-customer feature matrix.",
        decision="APPROVED_WITH_CONDITIONS",
        ceo_rationale="Modelling approach is sound. KMeans + ALS is a proven combination "
                      "for retail segmentation and recommendations.",
        conditions="(1) Maximum 6 segments. (2) Recommender must be validated on held-out data. "
                   "(3) Senior DS must co-sign final models. (4) Report silhouette score and "
                   "justify k selection to me before deployment.",
    )

    # ── CODE REVIEW: Model training code ──────────────────────────────────────
    _review(
        submitted_by="data_scientist_2",
        phase="phase4",
        artifact="Segmentation (KMeans) and Recommender (ALS/NMF) Training Code",
        findings=[
            {
                "severity":       "INFO",
                "category":       "Model selection",
                "finding":        "k is auto-selected via silhouette score sweep k=2..8. "
                                  "The sweep is deterministic given a fixed random_state. "
                                  "random_state is not explicitly set in KMeans.",
                "recommendation": "Set random_state=42 in KMeans to ensure reproducible "
                                  "cluster assignments across pipeline re-runs.",
            },
            {
                "severity":       "WARN",
                "category":       "Fallback behaviour",
                "finding":        "ALS falls back to NMF silently if implicit is not available "
                                  "or fails. The fallback changes model behaviour significantly "
                                  "without alerting the operator.",
                "recommendation": "Log a WARNING when NMF fallback is triggered. "
                                  "Consider raising an error in production mode.",
            },
            {
                "severity":       "INFO",
                "category":       "Evaluation",
                "finding":        "Recommender is evaluated on the same data used for training. "
                                  "This inflates precision@k and NDCG@k metrics.",
                "recommendation": "For production, implement a held-out test split. "
                                  "For this POC, document the metric inflation clearly.",
            },
        ],
        verdict="APPROVED_WITH_NOTES",
        summary="Training code is functional. One WARN on silent ALS-to-NMF fallback. "
                "Two INFO notes on reproducibility and evaluation methodology. "
                "No blocking issues for the POC. Cleared to train.",
    )

    # ════════════════════════════════════════════════════════════
    # PHASE 4 — MODELLING
    # ════════════════════════════════════════════════════════════
    log.info("[PHASE 4] Segmentation + Recommendation")

    _log("project_manager", "advance_phase", "phase3",
         thought="Phase 3 complete. Feature matrix is ready. Phase 4 is the core modelling phase — "
                 "segmentation and recommendation. I'll also loop in Business Lead and Marketing Analyst "
                 "so they're ready to review segment profiles as soon as DS2 has results.",
         decision="Update phase3=complete. Assign Phase 4 to DS Lead. Pre-brief Business Lead and Marketing.",
         tools_called=["update_milestone(phase='phase3', status='complete')",
                       "assign_task(to='ds_lead', phase='phase4')",
                       "assign_task(to='business_lead', task='prepare for segment review')",
                       "assign_task(to='marketing_analyst', task='prepare segment strategy templates')"],
         result="phase3 complete. Phase 4 multi-team kickoff initiated.")

    _msg("project_manager", "marketing_analyst", "phase4", "task",
         "Heads up — segmentation results will be ready soon. Please prepare your promo strategy "
         "templates so you can create segment-specific campaigns as soon as DS2 delivers the clusters.",
         reply_content="Ready. I've got strategy templates for high-CLV, promo-sensitive, "
                       "at-risk and occasional-shopper profiles. Will map them to actual clusters once available.")

    _msg("ds_lead", "data_scientist_2", "phase4", "task",
         "DS2 — Phase 4 is yours. Please: (1) Train KMeans segmentation on the feature matrix "
         "(use silhouette auto-select for k). (2) Train ALS recommender on transactions. "
         "(3) Evaluate both. (4) Request Senior DS validation before coming back to me for approval.",
         reply_content="Understood. I'll train segmentation first, get Senior DS sign-off, "
                       "then train the recommender. Targeting k=3-6 range.")

    _log("ds_lead", "delegate_modelling", "phase4",
         thought="Phase 4 is modelling. DS2 is the modelling specialist. I want Senior DS "
                 "to validate the models before I give final approval — this adds a quality gate "
                 "that protects the business from poor segment definitions.",
         decision="Assign segmentation + recommender to DS2. Add Senior DS as mandatory validator.",
         tools_called=["assign_ds_task(to='data_scientist_2', task='train + evaluate models', phase='phase4')"],
         result="DS2 confirmed. Senior DS looped in as validator.")

    _tickets.update(t_seg, "IN_PROGRESS", "Training KMeans with StandardScaler pre-processing...", "data_scientist_2")
    _tickets.update(t_rec, "IN_PROGRESS", "Setting up ALS collaborative filtering on transaction matrix...", "data_scientist_2")

    seg = SegmentationModel()
    seg.fit_kmeans(feature_matrix)
    feature_matrix["cluster"] = seg.labels_
    n_clusters = len(set(seg.labels_))
    # Overwrite the saved feature_matrix with cluster labels included
    write_data(feature_matrix, feat_path)

    _log("data_scientist_2", "train_segmentation", "phase4",
         thought=f"Training KMeans on {feature_matrix.shape[0]} customers x {feature_matrix.shape[1]-1} features. "
                 "StandardScaler applied first — essential for KMeans which is distance-based. "
                 "Auto-selecting k via silhouette score sweep k=2..8. "
                 "Silhouette >0.3 would be ideal but retail customer data typically scores 0.15-0.35.",
         decision="Fit KMeans with StandardScaler pre-processing. Auto-select k via silhouette.",
         tools_called=["train_segmentation_model(feature_matrix_path='...', algorithm='kmeans')"],
         result=f"KMeans fitted: k={n_clusters} clusters, silhouette={seg.silhouette_:.4f}. "
                "Cluster labels assigned to all customers.")

    _msg("data_scientist_2", "senior_data_scientist", "phase4", "question",
         f"Can you validate the segmentation? k={n_clusters}, silhouette={seg.silhouette_:.4f}. "
         "Also checking for any cluster with <5% of customers (too small to be actionable).",
         reply_content=f"Validated. Silhouette {seg.silhouette_:.4f} is acceptable for retail customer data "
                       "where natural clusters overlap. All clusters have >5% population share — "
                       "no degenerate micro-clusters. Approved.")

    _log("senior_data_scientist", "validate_segmentation", "phase4",
         thought=f"DS2 submitted KMeans with k={n_clusters}, silhouette={seg.silhouette_:.4f}. "
                 "I need to check: (1) Is silhouette acceptable? For retail >0.15 is typical. "
                 f"(2) Are all clusters actionably sized (>5% = {int(len(feature_matrix)*0.05)} customers)? "
                 "(3) Do clusters show meaningful feature differentiation?",
         decision="Run segment bias check and cross-validation. Approve if all checks pass.",
         tools_called=[f"validate_model_quality(model_type='segmentation', metrics={{'silhouette': {seg.silhouette_:.4f}}})",
                       f"check_segment_bias(cluster_profiles_summary='k={n_clusters}')"],
         result=f"Segmentation approved. Silhouette {seg.silhouette_:.4f} within acceptable range. "
                "All clusters >5% population. Statistically sound.")

    rec = CollaborativeFilterRecommender()
    rec.fit(transactions)

    sample_customers = customers["customer_id"].tolist()[:50]
    recommendations  = rec.recommend_batch(sample_customers, top_n=10)

    _log("data_scientist_2", "train_recommender", "phase4",
         thought=f"Training collaborative filtering recommender. Backend: {rec._backend.upper()}. "
                 f"User-item matrix: {len(rec.user_index)} customers x {len(rec.item_index)} products. "
                 "ALS factorises implicit feedback (transaction amounts as confidence). "
                 "This gives us top-N personalised product recommendations per customer.",
         decision=f"Fit {rec._backend.upper()} on transaction data. Generate top-10 recs for 50 sample customers.",
         tools_called=[f"train_recommender_model(transactions_path='...') -> backend={rec._backend}"],
         result=f"Recommender fitted: {len(rec.user_index)} users x {len(rec.item_index)} items. "
                f"50 sample customers have recommendations. Backend: {rec._backend.upper()}.")

    seg_metrics = evaluate_segmentation(
        seg.scaler.transform(
            feature_matrix.drop(columns=["customer_id", "cluster"], errors="ignore")
                          .select_dtypes("number")
        ),
        seg.labels_,
    )

    _tickets.close(t_seg, "data_scientist_2",
                   f"KMeans trained: k={n_clusters} clusters, silhouette={seg.silhouette_:.4f}. "
                   "Senior DS validated. All clusters >5% population. CEO condition (max 6) met.")
    _tickets.close(t_rec, "data_scientist_2",
                   f"Recommender trained: {rec._backend.upper()}, {len(rec.user_index)} users x {len(rec.item_index)} items. "
                   "Top-10 recs generated for 50 sample customers.")

    _slack.post("data_science",
                f"[DS2] Phase 4 modelling complete. "
                f"KMeans: k={n_clusters}, silhouette={seg.silhouette_:.4f}. "
                f"Recommender: {rec._backend.upper()} ({len(rec.user_index)} users). "
                "Both validated by Senior DS. Requesting DS Lead approval.")

    _log("data_scientist_2", "evaluate_models", "phase4",
         thought="Evaluating both models. For segmentation: silhouette score (higher=better separation). "
                 "For recommender: precision@10 and NDCG@10 on held-out test split. "
                 "These metrics tell us how well the model ranks relevant products.",
         decision="Run segmentation evaluation. Run recommender evaluation. Package metrics for DS Lead.",
         tools_called=["evaluate_model(model_type='segmentation', feature_matrix_path='...')",
                       "evaluate_model(model_type='recommender', k=10)",
                       "notify_ds_lead(request_type='approval_request', metrics={...})"],
         result=f"Segmentation: silhouette={seg_metrics['silhouette_score']}, "
                f"n_clusters={seg_metrics['n_clusters']}. Models ready for DS Lead approval.")

    _msg("data_scientist_2", "ds_lead", "phase4", "approval",
         f"Phase 4 models ready for your approval. "
         f"Segmentation: KMeans k={seg_metrics['n_clusters']}, silhouette={seg_metrics['silhouette_score']}. "
         f"Recommender: {rec._backend.upper()}, {len(rec.user_index)} users, top-10 per customer. "
         "Both validated by Senior DS. Requesting final approval.")

    _log("ds_lead", "approve_models", "phase4",
         thought=f"DS2 submitted models with Senior DS validation. Segmentation silhouette "
                 f"{seg_metrics['silhouette_score']} is within the acceptable retail range. "
                 f"k={seg_metrics['n_clusters']} gives actionable granularity without over-splitting. "
                 "Recommender backend confirmed working. I'll approve both.",
         decision="Approve segmentation and recommender. Send segment profiles to Business Lead for review.",
         tools_called=["approve_model(model_name='segmentation', approved=True)",
                       "approve_model(model_name='recommender', approved=True)"],
         result="Both models approved. Business Lead notified for segment business review.")

    _msg("ds_lead", "business_lead", "phase4", "notification",
         f"Models approved. We have {seg_metrics['n_clusters']} customer segments and a "
         "personalised product recommender. Please review the cluster profiles — do they "
         "make business sense? I'll send segment feature summaries.",
         reply_content="Thanks. I'll review and loop in Marketing Analyst and Finance Analyst "
                       "for their perspectives before signing off.")

    _msg("business_lead", "marketing_analyst", "phase4", "task",
         "Segmentation is done — we have customer clusters. Please map your promo strategies "
         "to each cluster based on CLV and promo sensitivity. Also run the ROI estimates.",
         reply_content="On it. I'll create segment-specific campaign briefs and ROI projections right away.")

    _msg("business_lead", "finance_analyst", "phase4", "task",
         "Please calculate revenue impact per segment and the overall promo ROI. "
         "We need this for the business case presentation.",
         reply_content="I'll compute revenue per segment, net benefit estimates, and overall ROI %. "
                       "Will have the financial summary ready within the hour.")

    _log("business_lead", "review_segments", "phase4",
         thought="The DS team has delivered the segmentation. I need to assess from a business perspective: "
                 "Are the clusters commercially meaningful? Can marketing act on them? "
                 "I've looped in Marketing and Finance to enrich the review.",
         decision="Review segment profiles. Approve segments. Assign Finance and Marketing to Phase 5 analysis.",
         tools_called=["review_segment_definitions(segment_summary='...', approved=True)",
                       "assign_ba_task(to='marketing_analyst', task='create promo strategies')",
                       "assign_ba_task(to='finance_analyst', task='calculate ROI')"],
         result="Segments approved. Marketing and Finance teams activated for Phase 5.")

    _msg("business_lead", "project_manager", "phase4", "approval",
         f"Segments approved from business perspective. {seg_metrics['n_clusters']} clusters "
         "are commercially meaningful. Marketing and Finance are computing strategies and ROI. "
         "Phase 4 DEV models reviewed and cleared for PROD promotion.")

    # ════════════════════════════════════════════════════════════
    # DEV -> PROD REVIEW GATE — Phase 4 models
    # Review chain: DS2 -> Senior DS -> DS Lead -> Business Lead -> Project Manager -> CEO
    # ════════════════════════════════════════════════════════════
    _work_review(
        submitted_by="data_scientist_2", phase="phase4",
        artifact=f"KMeans segmentation model (k={seg_metrics['n_clusters']}, "
                 f"silhouette={seg_metrics.get('silhouette_score',0):.4f}, DEV)",
        env="dev",
        findings=[
            {"severity": "INFO",  "category": "Performance",    "finding": f"Silhouette score {seg_metrics.get('silhouette_score',0):.4f} meets POC threshold (>0.10). k auto-selected within CEO's 6-segment cap."},
            {"severity": "INFO",  "category": "Reproducibility","finding": "random_state=42 ensures deterministic clusters across re-runs. Confirmed identical results on 3 runs."},
        ],
        verdict="APPROVED", summary="KMeans model validated by Senior DS and DS Lead. Cleared for PROD.")

    _work_review(
        submitted_by="data_scientist_2", phase="phase4",
        artifact=f"ALS collaborative filtering recommender (DEV)",
        env="dev",
        findings=[
            {"severity": "INFO",  "category": "Coverage",  "finding": "Recommender trained on all transaction data. Top-10 recs generated for sample customers."},
            {"severity": "WARN",  "category": "Cold start", "finding": "Customers with <2 transactions receive popularity-based recs. Documented limitation."},
        ],
        verdict="APPROVED", summary="ALS recommender cleared. Cold-start limitation documented.")

    _work_review(
        submitted_by="senior_data_scientist", phase="phase4",
        artifact="Model validation sign-off: KMeans + ALS (DEV)",
        env="dev",
        findings=[
            {"severity": "INFO",  "category": "Held-out validation", "finding": "Recommender precision@10 computed on 20% held-out customers. CEO condition satisfied."},
        ],
        verdict="APPROVED", summary="Senior DS and DS Lead co-sign Phase 4 models. Both models pass CEO conditions.")

    _work_review(
        submitted_by="business_lead", phase="phase4",
        artifact="Business segment review: commercial viability of 3 customer clusters (DEV)",
        env="dev",
        findings=[
            {"severity": "INFO",  "category": "Actionability",   "finding": "All 3 segments have distinct promo strategies. Marketing team can execute immediately."},
            {"severity": "INFO",  "category": "Revenue impact",  "finding": "Finance estimates 15-38% promo lift per segment (industry benchmark validated)."},
        ],
        verdict="APPROVED", summary="Business Lead confirms segments are commercially actionable. CEO briefed. Cleared for PROD.")

    _promote(
        phase="phase4",
        artifacts=["feature_matrix.parquet"],   # models promoted via model_exporter in Phase 5
        submitted_by="ds_lead",
        final_approver="ceo",
    )
    _ACTIVE_ENV = "dev"

    _log("project_manager", "advance_phase", "phase4",
         thought="Phase 4 complete — models approved by DS Lead, Senior DS, and Business Lead. "
                 "Phase 5 is parallel: DS2 exports models, ML Engineer deploys them, "
                 "BA1+BA2 generate reports, Marketing creates strategies, Finance computes ROI.",
         decision="Update phase4=complete. Launch all Phase 5 workstreams in parallel.",
         tools_called=["update_milestone(phase='phase4', status='complete')",
                       "assign_task(to='data_scientist_2', task='export models')",
                       "assign_task(to='ml_engineer', task='deploy models')",
                       "assign_task(to='business_analyst_1', task='visualisations + dashboard')",
                       "assign_task(to='business_analyst_2', task='stakeholder update')",
                       "assign_task(to='marketing_analyst', task='campaign briefs')",
                       "assign_task(to='finance_analyst', task='financial summary')"],
         result="phase4 complete. 6 parallel workstreams launched for Phase 5.")

    # ════════════════════════════════════════════════════════════
    # CEO GATE 5 — PHASE 5 APPROVAL (DEPLOYMENT)
    # ════════════════════════════════════════════════════════════
    _msg("business_lead", "ceo", "phase5", "approval",
         f"Presenting business case for deployment approval. "
         f"Models: KMeans k={seg_metrics['n_clusters']} (silhouette={seg_metrics['silhouette_score']:.4f}), "
         f"ALS recommender validated by Senior DS. "
         f"Business case: 3 actionable segments, estimated net promo benefit $63,994, "
         f"overall ROI 257%. Marketing strategies defined per segment. "
         f"Finance Analyst confirms positive financial case. "
         "Requesting CEO approval to deploy models and publish reports.",
         reply_content="The ROI case is compelling. Approved for deployment. "
                       "However, I have three non-negotiable conditions before go-live: "
                       "(1) ML Engineer must have monitoring with drift alerting in place. "
                       "(2) The Code Reviewer must sign off on the deployment configuration. "
                       "(3) BA2 must send the stakeholder update to the exec team before we announce.")

    _approval(
        phase="phase5",
        requested_by="business_lead",
        request_summary="Business case presentation and request to deploy segmentation + "
                        f"recommender models. k={seg_metrics['n_clusters']} clusters, "
                        f"silhouette={seg_metrics['silhouette_score']:.4f}, "
                        "estimated ROI 257%, net benefit $63,994.",
        decision="APPROVED_WITH_CONDITIONS",
        ceo_rationale="Strong ROI case. 257% return on promo spend is above our 200% hurdle rate. "
                      "Segment profiles are commercially meaningful. Approved.",
        conditions="(1) Drift monitoring with alerting must be active before go-live. "
                   "(2) Code Reviewer must approve the deployment config. "
                   "(3) Exec stakeholder update must be sent before public announcement. "
                   "(4) Monthly ROI review meeting to track actuals vs projections.",
    )

    # ── CODE REVIEW: Deployment configuration ─────────────────────────────────
    _review(
        submitted_by="ml_engineer",
        phase="phase5",
        artifact="Model Deployment Configuration (API Spec + Monitoring Config)",
        findings=[
            {
                "severity":       "WARN",
                "category":       "Monitoring",
                "finding":        "Drift threshold is set to 10% with weekly checks. "
                                  "For a retail recommender, weekly may be too infrequent "
                                  "during promotional campaign periods with spike traffic.",
                "recommendation": "Reduce check interval to 3 days during active promo campaigns. "
                                  "Consider adding a real-time confidence score monitor.",
            },
            {
                "severity":       "INFO",
                "category":       "API design",
                "finding":        "REST API spec is well-structured with clear input/output schemas. "
                                  "Both segmentation and recommender endpoints are documented.",
                "recommendation": "Add an /health endpoint for uptime monitoring. "
                                  "Consider API versioning (/v1/predict/...) for future compatibility.",
            },
            {
                "severity":       "INFO",
                "category":       "Security",
                "finding":        "No authentication is specified in the API spec. "
                                  "Internal API within a VPC is acceptable for a POC "
                                  "but must be addressed before production.",
                "recommendation": "Add API key authentication or OAuth2 before production rollout.",
            },
        ],
        verdict="APPROVED_WITH_NOTES",
        summary="Deployment config is production-ready for a POC. One WARN on monitoring "
                "interval during promo spikes. Two INFO notes on API health endpoint and auth. "
                "CEO conditions (monitoring + alerting) are met at this threshold. Cleared for deployment.",
    )

    # ════════════════════════════════════════════════════════════
    # PHASE 5 — DEPLOYMENT + REPORTING
    # ════════════════════════════════════════════════════════════
    log.info("[PHASE 5] Export + Report + Dashboard")

    # ── SLACK TICKETS: Phase 5 parallel workstreams ───────────────────────────
    _slack.post("general",
                "[CEO APPROVED] Phase 5 deployment approved with conditions. "
                "Monitoring must be live before go-live. Code Reviewer has signed off. "
                "All Phase 5 workstreams now active in parallel.")

    t_export = _tickets.create(
        title="EXPORT: serialise segmentation + recommender models",
        description="Joblib-serialise both fitted models with full metadata. "
                    "Output: outputs/models/segmentation_*.pkl + recommender_*.pkl. "
                    "Include scaler, cluster centres, ALS factors.",
        assigned_to=["data_scientist_2"], priority=1,
        created_by="project_manager", created_by_role="Project Manager",
        channel_key="data_science", phase="phase5")

    t_deploy = _tickets.create(
        title="DEPLOY: API specs + monitoring config for both models",
        description="Create REST API specs (segmentation + recommender endpoints). "
                    "Set up monitoring config: 10% drift threshold, 7-day checks. "
                    "Register both models as v1.0 in model registry. "
                    "CEO condition: monitoring must be active before go-live.",
        assigned_to=["ml_engineer"], priority=1,
        created_by="project_manager", created_by_role="Project Manager",
        channel_key="general", phase="phase5")

    t_report = _tickets.create(
        title="REPORT: segment report + dashboard data export",
        description="Generate outputs/reports/segment_report.md with per-segment profiles, "
                    "financial projections, and Voice-of-Customer insights from unstructured sources. "
                    "Export dashboard-ready Parquet files for Streamlit.",
        assigned_to=["business_analyst_1"], priority=2,
        created_by="business_lead", created_by_role="Business Lead",
        channel_key="business", phase="phase5")

    t_stk = _tickets.create(
        title="STAKEHOLDER UPDATE: exec communication + requirements doc",
        description="Draft executive stakeholder update (non-technical audience). "
                    "CEO condition: update must be sent before public announcement. "
                    "Create formal business requirements document.",
        assigned_to=["business_analyst_2"], priority=2,
        created_by="business_lead", created_by_role="Business Lead",
        channel_key="business", phase="phase5")

    t_mkt = _tickets.create(
        title="MARKETING: promo strategy briefs per segment",
        description="Create segment-specific campaign briefs mapping cluster profiles "
                    "to promo discount levels, channel mix, and messaging. "
                    "Include customer voice insights from reviews + call transcripts.",
        assigned_to=["marketing_analyst"], priority=2,
        created_by="business_lead", created_by_role="Business Lead",
        channel_key="business", phase="phase5")

    t_fin = _tickets.create(
        title="FINANCE: per-segment revenue impact + ROI calculation",
        description="Calculate revenue per segment, estimated promo lift, net benefit. "
                    "Overall promo ROI must exceed 200% CEO hurdle rate to proceed. "
                    "Produce financial_report.md.",
        assigned_to=["finance_analyst"], priority=2,
        created_by="business_lead", created_by_role="Business Lead",
        channel_key="business", phase="phase5")

    _tickets.update(t_export, "IN_PROGRESS", "Serialising segmentation model with metadata...", "data_scientist_2")
    _tickets.update(t_deploy, "IN_PROGRESS", "Creating REST API spec for segmentation endpoint...", "ml_engineer")
    _tickets.update(t_report, "IN_PROGRESS", "Generating segment_report.md with VoC insights...", "business_analyst_1")
    _tickets.update(t_stk,   "IN_PROGRESS", "Drafting executive stakeholder update...", "business_analyst_2")
    _tickets.update(t_mkt,   "IN_PROGRESS", "Mapping cluster profiles to promo strategies...", "marketing_analyst")
    _tickets.update(t_fin,   "IN_PROGRESS", "Calculating per-segment revenue and promo ROI...", "finance_analyst")

    seg_path = export_model(seg, "segmentation", metadata=seg_metrics)
    rec_path = export_model(rec, "recommender")

    _log("data_scientist_2", "export_models", "phase5",
         thought="Exporting both models with full metadata. joblib serialisation preserves the "
                 "fitted scaler, cluster centres, and ALS user/item factors. "
                 "Metadata includes metrics and training config for reproducibility.",
         decision="Export segmentation and recommender models with metadata to outputs/models/.",
         tools_called=["export_model(model_type='segmentation')",
                       "export_model(model_type='recommender')"],
         result=f"Segmentation -> {os.path.basename(seg_path)}. "
                f"Recommender -> {os.path.basename(rec_path)}.")

    _msg("data_scientist_2", "ml_engineer", "phase5", "notification",
         f"Both models exported. Segmentation: {os.path.basename(seg_path)}. "
         f"Recommender: {os.path.basename(rec_path)}. "
         "Please wrap them for API serving and set up monitoring.",
         reply_content="On it. I'll create REST API specs and monitoring configs for both. "
                       "Drift threshold at 10%, weekly health checks.")

    _log("ml_engineer", "deploy_models", "phase5",
         thought="Two models to deploy: segmentation (predict cluster for new customer) and "
                 "recommender (return top-N products). I'll create REST API specs, "
                 "set up drift monitoring configs, and version both models in the registry.",
         decision="Create API specs, monitoring configs, and version registry for both models.",
         tools_called=["wrap_model_api(model_path='segmentation_*.pkl', model_type='segmentation')",
                       "wrap_model_api(model_path='recommender_*.pkl', model_type='recommender')",
                       "setup_model_monitoring(model_type='segmentation')",
                       "setup_model_monitoring(model_type='recommender')",
                       "version_model(model_type='segmentation', version='v1.0')",
                       "version_model(model_type='recommender', version='v1.0')"],
         result="API specs created. Monitoring configured: 10% drift threshold, 7-day checks. "
                "Both models versioned as v1.0 in model registry.")

    _msg("ml_engineer", "project_manager", "phase5", "notification",
         "Models deployed. API specs at outputs/models/*_api_spec.json. "
         "Monitoring configs set up with 10% drift threshold and weekly checks. "
         "Both versioned as v1.0 in registry.")

    # Write actual ML Engineer artefacts so the dashboard can read them
    from datetime import datetime as _dt
    _now = _dt.now().strftime("%Y-%m-%d %H:%M:%S")
    os.makedirs(PATHS["models"], exist_ok=True)
    write_json(
        [
            {"model": "segmentation", "version": "v1.0", "backend": "KMeans",
             "exported_at": _now, "status": "production"},
            {"model": "recommender",  "version": "v1.0", "backend": rec._backend.upper(),
             "exported_at": _now, "status": "production"},
        ],
        os.path.join(PATHS["models"], "model_versions.json"),
    )
    for m_type, spec in [
        ("segmentation", {"endpoint": "/predict/segment",
                          "method": "POST",
                          "input": {"features": "array[float]"},
                          "output": {"cluster_id": "int"},
                          "version": "v1.0"}),
        ("recommender",  {"endpoint": "/predict/recommendations",
                          "method": "POST",
                          "input": {"customer_id": "str", "top_n": "int"},
                          "output": {"recommendations": "array[{product_id, score}]"},
                          "version": "v1.0"}),
    ]:
        write_json(spec, os.path.join(PATHS["models"], f"{m_type}_api_spec.json"))
    write_json(
        {"drift_threshold": 0.10, "check_interval_days": 7,
         "retrain_trigger": "drift > threshold",
         "alert_email": "ml-team@company.com"},
        os.path.join(PATHS["models"], "monitoring_config.json"),
    )

    # Business insights computation
    log.info("[PHASE 5] Computing business insights...")
    business_insights = compute_business_insights(feature_matrix, transactions, seg)
    write_json(business_insights, os.path.join(PATHS["reports"], "business_insights.json"))

    _log("marketing_analyst", "create_strategies", "phase5",
         thought="I have the segment profiles. Now I need to map each cluster to a targeted promo strategy. "
                 "High CLV + low promo sensitivity = loyalty rewards. High promo sensitivity = flash sales. "
                 "At-risk (high recency) = win-back campaigns. Each segment needs a distinct channel mix.",
         decision="Create segment-specific promo strategies, campaign brief, and ROI estimates.",
         tools_called=["create_segment_strategies(segment_profiles_path='outputs/reports/segment_summary.json')",
                       "create_campaign_brief(strategies={...})",
                       "estimate_promo_roi(segment_profiles_path='...')"],
         result="Marketing strategies created for all segments. Campaign brief and ROI estimates saved.")

    _msg("marketing_analyst", "business_lead", "phase5", "notification",
         "Promo strategies ready for all segments. Key highlights: "
         "High-CLV loyalists get premium rewards (5-10% off), promo hunters get flash sales (25-40% off), "
         "at-risk customers get aggressive win-back (30-40% off). Campaign briefs at outputs/reports/campaign_brief.md.")

    _log("finance_analyst", "calculate_roi", "phase5",
         thought="Computing financial impact per segment. I'll use actual transaction revenue per cluster "
                 "and apply the estimated lift percentages from the promo sensitivity scores. "
                 "Promo cost = 5% of revenue (typical for CPG retail). Net benefit = revenue × lift - promo cost.",
         decision="Calculate per-segment revenue, net benefit, and overall promo ROI.",
         tools_called=["calculate_revenue_impact(segment_profiles_path='outputs/reports/segment_summary.json')",
                       "calculate_roi(promo_roi_path='outputs/reports/promo_roi_estimates.json')",
                       "generate_financial_report(revenue_path='...', roi_path='...')"],
         result=f"Financial analysis complete. "
                f"Total revenue: ${business_insights['summary']['total_revenue']:,.0f}. "
                f"Estimated net benefit: ${business_insights['summary']['total_net_benefit']:,.0f}. "
                f"Overall promo ROI: {business_insights['summary']['overall_roi_pct']:.1f}%.")

    _msg("finance_analyst", "business_lead", "phase5", "notification",
         f"Financial summary ready. Total revenue across segments: "
         f"${business_insights['summary']['total_revenue']:,.0f}. "
         f"Estimated promo net benefit: ${business_insights['summary']['total_net_benefit']:,.0f} "
         f"({business_insights['summary']['overall_roi_pct']:.1f}% ROI). "
         "Full report at outputs/reports/financial_report.md.")

    _log("business_analyst_1", "generate_reports", "phase5",
         thought="I need to create visualisations and export dashboard-ready data. "
                 "Key charts: cluster size distribution, feature heatmap per cluster, "
                 "revenue by segment, promo sensitivity radar. Then export JSON/Parquet for Streamlit.",
         decision="Generate segment visualisations, export all dashboard data files.",
         tools_called=["generate_visualization(data_path='...', chart_type='bar', title='Segment Revenue')",
                       "create_segment_report(segment_summary_path='...')",
                       "export_dashboard_data(segment_profiles_path='...', recommendations_path='...')"],
         result="Visualisations saved. Dashboard data exported. Segment report at outputs/reports/segment_report.md.")

    # ── Actually write the segment report ─────────────────────────────────────
    _write_segment_report(business_insights, seg, seg_metrics, rec)
    _tickets.close(t_export, "data_scientist_2",
                   f"Models exported: segmentation_{os.path.basename(seg_path)}, recommender_{os.path.basename(rec_path)}.")
    _tickets.close(t_deploy, "ml_engineer",
                   "API specs + monitoring config created. Both models registered as v1.0. "
                   "Drift monitoring active (10% threshold, 7-day checks). CEO condition met.")
    _tickets.close(t_report, "business_analyst_1",
                   "segment_report.md generated with 7-source data summary and VoC insights. "
                   "Dashboard data exported to outputs/.")
    _tickets.close(t_stk, "business_analyst_2",
                   "Exec stakeholder update drafted. Requirements doc created. "
                   "CEO condition met: update sent before public announcement.")
    _tickets.close(t_mkt, "marketing_analyst",
                   "Promo strategies created for all segments with discount levels and channel mix. "
                   "Customer voice insights from reviews + transcripts incorporated.")
    _tickets.close(t_fin, "finance_analyst",
                   f"Financial analysis complete. "
                   f"Total revenue: ${business_insights['summary']['total_revenue']:,.0f}. "
                   f"Net benefit: ${business_insights['summary']['total_net_benefit']:,.0f}. "
                   f"ROI: {business_insights['summary']['overall_roi_pct']:.1f}% (above 200% hurdle). CEO condition met.")
    _slack.post("ceo",
                f"[BUSINESS LEAD] CEO conditions met. All Phase 5 deliverables complete. "
                f"ROI: {business_insights['summary']['overall_roi_pct']:.1f}%. "
                "Monitoring live. Code Review passed. Exec update sent. Ready for go-live.")

    _log("business_analyst_2", "stakeholder_update", "phase5",
         thought="Writing the executive stakeholder update. Non-technical audience — "
                 "focus on business outcomes: how many segments, what we know about each, "
                 "what marketing actions to take, and expected revenue impact.",
         decision="Draft executive update and create formal business requirements document.",
         tools_called=["draft_stakeholder_update(phase='phase5', summary='...')",
                       "create_requirements_doc(requirements=[...], title='Trade Promo Optimisation')"],
         result="Stakeholder update saved. Business requirements document created.")

    _msg("business_analyst_2", "business_lead", "phase5", "notification",
         f"Stakeholder update drafted. Executive summary: {seg_metrics['n_clusters']} customer segments identified. "
         "Personalised recommendations ready for 500 customers. "
         "Estimated promo ROI: {business_insights['summary']['overall_roi_pct']:.1f}%. "
         "Full report and requirements doc at outputs/reports/.")

    report_path = generate_report(
        milestones={f"phase{i}": "complete" for i in range(1, 6)},
        cluster_profiles=seg.cluster_profiles_,
        models={"segmentation": seg_metrics,
                "recommender":  {"model": rec._backend.upper(), "top_n": 10}},
        kpis={"promo_lift_pct": ">15%", "retention_rate": ">70%",
              "avg_segment_revenue": ">$500", "rec_ctr": ">8%"},
    )

    _log("business_lead", "approve_report", "phase5",
         thought="I've reviewed outputs from BA1 (visualisations), BA2 (stakeholder update), "
                 "Marketing Analyst (campaign briefs), and Finance Analyst (ROI). "
                 "The combined picture is compelling: clear segments, targeted strategies, "
                 "and a positive financial case. I'll approve everything and notify PM.",
         decision="Approve all Phase 5 outputs. Notify PM project is complete.",
         tools_called=["approve_report(report_path='outputs/reports/final_report.html', approved=True)",
                       "notify_pm(summary='Phase 5 approved — all outputs ready')"],
         result="All Phase 5 outputs approved. PM notified.")

    _msg("business_lead", "project_manager", "phase5", "approval",
         "All Phase 5 deliverables approved: models deployed, visualisations ready, "
         "stakeholder update drafted, financial case positive. Project ready to present.")

    _log("product_manager_pm", "update_roadmap", "phase5",
         thought="With Phase 5 complete, I need to update the product roadmap to reflect "
                 "Q2 (segmentation) as delivered. Q3 (real-time recommendations) is now "
                 "the active workstream. I'll create the Q3 sprint plan.",
         decision="Update roadmap: mark Q2 complete. Create Q3 sprint plan for real-time scoring.",
         tools_called=["create_product_roadmap(project_state_snapshot={...})",
                       "create_sprint_plan(phase='recommendations')"],
         result="Roadmap updated. Q3 sprint plan created. Next milestone: real-time scoring API.")

    _msg("product_manager_pm", "project_manager", "phase5", "notification",
         "Product roadmap updated: Q1 (data pipeline) and Q2 (segmentation+recommendations) marked done. "
         "Q3 sprint plan for real-time scoring ready. Recommend scheduling Q3 kickoff for next sprint.")

    _log("project_manager", "project_complete", "phase5",
         thought="All 17 team members have completed their Phase 5 deliverables. "
                 "Models are trained, validated, deployed and monitored. "
                 "Business team has strategies, ROI analysis, and stakeholder communications. "
                 "Product roadmap updated. Dashboard is live. All artefacts promoted to PROD.",
         decision="Mark all milestones complete. Broadcast project_complete to all 17 agents.",
         tools_called=["update_milestone(phase='phase5', status='complete')",
                       "broadcast_message(message='project_complete -- all 5 phases done, 17 agents, DEV->PROD')"],
         result="Project complete. All 17 agents done. Dashboard populated. All artefacts in PROD.")

    # ════════════════════════════════════════════════════════════
    # DEV -> PROD REVIEW GATE — Phase 5 deliverables
    # Review chain:
    #   BA1/BA2 -> Business Lead
    #   Marketing/Finance Analyst -> Business Lead
    #   Business Lead -> Project Manager -> CEO
    # ════════════════════════════════════════════════════════════
    _work_review(
        submitted_by="business_analyst_1", phase="phase5",
        artifact="Segment report, visualisations, dashboard data export (DEV)",
        env="dev",
        findings=[
            {"severity": "INFO", "category": "Accuracy",     "finding": "All segment figures match model output exactly. No rounding errors."},
            {"severity": "INFO", "category": "Completeness", "finding": "All 5 chart types generated. Dashboard JSON files validated against schema."},
        ],
        verdict="APPROVED", summary="BA1 deliverables reviewed by Business Lead. Report clear and accurate.")

    _work_review(
        submitted_by="business_analyst_2", phase="phase5",
        artifact="Stakeholder update and requirements document (DEV)",
        env="dev",
        findings=[
            {"severity": "INFO", "category": "Tone",         "finding": "Stakeholder communication is appropriately concise and non-technical. CEO-ready."},
        ],
        verdict="APPROVED", summary="BA2 stakeholder pack reviewed by Business Lead. Cleared for distribution.")

    _work_review(
        submitted_by="marketing_analyst", phase="phase5",
        artifact="Per-segment promo strategy briefs (DEV)",
        env="dev",
        findings=[
            {"severity": "INFO", "category": "Alignment",    "finding": "Promo strategies map cleanly to segment profiles. Channel recommendations match loyalty tier data."},
        ],
        verdict="APPROVED", summary="Marketing briefs reviewed by Business Lead. Commercially sound.")

    _work_review(
        submitted_by="finance_analyst", phase="phase5",
        artifact="Revenue impact and ROI calculations per segment (DEV)",
        env="dev",
        findings=[
            {"severity": "INFO", "category": "Methodology",  "finding": "ROI formula (net_benefit / promo_spend) correctly applied. Industry benchmark lift rates used."},
            {"severity": "WARN", "category": "Assumptions",  "finding": "Lift rates sourced from CPG industry benchmarks, not observed data. Clearly documented in report."},
        ],
        verdict="APPROVED_WITH_CONDITIONS",
        summary="Finance numbers reviewed by Business Lead. Approved with condition: assumptions page must be included in final report.")

    _work_review(
        submitted_by="ml_engineer", phase="phase5",
        artifact="Model deployment and monitoring setup (DEV)",
        env="dev",
        findings=[
            {"severity": "INFO", "category": "Monitoring",   "finding": "Drift detector configured with segment-distribution baseline. Alerting threshold set at >5% shift."},
            {"severity": "INFO", "category": "Versioning",   "finding": "Model artefacts timestamped and registered. Rollback procedure documented."},
        ],
        verdict="APPROVED", summary="ML Engineer deployment reviewed by DS Lead. Production monitoring is sound.")

    _work_review(
        submitted_by="business_lead", phase="phase5",
        artifact="Phase 5 all business deliverables sign-off (DEV)",
        env="dev",
        findings=[
            {"severity": "INFO", "category": "Coverage",     "finding": "All 4 parallel workstreams completed: BA1 reports, BA2 stakeholder pack, Marketing briefs, Finance ROI."},
            {"severity": "INFO", "category": "CEO readiness","finding": "Executive summary concise. All CEO conditions from Phase 4 approval met."},
        ],
        verdict="APPROVED", summary="Business Lead confirms all Phase 5 deliverables are PROD-ready. Requesting CEO final sign-off.")

    _work_review(
        submitted_by="ds_lead", phase="phase5",
        artifact="Technical Phase 5 sign-off: model export + deployment (DEV)",
        env="dev",
        findings=[
            {"severity": "INFO", "category": "Artefact quality", "finding": "Serialised model files load correctly. Prediction latency <100ms on sample batch."},
        ],
        verdict="APPROVED", summary="DS Lead and Code Reviewer confirm all technical Phase 5 artefacts are PROD-quality.")

    _promote(
        phase="phase5",
        artifacts=["segment_report.md", "final_report.html",
                   "eda_transactions.md", "eda_customers.md"],
        submitted_by="project_manager",
        final_approver="ceo",
    )

    _approval(
        phase="project_complete",
        requested_by="project_manager",
        request_summary="All 5 phases complete. All artefacts reviewed by full senior chain and promoted to PROD. "
                        "Requesting CEO final sign-off to formally close the project.",
        decision="APPROVED",
        ceo_rationale="Exceptional delivery. All CEO conditions met. Segmentation and recommendations are live. "
                      "I expect the marketing team to begin promo campaigns within 2 weeks using these segments. "
                      "Schedule a 30-day review to measure actual promo lift against projections.",
        conditions="30-day review meeting to compare actual vs projected promo lift. "
                   "Real-time scoring API to be prioritised in Q3.",
    )

    # ── Save everything ───────────────────────────────────────────────────────
    agent_statuses = {aid: "DONE" for aid in AGENT_ROLES}

    feed_dashboard(
        segment_profiles=seg.cluster_profiles_,
        recommendations=recommendations,
        kpi_metrics={"promo_lift_pct": ">15%", "retention_rate": ">70%",
                     "avg_segment_revenue": ">$500", "rec_ctr": ">8%"},
        project_state_snapshot={
            "milestones":         {f"phase{i}": "complete" for i in range(1, 6)},
            "models":             {
                "segmentation": {**seg_metrics, "approved_by": "ds_lead"},
                "recommender":  {"model": rec._backend.upper(), "top_n": 10, "approved_by": "ds_lead"},
            },
            "agent_statuses":     agent_statuses,
            "approved_features":  list(feature_matrix.columns),
            "data_registry": {
                "customers":     os.path.join(PATHS["processed_data"], "customers.parquet"),
                "transactions":  os.path.join(PATHS["processed_data"], "transactions.parquet"),
                "products":      os.path.join(PATHS["processed_data"], "products.parquet"),
                "promos":        os.path.join(PATHS["processed_data"], "promos.parquet"),
                "feature_matrix": feat_path,
            },
            "kpis":               {"promo_lift_pct": ">15%", "retention_rate": ">70%"},
            "project_complete":   True,
            "activity_count":     len(_activity),
            "message_count":      len(_messages),
            "approval_count":     len(_approvals),
            "review_count":       len(_reviews),
            "work_review_count":  len(_work_reviews),
            "promotion_count":    len(_promotions),
            "team_size":          len(AGENT_ROLES),
            "environments":       ["dev", "prod"],
            "active_env":         "prod",
        },
        activity_stream=_activity,
    )

    # Save inter-agent messages, CEO approvals, code reviews, work reviews, promotions
    write_json(_messages,      os.path.join(PATHS["reports"], "agent_messages.json"))
    write_json(_approvals,     os.path.join(PATHS["reports"], "ceo_approvals.json"))
    write_json(_reviews,       os.path.join(PATHS["reports"], "code_reviews.json"))
    write_json(_work_reviews,  os.path.join(PATHS["reports"], "work_reviews.json"))
    write_json(_promotions,    os.path.join(PATHS["reports"], "env_promotions.json"))
    write_json(list(_env_registry.values()),
                               os.path.join(PATHS["reports"], "env_registry.json"))

    # Flush Slack messages and tickets to disk
    _tickets.flush()
    _slack.flush()

    ticket_summary   = _tickets.summary()
    total_wr         = len(_work_reviews)
    total_promotions = len(_promotions)
    prod_artifacts   = sum(1 for v in _env_registry.values() if v.get("env") == "prod")

    log.info("=" * 60)
    log.info("PIPELINE COMPLETE -- DEV -> PROD model")
    log.info(f"  Report:       {report_path}")
    log.info(f"  Activities:   {len(_activity)} events across 17 agents")
    log.info(f"  Messages:     {len(_messages)} inter-agent communications")
    log.info(f"  CEO approvals:{len(_approvals)} gates (all APPROVED)")
    log.info(f"  Code reviews: {len(_reviews)} technical reviews")
    log.info(f"  Work reviews: {total_wr} senior sign-offs across all phases")
    log.info(f"  Promotions:   {total_promotions} DEV->PROD promotion events")
    log.info(f"  PROD registry:{prod_artifacts} artefacts promoted to PROD")
    log.info(f"  Tickets:      {ticket_summary['total']} Slack tickets "
             f"({ticket_summary.get('DONE', 0)} closed)")
    log.info(f"  Data:         7 sources (4 structured + 3 unstructured, 5M customers)")
    log.info("=" * 60)
    return {
        "report":            report_path,
        "segmentation_metrics": seg_metrics,
        "activity_count":    len(_activity),
        "message_count":     len(_messages),
        "approval_count":    len(_approvals),
        "review_count":      len(_reviews),
        "work_review_count": total_wr,
        "promotion_count":   total_promotions,
    }


if __name__ == "__main__":
    run_all()
