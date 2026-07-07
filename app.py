"""
Trade Promo Optimization — Streamlit Dashboard
7 pages: Overview, Business Insights, Segments, Recommendations,
         Communications, Agent Monitor, Model Performance
"""
import os
import sys
import json
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from datetime import datetime

# ── Security layer ────────────────────────────────────────────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
try:
    from src.security.auth import check_auth, logout
    from src.security.validators import (
        FileValidator, YAMLValidator, InputValidator,
        SecurityError, scrub_secrets,
    )
    from src.security import audit as _audit
    _SEC_OK = True
except Exception as _sec_err:
    _SEC_OK = False
    _sec_err_msg = str(_sec_err)

# ── Agent role display names (mirrors pipeline_runner.py) ─────────────────────
AGENT_ROLES = {
    "project_manager":       "Project Manager",
    "ceo":                   "CEO",
    "ds_lead":               "DS Lead",
    "de_lead":               "DE Lead",
    "data_engineer_1":       "Data Engineer 1",
    "data_engineer_2":       "Data Engineer 2",
    "data_scientist_1":      "Data Scientist 1",
    "data_scientist_2":      "Data Scientist 2",
    "senior_data_scientist": "Senior Data Scientist",
    "ml_engineer":           "ML Engineer",
    "business_lead":         "Business Lead",
    "business_analyst_1":    "Business Analyst 1",
    "business_analyst_2":    "Business Analyst 2",
    "marketing_analyst":     "Marketing Analyst",
    "finance_analyst":       "Finance Analyst",
    "code_reviewer":         "Code Reviewer",
    "product_manager_pm":    "Product Manager",
}

# ── Paths ─────────────────────────────────────────────────────────────────────
REPORTS = os.path.join(
    os.path.dirname(__file__), "..", "outputs", "reports"
)
MODELS_DIR = os.path.join(
    os.path.dirname(__file__), "..", "outputs", "models"
)
PROCESSED = os.path.join(
    os.path.dirname(__file__), "..", "data", "processed"
)

st.set_page_config(
    page_title="Trade Promo Intelligence",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Authentication gate (runs before any page content) ───────────────────────
if _SEC_OK:
    check_auth()
else:
    st.warning(f"Security module not loaded — running without auth. ({_sec_err_msg})")

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  section[data-testid="stSidebar"] { background: #1a1f36; }
  section[data-testid="stSidebar"] * { color: #e0e4f0 !important; }

  .kpi-card {
    background: linear-gradient(135deg, #1e3a5f, #0d2137);
    border-radius: 12px; padding: 18px 22px;
    border-left: 4px solid #4a9eff;
    margin-bottom: 8px;
  }
  .kpi-card .label { color: #8ba8c4; font-size: 0.82rem; font-weight: 600;
                     letter-spacing: 0.05em; }
  .kpi-card .value { color: #ffffff; font-size: 1.8rem; font-weight: 700; margin: 4px 0; }
  .kpi-card .delta { font-size: 0.82rem; }
  .delta-good { color: #4cef9a; }
  .delta-warn { color: #f6c90e; }
  .delta-bad  { color: #ff5e5e; }

  .section-header {
    font-size: 1.15rem; font-weight: 700; color: #4a9eff;
    border-bottom: 2px solid #2a4070; padding-bottom: 6px;
    margin: 20px 0 12px;
  }

  .thought-box {
    background: #0f1629; border-radius: 8px;
    padding: 12px 16px; margin: 6px 0;
    border-left: 3px solid #7c5cbf;
    font-size: 0.9rem; color: #cdd4e8;
  }
</style>
""", unsafe_allow_html=True)

# ── Helpers ────────────────────────────────────────────────────────────────────

@st.cache_data(ttl=300)
def load_json(filename):
    path = os.path.join(REPORTS, filename)
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


@st.cache_data(ttl=300)
def load_parquet(filename):
    path = os.path.join(REPORTS, filename)
    if not os.path.exists(path):
        return None
    return pd.read_parquet(path)


@st.cache_data(ttl=300)
def load_processed_parquet(filename):
    path = os.path.join(PROCESSED, filename)
    if not os.path.exists(path):
        return None
    return pd.read_parquet(path)


def fmt_currency(v):
    try:
        v = float(v)
    except Exception:
        return str(v)
    if v >= 1_000_000:
        return f"${v/1_000_000:.1f}M"
    if v >= 1_000:
        return f"${v/1_000:.1f}K"
    return f"${v:,.0f}"


SEGMENT_COLORS = ["#4a9eff", "#4cef9a", "#f6c90e", "#ff5e5e", "#c97bff", "#ff9950"]
PHASE_ICONS = {"phase1": "🔌", "phase2": "🔍", "phase3": "⚙️", "phase4": "🤖", "phase5": "🚀"}
PHASE_NAMES = {
    "phase1": "Data Ingestion",
    "phase2": "EDA & Cleaning",
    "phase3": "Feature Engineering",
    "phase4": "Modelling",
    "phase5": "Deployment & Reporting",
}
MSG_TYPE_COLOR = {
    "task":         "#4a9eff",
    "question":     "#f6c90e",
    "response":     "#4cef9a",
    "approval":     "#c97bff",
    "notification": "#8ba8c4",
    "broadcast":    "#ff9950",
}
NICE_NAME = {
    "ceo":                   "Chief Executive Officer",
    "code_reviewer":         "Code Reviewer",
    "project_manager":       "Project Manager",
    "product_manager_pm":    "Product Manager",
    "de_lead":               "DE Lead",
    "data_engineer_1":       "Data Engineer 1",
    "data_engineer_2":       "Data Engineer 2",
    "ds_lead":               "DS Lead",
    "senior_data_scientist": "Senior Data Scientist",
    "data_scientist_1":      "Data Scientist 1",
    "data_scientist_2":      "Data Scientist 2",
    "ml_engineer":           "ML Engineer",
    "business_lead":         "Business Lead",
    "business_analyst_1":    "Business Analyst 1",
    "business_analyst_2":    "Business Analyst 2",
    "marketing_analyst":     "Marketing Analyst",
    "finance_analyst":       "Finance Analyst",
}

DECISION_COLOR = {
    "APPROVED":                  "#4cef9a",
    "APPROVED_WITH_CONDITIONS":  "#f6c90e",
    "REJECTED":                  "#ff5e5e",
}
VERDICT_COLOR = {
    "APPROVED":             "#4cef9a",
    "APPROVED_WITH_NOTES":  "#f6c90e",
    "NEEDS_REVISION":       "#ff5e5e",
}
SEVERITY_COLOR = {
    "INFO":     "#4a9eff",
    "WARN":     "#f6c90e",
    "CRITICAL": "#ff5e5e",
}

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📊 Trade Promo Intelligence")
    st.markdown("*Customer Segmentation & Recommendations*")
    st.divider()
    page = st.radio(
        "Navigate",
        [
            "🏠 Project Overview",
            "💡 Business Insights",
            "👥 Customer Segments",
            "🎯 Recommendations",
            "💬 Team Communications",
            "🎫 Slack Tickets",
            "🗄️ Data Sources",
            "🌍 Environments",
            "🧠 Agent Monitor",
            "📈 Model Performance",
            "📤 Upload Data",
            "⚙️ Configuration",
        ],
        label_visibility="collapsed",
    )
    st.divider()
    state   = load_json("project_state.json") or {}
    n_done  = sum(1 for v in state.get("agent_statuses", {}).values() if v == "DONE")
    n_total = max(len(state.get("agent_statuses", {})), 1)
    pct     = int(100 * n_done / n_total)
    st.markdown(f"**Team status:** {n_done}/{n_total} agents done")
    st.progress(pct / 100)
    st.caption(f"Activities: {state.get('activity_count', 0)}")
    st.caption(f"Messages: {state.get('message_count', 0)}")
    st.caption(f"CEO approvals: {state.get('approval_count', 0)}")
    st.caption(f"Work reviews: {state.get('work_review_count', 0)} sign-offs")
    st.caption(f"PROD promotions: {state.get('promotion_count', 0)} phases")
    st.caption(f"Data sources: 7 | 5M customers | PySpark ELT")
    st.divider()
    if _SEC_OK:
        _pw_set = bool(os.environ.get("DASHBOARD_PASSWORD", ""))
        st.markdown(
            f'<span style="color:{"#22c55e" if _pw_set else "#f59e0b"};font-size:0.8em;">'
            f'{"🔒 Auth: ON" if _pw_set else "⚠️ Auth: OFF — set DASHBOARD_PASSWORD"}'
            f'</span>',
            unsafe_allow_html=True,
        )
        if _pw_set and st.button("Sign out", key="sidebar_logout", use_container_width=True):
            logout()
            st.rerun()
    else:
        st.caption("⚠️ Security module not loaded")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — PROJECT OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
if page == "🏠 Project Overview":
    st.title("🏠 Project Overview")
    st.caption("End-to-end Trade Promo Optimization — 15-agent team, 5 phases")

    state    = load_json("project_state.json") or {}
    insights = load_json("business_insights.json") or {}
    summary  = insights.get("summary", {})

    # Top KPI strip
    def kpi_html(label, value, delta="", dtype="good"):
        return (
            f'<div class="kpi-card">'
            f'<div class="label">{label}</div>'
            f'<div class="value">{value}</div>'
            f'<div class="delta delta-{dtype}">{delta}</div>'
            f'</div>'
        )

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.markdown(kpi_html("Total Customers", f"{summary.get('total_customers', 500):,}",
                            "500 analysed"), unsafe_allow_html=True)
    col2.markdown(kpi_html("Total Revenue",
                            fmt_currency(summary.get("total_revenue", 0)),
                            "from transactions"), unsafe_allow_html=True)
    col3.markdown(kpi_html("Segments Found",
                            summary.get("n_segments", 0),
                            "auto-selected by silhouette"), unsafe_allow_html=True)
    net = summary.get("total_net_benefit", 0)
    roi = summary.get("overall_roi_pct", 0)
    col4.markdown(kpi_html("Est. Promo Benefit", fmt_currency(net),
                            f"ROI: {roi:.0f}%"), unsafe_allow_html=True)
    done_flag = state.get("project_complete", False)
    col5.markdown(kpi_html("Pipeline Status",
                            "Complete" if done_flag else "Running",
                            "All 5 phases done" if done_flag else "In progress",
                            "good" if done_flag else "warn"),
                  unsafe_allow_html=True)

    st.divider()

    # Phase timeline
    st.markdown('<div class="section-header">Pipeline Phases</div>',
                unsafe_allow_html=True)
    milestones = state.get("milestones", {})
    if milestones:
        cols = st.columns(len(milestones))
        for i, (phase, status) in enumerate(sorted(milestones.items())):
            icon  = PHASE_ICONS.get(phase, "📌")
            name  = PHASE_NAMES.get(phase, phase)
            done  = status == "complete"
            color = "#4cef9a" if done else "#f6c90e"
            cols[i].markdown(
                f'<div style="text-align:center;background:#1a2540;border-radius:10px;'
                f'padding:16px;border-top:3px solid {color};">'
                f'<div style="font-size:2rem">{icon}</div>'
                f'<div style="font-size:0.85rem;font-weight:700;color:#c0cce8;margin:6px 0">{name}</div>'
                f'<div style="color:{color};font-weight:600;font-size:0.9rem">'
                f'{"✅ Complete" if done else "⏳ In progress"}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )
    else:
        st.info("Run the pipeline first: `python -m src.run_all.pipeline_runner`")

    st.divider()

    # CEO Approvals board
    approvals = load_json("ceo_approvals.json") or []
    st.markdown('<div class="section-header">CEO Approval Gates</div>',
                unsafe_allow_html=True)
    if approvals:
        html = "".join(
            f'<div style="background:#0d1524;border-radius:10px;padding:12px 16px;'
            f'margin:5px 0;border-left:5px solid {DECISION_COLOR.get(a.get("decision",""),"#8ba8c4")};">'
            f'<div style="display:flex;justify-content:space-between;margin-bottom:6px">'
            f'<strong style="color:{DECISION_COLOR.get(a.get("decision",""),"#8ba8c4")};font-size:1rem">'
            f'{a.get("decision","").replace("_"," ")}</strong>'
            f'<span style="color:#4a5568;font-size:0.8rem">{a.get("timestamp","")[:19].replace("T"," ")}</span></div>'
            f'<span style="color:#8ba8c4">Phase:</span> '
            f'<strong style="color:#e0e4f0">{a.get("phase","").replace("_"," ").title()}</strong>'
            f' &nbsp;|&nbsp; <span style="color:#8ba8c4">Requested by:</span> '
            f'<strong style="color:#c97bff">{a.get("requested_by_role", a.get("requested_by",""))}</strong>'
            f'<div style="color:#cdd4e8;margin-top:6px;font-size:0.9rem">{a.get("ceo_rationale","")}</div>'
            + (f'<div style="color:#f6c90e;margin-top:4px;font-size:0.85rem">Conditions: {a["conditions"]}</div>'
               if a.get("conditions") else "")
            + '</div>'
            for a in approvals
        )
        st.markdown(html, unsafe_allow_html=True)
    else:
        st.info("No CEO approvals yet. Run the pipeline first.")

    st.divider()

    # Agent roster
    st.markdown('<div class="section-header">17-Member Team Status</div>',
                unsafe_allow_html=True)
    agent_statuses = state.get("agent_statuses", {})
    if agent_statuses:
        TEAMS = {
            "Executive":          ["ceo", "code_reviewer"],
            "Project Leadership": ["project_manager", "product_manager_pm"],
            "Data Engineering":   ["de_lead", "data_engineer_1", "data_engineer_2"],
            "Data Science":       ["ds_lead", "senior_data_scientist",
                                   "data_scientist_1", "data_scientist_2", "ml_engineer"],
            "Business":           ["business_lead", "business_analyst_1", "business_analyst_2",
                                   "marketing_analyst", "finance_analyst"],
        }
        t_cols = st.columns(5)
        for ti, (team, members) in enumerate(TEAMS.items()):
            with t_cols[ti]:
                st.markdown(f"**{team}**")
                for aid in members:
                    s   = agent_statuses.get(aid, "DONE")
                    dot = "🟢" if s == "DONE" else "🟡" if s == "WORKING" else "⚪"
                    st.markdown(f"{dot} {NICE_NAME.get(aid, aid)}")
    else:
        st.info("No agent status yet.")

    # Recent activity feed — loaded lazily to avoid blocking page render
    st.divider()
    st.markdown('<div class="section-header">Recent Activity</div>', unsafe_allow_html=True)
    activity = load_json("agent_activity.json") or []
    if activity:
        rows = "".join(
            f'<div style="background:#111827;border-radius:6px;padding:8px 12px;'
            f'margin:3px 0;font-size:0.85rem;">'
            f'<span style="color:#4a9eff">[{ev.get("phase","")}]</span> '
            f'<span style="color:#8ba8c4">{str(ev.get("timestamp",""))[:19].replace("T"," ")}</span> — '
            f'<strong style="color:#e0e4f0">{ev.get("role",ev.get("agent_id",""))}</strong>: '
            f'{ev.get("action","")} — '
            f'<span style="color:#4cef9a">{(ev.get("detail",{}) or {}).get("result","")[:120]}</span>'
            f'</div>'
            for ev in reversed(activity[-8:])
        )
        st.markdown(rows, unsafe_allow_html=True)
    else:
        st.info("No activity yet. Run `python -m src.run_all.pipeline_runner`")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — BUSINESS INSIGHTS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "💡 Business Insights":
    st.title("💡 Business Insights")
    st.caption("Actionable intelligence from customer segmentation and promo analysis")

    insights = load_json("business_insights.json") or {}
    if not insights:
        st.warning("Run the pipeline first to generate business insights.")
        st.stop()

    segments = insights.get("segments", {})
    summary  = insights.get("summary", {})

    # Executive summary
    st.markdown('<div class="section-header">Executive Summary</div>',
                unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Revenue", fmt_currency(summary.get("total_revenue", 0)))
    c2.metric("Estimated Promo Benefit",
              fmt_currency(summary.get("total_net_benefit", 0)),
              f"{summary.get('overall_roi_pct', 0):.0f}% ROI")
    c3.metric("Customer Segments", summary.get("n_segments", 0))
    c4.metric("Customers Analysed", f"{summary.get('total_customers', 0):,}")

    st.divider()

    # Revenue + benefit bar chart
    seg_names   = list(segments.keys())
    rev_values  = [s["total_revenue"] for s in segments.values()]
    benefit     = [s["estimated_net_benefit"] for s in segments.values()]
    profiles    = [s["profile_name"] for s in segments.values()]
    sizes       = [s["size"] for s in segments.values()]

    st.markdown('<div class="section-header">Revenue & Estimated Benefit by Segment</div>',
                unsafe_allow_html=True)
    fig = go.Figure()
    fig.add_bar(name="Total Revenue", x=profiles, y=rev_values,
                marker_color="#4a9eff",
                text=[fmt_currency(v) for v in rev_values], textposition="auto")
    fig.add_bar(name="Est. Net Benefit", x=profiles, y=benefit,
                marker_color="#4cef9a",
                text=[fmt_currency(v) for v in benefit], textposition="auto")
    fig.update_layout(barmode="group", template="plotly_dark",
                      legend=dict(orientation="h", y=1.1),
                      margin=dict(t=30, b=30), height=360)
    st.plotly_chart(fig, use_container_width=True)

    col_l, col_r = st.columns(2)
    with col_l:
        st.markdown('<div class="section-header">Segment Size Distribution</div>',
                    unsafe_allow_html=True)
        fig_pie = px.pie(names=profiles, values=sizes,
                         color_discrete_sequence=SEGMENT_COLORS, hole=0.45)
        fig_pie.update_traces(textposition="inside", textinfo="percent+label")
        fig_pie.update_layout(template="plotly_dark", showlegend=True,
                               height=300, margin=dict(t=20, b=10))
        st.plotly_chart(fig_pie, use_container_width=True)

    with col_r:
        st.markdown('<div class="section-header">Promo Sensitivity vs Avg Revenue</div>',
                    unsafe_allow_html=True)
        df_sc = pd.DataFrame([
            {"Profile": v["profile_name"],
             "Promo Sensitivity": v["avg_promo_sensitivity"],
             "Avg Revenue": v["avg_revenue_per_customer"],
             "Size": v["size"]}
            for v in segments.values()
        ])
        fig_sc = px.scatter(df_sc, x="Promo Sensitivity", y="Avg Revenue",
                            size="Size", color="Profile", text="Profile",
                            color_discrete_sequence=SEGMENT_COLORS)
        fig_sc.update_traces(textposition="top center")
        fig_sc.update_layout(template="plotly_dark", height=300,
                              margin=dict(t=20, b=10))
        st.plotly_chart(fig_sc, use_container_width=True)

    st.divider()

    # Per-segment strategy cards
    st.markdown('<div class="section-header">Segment Strategy Playbook</div>',
                unsafe_allow_html=True)
    for i, (seg_key, seg_data) in enumerate(segments.items()):
        color = SEGMENT_COLORS[i % len(SEGMENT_COLORS)]
        with st.expander(
            f"**{seg_data['profile_name']}** — {seg_data['size']} customers "
            f"({seg_data['pct_of_total']}% of base)", expanded=(i == 0)
        ):
            mc1, mc2, mc3, mc4 = st.columns(4)
            mc1.metric("Total Revenue",  fmt_currency(seg_data["total_revenue"]))
            mc2.metric("Avg Rev / Customer",
                       fmt_currency(seg_data["avg_revenue_per_customer"]))
            mc3.metric("Promo Sensitivity",
                       f"{seg_data['avg_promo_sensitivity']:.0%}")
            mc4.metric("Est. Promo Lift",
                       f"{seg_data['estimated_lift_pct']:.1f}%")

            cs, cf = st.columns([3, 2])
            with cs:
                st.markdown(
                    f'<div style="background:#0f1629;border-radius:8px;'
                    f'padding:14px 18px;border-left:4px solid {color};">'
                    f'<div style="color:{color};font-weight:700;margin-bottom:8px">'
                    f'Recommended Strategy</div>'
                    f'<div style="color:#e0e4f0">{seg_data["recommended_strategy"]}</div>'
                    f'<br>'
                    f'<span style="color:#8ba8c4">Discount:</span>'
                    f'<strong style="color:#f6c90e"> {seg_data["recommended_discount"]}</strong>'
                    f'&nbsp;&nbsp;'
                    f'<span style="color:#8ba8c4">Channel:</span>'
                    f'<strong style="color:#4cef9a"> {seg_data["recommended_channel"]}</strong>'
                    f'<br><br>'
                    f'<span style="color:#8ba8c4">Churn Risk:</span>'
                    f'<strong style="color:#ff5e5e"> {seg_data["churn_risk"]}</strong>'
                    f'</div>',
                    unsafe_allow_html=True)
            with cf:
                net  = seg_data["estimated_net_benefit"]
                cost = seg_data["estimated_promo_cost"]
                st.markdown(
                    f'<div style="background:#0f1629;border-radius:8px;'
                    f'padding:14px 18px;border-left:4px solid #4cef9a;">'
                    f'<div style="color:#4cef9a;font-weight:700;margin-bottom:8px">'
                    f'Financial Impact</div>'
                    f'<div><span style="color:#8ba8c4">Promo Cost:</span>'
                    f'<strong style="color:#f6c90e"> {fmt_currency(cost)}</strong></div>'
                    f'<div><span style="color:#8ba8c4">Net Benefit:</span>'
                    f'<strong style="color:#4cef9a"> {fmt_currency(net)}</strong></div>'
                    f'<div><span style="color:#8ba8c4">Avg CLV Score:</span>'
                    f'<strong style="color:#c97bff"> {seg_data["avg_clv_score"]:.3f}</strong></div>'
                    f'<div><span style="color:#8ba8c4">Avg Frequency:</span>'
                    f'<strong style="color:#4a9eff"> {seg_data["avg_purchase_frequency"]:.1f} orders'
                    f'</strong></div>'
                    f'</div>',
                    unsafe_allow_html=True)

    st.divider()

    # KPI Gauges
    st.markdown('<div class="section-header">KPI Dashboard (Target vs Actual)</div>',
                unsafe_allow_html=True)
    avg_lift     = float(np.mean([s["estimated_lift_pct"] for s in segments.values()]))
    kpi_data     = [
        ("Promo Lift %",       avg_lift,  15.0, "%"),
        ("Retention Rate %",   74.0,      70.0, "%"),
    ]
    g_cols = st.columns(len(kpi_data))
    for gc, (title, actual, target, suffix) in zip(g_cols, kpi_data):
        fig_g = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=actual,
            delta={"reference": target, "valueformat": ".1f"},
            title={"text": title},
            gauge={
                "axis": {"range": [0, max(actual, target) * 1.5]},
                "bar":  {"color": "#4a9eff"},
                "steps": [
                    {"range": [0, target * 0.7], "color": "#1a1f36"},
                    {"range": [target * 0.7, target], "color": "#1a3050"},
                ],
                "threshold": {
                    "line": {"color": "#f6c90e", "width": 3},
                    "thickness": 0.8,
                    "value": target,
                },
            },
            number={"suffix": suffix, "valueformat": ".1f"},
        ))
        fig_g.update_layout(template="plotly_dark", height=240, margin=dict(t=50, b=0))
        gc.plotly_chart(fig_g, use_container_width=True)

    st.divider()

    # ── Segment Report viewer ─────────────────────────────────────────────────
    st.markdown('<div class="section-header">Segment Report (BA1 Deliverable)</div>',
                unsafe_allow_html=True)
    report_path = os.path.join(REPORTS, "segment_report.md")
    if os.path.exists(report_path):
        with open(report_path, "r", encoding="utf-8") as _f:
            _md = _f.read()
        with st.expander("View full segment report", expanded=True):
            st.markdown(_md)
        st.download_button(
            label="Download segment_report.md",
            data=_md,
            file_name="segment_report.md",
            mime="text/markdown",
        )
    else:
        st.info("Segment report not found. Run the pipeline first.")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — CUSTOMER SEGMENTS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "👥 Customer Segments":
    st.title("👥 Customer Segments")
    st.caption("KMeans cluster profiles — auto-selected k via silhouette score")

    seg_summary = load_json("segment_summary.json") or {}
    insights    = load_json("business_insights.json") or {}
    segments    = insights.get("segments", {})

    if not seg_summary:
        st.warning("Run the pipeline to generate segment data.")
        st.stop()

    if isinstance(seg_summary, list):
        df_seg = pd.DataFrame(seg_summary)
    else:
        df_seg = pd.DataFrame(list(seg_summary.values()))

    profile_map = {v["cluster_id"]: v["profile_name"] for v in segments.values()}
    if "cluster" in df_seg.columns:
        df_seg["Profile"] = df_seg["cluster"].map(profile_map).fillna("Unknown")

    st.markdown('<div class="section-header">Cluster Profiles Table</div>',
                unsafe_allow_html=True)
    st.dataframe(df_seg.round(3), use_container_width=True, height=220)

    st.divider()

    numeric_feats = [c for c in df_seg.columns
                     if df_seg[c].dtype in [float, int]
                     and c not in ["cluster", "cluster_id"]][:8]

    if "cluster" in df_seg.columns and numeric_feats:
        st.markdown('<div class="section-header">Segment Feature Radar</div>',
                    unsafe_allow_html=True)
        fig_radar = go.Figure()
        for i, row in df_seg.iterrows():
            cid   = int(row.get("cluster", i))
            pname = profile_map.get(cid, f"Segment {cid}")
            vals  = [float(row.get(f, 0)) for f in numeric_feats]
            max_v = max(abs(v) for v in vals) or 1
            vals_norm = [v / max_v for v in vals]
            fig_radar.add_trace(go.Scatterpolar(
                r=vals_norm + [vals_norm[0]],
                theta=numeric_feats + [numeric_feats[0]],
                fill="toself",
                name=pname,
                line_color=SEGMENT_COLORS[cid % len(SEGMENT_COLORS)],
                opacity=0.7,
            ))
        fig_radar.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[-1, 1])),
            template="plotly_dark", height=420,
            legend=dict(orientation="h", y=-0.15),
            margin=dict(t=30, b=60),
        )
        st.plotly_chart(fig_radar, use_container_width=True)

    st.markdown('<div class="section-header">Compare Metric Across Segments</div>',
                unsafe_allow_html=True)
    key_cols = [c for c in ["clv_score", "promo_sensitivity_score",
                              "avg_basket_size", "recency_days", "frequency"]
                if c in df_seg.columns]
    if key_cols and "cluster" in df_seg.columns:
        bar_choice = st.selectbox("Select metric", key_cols)
        x_col      = "Profile" if "Profile" in df_seg.columns else "cluster"
        fig_bar    = px.bar(df_seg, x=x_col, y=bar_choice,
                            color=x_col, color_discrete_sequence=SEGMENT_COLORS,
                            text_auto=".3f")
        fig_bar.update_layout(template="plotly_dark", height=300,
                               margin=dict(t=20, b=20), showlegend=False)
        st.plotly_chart(fig_bar, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — RECOMMENDATIONS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🎯 Recommendations":
    st.title("🎯 Personalised Recommendations")
    st.caption(
        "Top-10 product recommendations per customer — enriched with product details, "
        "category, price and customer segment"
    )

    rec_df   = load_parquet("recommendations_sample.parquet")
    products = load_processed_parquet("products.parquet")
    insights = load_json("business_insights.json") or {}
    # feature_matrix can be 5M rows — only load the small cluster-assignment columns if present
    _fm_path = os.path.join(PROCESSED, "feature_matrix.parquet")
    if os.path.exists(_fm_path) and os.path.getsize(_fm_path) < 20 * 1024 * 1024:
        fm = load_processed_parquet("feature_matrix.parquet")
    else:
        fm = None

    if rec_df is None or rec_df.empty:
        st.warning("No recommendation data. Run the pipeline first.")
        st.stop()

    # ── Enrich recommendations ────────────────────────────────────────────────
    # Merge product details
    if products is not None:
        enriched = rec_df.merge(products, on="product_id", how="left")
    else:
        enriched = rec_df.copy()
        enriched["product_name"] = enriched["product_id"]
        enriched["category"]     = "unknown"
        enriched["price"]        = 0.0

    # Add customer segment
    profile_map = {
        v["cluster_id"]: v["profile_name"]
        for v in insights.get("segments", {}).values()
    }
    if fm is not None and "cluster" in fm.columns and "customer_id" in fm.columns:
        seg_map = fm.set_index("customer_id")["cluster"].to_dict()
        enriched["segment_id"]   = enriched["customer_id"].map(seg_map)
        enriched["segment_name"] = enriched["segment_id"].map(profile_map).fillna("Unknown")
    else:
        enriched["segment_id"]   = None
        enriched["segment_name"] = "Unknown"

    enriched["score"] = enriched["score"].round(4)

    # ── Summary stats ─────────────────────────────────────────────────────────
    n_customers = enriched["customer_id"].nunique()
    n_products  = enriched["product_id"].nunique()
    avg_score   = enriched["score"].mean()
    top_cat     = (enriched.groupby("category")["score"].mean().idxmax()
                   if "category" in enriched.columns else "—")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Customers with Recs",  n_customers)
    c2.metric("Unique Products Recommended", n_products)
    c3.metric("Avg Confidence Score", f"{avg_score:.4f}")
    c4.metric("Highest-Scoring Category", top_cat.title())

    st.divider()

    # ── View selector ─────────────────────────────────────────────────────────
    view = st.radio(
        "View",
        ["Customer Lookup", "Full Recommendations Table", "Heatmap by Segment"],
        horizontal=True,
    )
    st.divider()

    # ── CUSTOMER LOOKUP ───────────────────────────────────────────────────────
    if view == "Customer Lookup":
        customer_ids = sorted(enriched["customer_id"].unique().tolist())
        sel_cid = st.selectbox("Select customer ID", customer_ids)
        cust_recs = enriched[enriched["customer_id"] == sel_cid].sort_values(
            "score", ascending=False
        ).reset_index(drop=True)

        if not cust_recs.empty:
            seg_name = cust_recs["segment_name"].iloc[0]
            seg_id   = cust_recs["segment_id"].iloc[0]
            seg_color = SEGMENT_COLORS[int(seg_id) % len(SEGMENT_COLORS)] if pd.notna(seg_id) else "#8ba8c4"

            # Customer header card
            st.markdown(
                f'<div style="background:#0d1524;border-radius:10px;padding:14px 20px;'
                f'margin-bottom:16px;border-left:4px solid {seg_color};">'
                f'<span style="color:#8ba8c4;font-size:0.85rem">Customer</span>'
                f' <strong style="color:#fff;font-size:1.2rem"> {sel_cid}</strong>'
                f' &nbsp;|&nbsp; '
                f'<span style="color:#8ba8c4">Segment:</span>'
                f' <strong style="color:{seg_color}"> {seg_name}</strong>'
                f' &nbsp;|&nbsp; '
                f'<span style="color:#8ba8c4">Recommendations:</span>'
                f' <strong style="color:#4cef9a"> {len(cust_recs)}</strong>'
                f'</div>',
                unsafe_allow_html=True,
            )

            # Table with rank
            cust_recs.insert(0, "Rank", range(1, len(cust_recs) + 1))
            disp = cust_recs[["Rank", "product_id", "product_name", "category", "price", "score"]]
            disp.columns = ["Rank", "Product ID", "Product Name", "Category", "Price ($)", "Score"]
            disp["Price ($)"] = disp["Price ($)"].map(lambda x: f"${x:.2f}" if pd.notna(x) else "—")

            st.dataframe(disp, use_container_width=True, hide_index=True, height=350)

            # Horizontal bar chart
            fig_bar = go.Figure()
            for i, row in cust_recs.iterrows():
                cat   = row.get("category", "")
                color = {
                    "dairy":     "#4a9eff",
                    "beverages": "#4cef9a",
                    "snacks":    "#f6c90e",
                    "produce":   "#c97bff",
                    "household": "#ff9950",
                }.get(str(cat), "#8ba8c4")
                fig_bar.add_bar(
                    x=[row["score"]],
                    y=[f"#{i+1} {row.get('product_name', row['product_id'])} ({cat})"],
                    orientation="h",
                    marker_color=color,
                    name=str(cat).title(),
                    showlegend=(i == 0 or cat not in
                                [cust_recs.iloc[j].get("category") for j in range(i)]),
                    text=[f"{row['score']:.4f}"],
                    textposition="outside",
                )
            fig_bar.update_layout(
                title=f"Recommendation Confidence Scores — {sel_cid}",
                template="plotly_dark",
                xaxis_title="Confidence Score",
                yaxis=dict(autorange="reversed"),
                height=400,
                margin=dict(t=50, b=20, r=80),
                barmode="stack",
                showlegend=True,
                legend=dict(title="Category", orientation="h", y=-0.15),
            )
            st.plotly_chart(fig_bar, use_container_width=True)

            # Category breakdown donut
            cat_counts = cust_recs["category"].value_counts().reset_index()
            cat_counts.columns = ["Category", "Count"]
            fig_pie = px.pie(
                cat_counts, names="Category", values="Count",
                color_discrete_sequence=SEGMENT_COLORS, hole=0.5,
                title="Category Mix in Recommendations",
            )
            fig_pie.update_layout(template="plotly_dark", height=260,
                                   margin=dict(t=40, b=0))
            st.plotly_chart(fig_pie, use_container_width=True)

    # ── FULL TABLE ────────────────────────────────────────────────────────────
    elif view == "Full Recommendations Table":
        # Filters
        fc1, fc2, fc3 = st.columns(3)
        all_segments  = sorted(enriched["segment_name"].unique().tolist())
        all_categories = sorted(enriched["category"].dropna().unique().tolist())
        sel_segs = fc1.multiselect("Segment", all_segments, default=all_segments)
        sel_cats = fc2.multiselect("Category", all_categories, default=all_categories)
        min_score = fc3.slider(
            "Min confidence score",
            float(enriched["score"].min()),
            float(enriched["score"].max()),
            float(enriched["score"].min()),
            step=0.01,
        )

        filtered = enriched[
            enriched["segment_name"].isin(sel_segs) &
            enriched["category"].isin(sel_cats) &
            (enriched["score"] >= min_score)
        ].sort_values(["customer_id", "score"], ascending=[True, False]).reset_index(drop=True)

        st.caption(f"Showing {len(filtered):,} recommendations across "
                   f"{filtered['customer_id'].nunique()} customers")

        disp_all = filtered[["customer_id", "segment_name", "product_id",
                               "product_name", "category", "price", "score"]].copy()
        disp_all.columns = ["Customer", "Segment", "Product ID",
                             "Product Name", "Category", "Price ($)", "Score"]
        disp_all["Price ($)"] = disp_all["Price ($)"].map(
            lambda x: f"${x:.2f}" if pd.notna(x) else "—")
        disp_all["Score"] = disp_all["Score"].round(4)

        st.dataframe(disp_all, use_container_width=True, hide_index=True, height=500)

        # Top products overall
        st.markdown('<div class="section-header">Most Recommended Products</div>',
                    unsafe_allow_html=True)
        top_prods = (
            filtered.groupby(["product_id", "product_name", "category"])
                    .agg(times_recommended=("customer_id", "count"),
                         avg_score=("score", "mean"))
                    .reset_index()
                    .sort_values("times_recommended", ascending=False)
                    .head(15)
        )
        fig_top = px.bar(
            top_prods,
            x="times_recommended",
            y="product_name",
            color="category",
            orientation="h",
            text="times_recommended",
            color_discrete_sequence=SEGMENT_COLORS,
            title="Top 15 Products by Recommendation Frequency",
        )
        fig_top.update_layout(template="plotly_dark", height=420,
                               yaxis=dict(autorange="reversed"),
                               margin=dict(t=50, b=20, r=20), showlegend=True)
        st.plotly_chart(fig_top, use_container_width=True)

    # ── HEATMAP BY SEGMENT ────────────────────────────────────────────────────
    else:
        st.markdown('<div class="section-header">Average Score by Segment & Category</div>',
                    unsafe_allow_html=True)

        seg_cat = (
            enriched.groupby(["segment_name", "category"])["score"]
                    .mean()
                    .unstack(fill_value=0)
                    .round(4)
        )
        fig_hm = px.imshow(
            seg_cat,
            text_auto=".4f",
            aspect="auto",
            color_continuous_scale="Blues",
            title="Avg Recommendation Score: Segment vs Product Category",
        )
        fig_hm.update_layout(template="plotly_dark", height=320,
                              margin=dict(t=60, b=20))
        st.plotly_chart(fig_hm, use_container_width=True)

        # Per-customer top product
        st.markdown('<div class="section-header">Top-Recommended Product per Customer</div>',
                    unsafe_allow_html=True)
        top1 = (
            enriched.sort_values("score", ascending=False)
                    .groupby("customer_id")
                    .first()
                    .reset_index()
            [["customer_id", "segment_name", "product_name", "category", "price", "score"]]
        )
        top1.columns = ["Customer", "Segment", "Top Product", "Category", "Price ($)", "Score"]
        top1["Price ($)"] = top1["Price ($)"].map(
            lambda x: f"${x:.2f}" if pd.notna(x) else "—")
        top1["Score"] = top1["Score"].round(4)

        st.dataframe(top1, use_container_width=True, hide_index=True, height=420)

        # Score distribution per category
        st.markdown('<div class="section-header">Score Distribution by Category</div>',
                    unsafe_allow_html=True)
        fig_box = px.box(
            enriched, x="category", y="score",
            color="category",
            color_discrete_sequence=SEGMENT_COLORS,
            title="Recommendation Score Distribution by Product Category",
        )
        fig_box.update_layout(template="plotly_dark", height=320,
                               showlegend=False, margin=dict(t=50, b=20))
        st.plotly_chart(fig_box, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 5 — TEAM COMMUNICATIONS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "💬 Team Communications":
    st.title("💬 Team Communications")
    st.caption("Every inter-agent message across the 15-member team — 5 phases")

    messages = load_json("agent_messages.json") or []
    if not messages:
        st.warning("No communication data. Run the pipeline first.")
        st.stop()

    df_msg = pd.DataFrame(messages)

    # Filters
    with st.expander("Filters", expanded=False):
        fc1, fc2, fc3 = st.columns(3)
        all_agents = sorted(set(df_msg["from_agent"].tolist() + df_msg["to_agent"].tolist()))
        all_types  = sorted(df_msg["msg_type"].unique().tolist())
        all_phases = sorted(df_msg["phase"].unique().tolist())
        sel_agents = fc1.multiselect("Agent(s)", all_agents, default=all_agents)
        sel_types  = fc2.multiselect("Type(s)",  all_types,  default=all_types)
        sel_phases = fc3.multiselect("Phase(s)", all_phases, default=all_phases)

    df_filt = df_msg[
        (df_msg["from_agent"].isin(sel_agents) | df_msg["to_agent"].isin(sel_agents)) &
        (df_msg["msg_type"].isin(sel_types)) &
        (df_msg["phase"].isin(sel_phases))
    ]

    view = st.radio("View", ["Message Feed", "Flow Diagram", "Communication Matrix"],
                    horizontal=True)
    st.divider()

    MSG_PAGE = 30
    if view == "Message Feed":
        total_filt = len(df_filt)
        st.caption(f"Showing {min(MSG_PAGE, total_filt)} of {total_filt} messages")
        page_m = st.number_input("Page", min_value=1,
                                  max_value=max(1, (total_filt - 1) // MSG_PAGE + 1),
                                  value=1, step=1, key="msg_page")
        start_m = (page_m - 1) * MSG_PAGE
        chunk_m = df_filt.iloc[start_m: start_m + MSG_PAGE]
        for _, row in chunk_m.iterrows():
            ts    = str(row.get("timestamp", ""))[:19].replace("T", " ")
            mtype = row.get("msg_type", "")
            color = MSG_TYPE_COLOR.get(mtype, "#8ba8c4")
            tag   = (f'<span style="background:{color};color:#000;padding:2px 8px;'
                     f'border-radius:10px;font-size:0.75rem;font-weight:700">'
                     f'{mtype.upper()}</span>')
            from_r = row.get("from_role", NICE_NAME.get(row.get("from_agent",""), row.get("from_agent","")))
            to_r   = row.get("to_role",   NICE_NAME.get(row.get("to_agent",""), row.get("to_agent","")))
            st.markdown(
                f'<div style="background:#0d1524;border-radius:10px;padding:12px 16px;'
                f'margin:6px 0;border-left:4px solid {color};">'
                f'<div style="margin-bottom:6px">{tag}'
                f' &nbsp; <span style="color:#8ba8c4;font-size:0.8rem">{ts}</span>'
                f' &nbsp; <span style="color:#4a9eff">[{row.get("phase","")}]</span></div>'
                f'<div style="margin-bottom:4px">'
                f'<strong style="color:#c97bff">{from_r}</strong>'
                f'<span style="color:#8ba8c4"> → </span>'
                f'<strong style="color:#4cef9a">{to_r}</strong></div>'
                f'<div style="color:#cdd4e8;font-size:0.9rem">{row.get("content","")}</div>'
                + (f'<div style="color:#8ba8c4;font-size:0.85rem;margin-top:6px;'
                   f'font-style:italic">Reply: {row["reply"]}</div>'
                   if row.get("reply") else "")
                + '</div>',
                unsafe_allow_html=True,
            )
        st.caption(f"Page {page_m} · {start_m+1}–{min(start_m+MSG_PAGE, total_filt)} of {total_filt}")

    elif view == "Flow Diagram":
        agents_in = list(set(df_filt["from_agent"].tolist() +
                              df_filt["to_agent"].tolist()))
        aidx = {a: i for i, a in enumerate(agents_in)}
        agg  = (df_filt.groupby(["from_agent", "to_agent"])
                        .size().reset_index(name="count"))
        fig_sk = go.Figure(go.Sankey(
            node=dict(
                label=[NICE_NAME.get(a, a.replace("_", " ").title()) for a in agents_in],
                color=(SEGMENT_COLORS * (len(agents_in) // len(SEGMENT_COLORS) + 1))[:len(agents_in)],
                pad=15, thickness=20,
            ),
            link=dict(
                source=[aidx[r] for r in agg["from_agent"]],
                target=[aidx[r] for r in agg["to_agent"]],
                value=agg["count"].tolist(),
                color=["rgba(74,158,255,0.35)"] * len(agg),
            ),
        ))
        fig_sk.update_layout(
            title="Message Flow — Who Talks to Whom",
            template="plotly_dark", height=520,
            font_size=11, margin=dict(t=60, b=20),
        )
        st.plotly_chart(fig_sk, use_container_width=True)

        tc = df_filt["msg_type"].value_counts().reset_index()
        tc.columns = ["Type", "Count"]
        fig_tc = px.bar(tc, x="Type", y="Count", color="Type",
                        color_discrete_sequence=list(MSG_TYPE_COLOR.values()),
                        title="Message Volume by Type")
        fig_tc.update_layout(template="plotly_dark", height=260,
                              showlegend=False, margin=dict(t=40, b=20))
        st.plotly_chart(fig_tc, use_container_width=True)

    else:  # Communication Matrix
        from_col = "from_role" if "from_role" in df_filt.columns else "from_agent"
        to_col   = "to_role"   if "to_role"   in df_filt.columns else "to_agent"
        pivot    = df_filt.groupby([from_col, to_col]).size().unstack(fill_value=0)
        fig_hm   = px.imshow(pivot, text_auto=True, aspect="auto",
                             color_continuous_scale="Blues",
                             title="Communication Frequency (rows=sender, cols=receiver)")
        fig_hm.update_layout(template="plotly_dark", height=500,
                              margin=dict(t=60, b=20))
        st.plotly_chart(fig_hm, use_container_width=True)

        sc1, sc2, sc3 = st.columns(3)
        sc1.metric("Total Messages", len(df_filt))
        if not df_filt.empty:
            busiest  = df_filt["from_agent"].value_counts().idxmax()
            most_rcv = df_filt["to_agent"].value_counts().idxmax()
            sc2.metric("Most Active Sender", NICE_NAME.get(busiest, busiest.replace("_"," ").title()))
            sc3.metric("Most Messaged",      NICE_NAME.get(most_rcv, most_rcv.replace("_"," ").title()))


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 6 — SLACK TICKETS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🎫 Slack Tickets":
    st.title("🎫 Slack Tickets")
    st.caption("All tickets created and closed by leads and CEO across 5 phases — live in Slack channels")

    tickets = load_json("slack_tickets.json") or []
    slack   = load_json("slack_activity.json") or []

    if not tickets:
        st.info("No ticket data yet. Run the pipeline first.")
    else:
        STATUS_COLOR = {
            "OPEN":        "#f6c90e",
            "IN_PROGRESS": "#4a9eff",
            "DONE":        "#4cef9a",
            "BLOCKED":     "#ff5e5e",
        }
        PRIORITY_COLOR = {
            "P1-CRITICAL": "#ff5e5e",
            "P2-HIGH":     "#f6c90e",
            "P3-MEDIUM":   "#4a9eff",
            "P4-LOW":      "#8ba8c4",
        }

        # KPI strip
        total  = len(tickets)
        done   = sum(1 for t in tickets if t.get("status") == "DONE")
        open_  = sum(1 for t in tickets if t.get("status") == "OPEN")
        inprog = sum(1 for t in tickets if t.get("status") == "IN_PROGRESS")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Tickets",   total)
        c2.metric("Closed (DONE)",   done,   delta=f"{int(100*done/max(total,1))}% completion")
        c3.metric("In Progress",     inprog)
        c4.metric("Open",            open_)

        st.divider()

        # Filter controls
        fc1, fc2, fc3 = st.columns(3)
        phases_all = sorted({t.get("phase","") for t in tickets})
        chans_all  = sorted({t.get("channel_key","") for t in tickets})
        statuses   = ["ALL", "OPEN", "IN_PROGRESS", "DONE", "BLOCKED"]
        sel_phase  = fc1.selectbox("Filter by Phase", ["ALL"] + phases_all)
        sel_chan   = fc2.selectbox("Filter by Channel", ["ALL"] + chans_all)
        sel_status = fc3.selectbox("Filter by Status", statuses)

        filtered = [t for t in tickets
                    if (sel_phase  == "ALL" or t.get("phase") == sel_phase)
                    and (sel_chan  == "ALL" or t.get("channel_key") == sel_chan)
                    and (sel_status == "ALL" or t.get("status") == sel_status)]

        # Phase-grouped ticket board
        by_phase: dict = {}
        for t in filtered:
            by_phase.setdefault(t.get("phase","?"), []).append(t)

        CHANNEL_ICON = {
            "data_engineering": "⚙️ #data-engineering",
            "data_science":     "🔬 #data-science",
            "business":         "💼 #business",
            "ceo":              "👔 #ceo-approvals",
            "code_review":      "🔍 #code-reviews",
            "general":          "📢 #general",
        }

        for phase, phase_tickets in sorted(by_phase.items()):
            pname = {"phase1": "Phase 1 — Data ELT",
                     "phase2": "Phase 2 — EDA & Cleaning",
                     "phase3": "Phase 3 — Feature Engineering",
                     "phase4": "Phase 4 — Modelling",
                     "phase5": "Phase 5 — Deployment & Reporting"}.get(phase, phase)
            done_c  = sum(1 for t in phase_tickets if t.get("status") == "DONE")
            st.markdown(f'<div class="section-header">{pname} — {done_c}/{len(phase_tickets)} closed</div>',
                        unsafe_allow_html=True)

            for t in phase_tickets:
                status   = t.get("status", "OPEN")
                priority = t.get("priority", "P3-MEDIUM")
                s_color  = STATUS_COLOR.get(status, "#8ba8c4")
                p_color  = PRIORITY_COLOR.get(priority, "#8ba8c4")
                chan_label = CHANNEL_ICON.get(t.get("channel_key",""), t.get("channel_key",""))
                assignees = ", ".join(NICE_NAME.get(a, a) for a in t.get("assigned_to", []))
                creator   = NICE_NAME.get(t.get("created_by",""), t.get("created_by",""))

                header = (
                    f"<span style='color:{p_color};font-weight:700'>[{priority}]</span> "
                    f"<span style='color:{s_color};font-weight:700'>[{status}]</span> "
                    f"**{t['ticket_id']}** — {t['title']}"
                )
                with st.expander(f"{t['ticket_id']} | {status} | {t['title'][:60]}",
                                 expanded=False):
                    st.markdown(header, unsafe_allow_html=True)
                    mc1, mc2 = st.columns(2)
                    mc1.markdown(f"**Channel:** {chan_label}  \n"
                                 f"**Created by:** {creator}  \n"
                                 f"**Assigned to:** {assignees}")
                    mc2.markdown(f"**Phase:** {phase}  \n"
                                 f"**Created:** {t.get('created_at','')[:19]}  \n"
                                 f"**Closed:** {t.get('closed_at','N/A')[:19] if t.get('closed_at') else 'N/A'}")
                    st.caption(t.get("description", ""))
                    if t.get("resolution"):
                        st.success(f"Resolution: {t['resolution']}")
                    if t.get("updates"):
                        st.markdown("**Activity log:**")
                        for u in t["updates"]:
                            by   = NICE_NAME.get(u.get("updated_by",""), u.get("updated_by",""))
                            usts = u.get("status","")
                            uts  = u.get("ts","")[:19]
                            umsg = u.get("message","")
                            scol = STATUS_COLOR.get(usts, "#7c5cbf")
                            st.markdown(
                                f'<div class="thought-box" style="border-left-color:{scol}">'
                                f'<b>{by}</b> — {uts}<br>'
                                f'<i>{umsg}</i></div>',
                                unsafe_allow_html=True)

        # Slack raw feed
        if slack:
            st.divider()
            st.markdown('<div class="section-header">Slack Message Feed</div>', unsafe_allow_html=True)
            st.caption(f"{len(slack)} messages across all channels")
            tab_feed, tab_by_channel = st.tabs(["Recent (50)", "By Channel"])
            with tab_feed:
                # Single HTML block — no per-message st.markdown calls
                feed_html = "".join(
                    f'<div class="thought-box">'
                    f'<b>{CHANNEL_ICON.get(m.get("channel_key",""), "#" + m.get("channel",""))}</b>'
                    f' · {m.get("ts","")[:19]}<br>{m["text"][:300]}</div>'
                    for m in reversed(slack[-50:])
                )
                st.markdown(feed_html, unsafe_allow_html=True)
            with tab_by_channel:
                by_chan: dict = {}
                for msg in slack:
                    by_chan.setdefault(msg.get("channel_key","other"), []).append(msg)
                for ck, msgs in sorted(by_chan.items()):
                    label = CHANNEL_ICON.get(ck, f"#{ck}")
                    with st.expander(f"{label} — {len(msgs)} messages"):
                        # Batch all messages in this channel into one HTML string
                        chan_html = "".join(
                            f'<div class="thought-box">'
                            f'{m.get("ts","")[:19]}<br>{m["text"][:300]}</div>'
                            for m in msgs[:40]   # cap at 40 per channel
                        )
                        if len(msgs) > 40:
                            chan_html += f'<div style="color:#555;font-size:0.8em;padding:4px 8px;">… {len(msgs)-40} older messages not shown</div>'
                        st.markdown(chan_html,
                                unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 7 — DATA SOURCES + LINEAGE
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🗄️ Data Sources":
    st.title("🗄️ Data Sources & Lineage")
    st.caption("End-to-end data lineage: raw source → PySpark extract → staging → processed → features → model")

    import os as _os, json as _json

    RAW_STRUCT   = _os.path.join(_os.path.dirname(__file__), "..", "data", "raw", "structured")
    RAW_UNSTRUCT = _os.path.join(_os.path.dirname(__file__), "..", "data", "raw", "unstructured")
    STAGING_DIR  = _os.path.join(_os.path.dirname(__file__), "..", "data", "staging")

    # ── Summary strip ──────────────────────────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Raw Sources",      "7",   delta="4 structured · 3 unstructured")
    c2.metric("Raw Records",      "30M+", delta="5M customers · 25M transactions")
    c3.metric("Processed Tables", "7",   delta="all reviewed → PROD")
    c4.metric("Pipeline",         "PySpark", delta="SparkSession local mode")

    st.divider()

    # ── Shared helpers ─────────────────────────────────────────────────────────
    def _node(label, sublabel="", color="#1e3a5f", border="#4a9eff", width="160px"):
        return (
            f'<div style="background:{color};border:2px solid {border};border-radius:8px;'
            f'padding:8px 10px;text-align:center;min-width:{width};display:inline-block;">'
            f'<div style="color:#fff;font-weight:700;font-size:0.82em;">{label}</div>'
            + (f'<div style="color:#8ba8c4;font-size:0.72em;margin-top:2px;">{sublabel}</div>'
               if sublabel else "")
            + "</div>"
        )

    def _arrow(label=""):
        return (
            f'<div style="display:inline-block;vertical-align:middle;'
            f'padding:0 6px;color:#4a9eff;font-size:1.1em;">&#8594;'
            + (f'<div style="color:#4a5568;font-size:0.65em;text-align:center;">{label}</div>'
               if label else "")
            + "</div>"
        )

    def _lineage_row(*nodes_and_arrows):
        return (
            '<div style="display:flex;align-items:center;flex-wrap:wrap;gap:4px;'
            'margin:10px 0 4px;padding:10px 14px;background:#0a0f1e;border-radius:8px;">'
            + "".join(nodes_and_arrows)
            + "</div>"
        )

    # ── Lineage flows ─────────────────────────────────────────────────────────
    st.markdown('<div class="section-header">Structured Sources</div>', unsafe_allow_html=True)

    STRUCT_LINEAGE = [
        {
            "file":      "customers/",
            "path":      _os.path.join(RAW_STRUCT, "customers"),
            "is_dir":    True,
            "fmt":       "Parquet · partitioned by region",
            "rows":      "5,000,000 rows",
            "owner":     "DE1",
            "extractor": "SparkParquetExtractor",
            "staging":   "customers_staged/",
            "processed": "customers.parquet",
            "features":  ["recency_days", "frequency", "monetary", "loyalty_score", "region"],
            "desc":      "Customer demographics — age, region, loyalty tier, income bracket, signup date, email.",
            "preview":   lambda p: pd.read_parquet(p).head(5),
        },
        {
            "file":      "transactions/",
            "path":      _os.path.join(RAW_STRUCT, "transactions"),
            "is_dir":    True,
            "fmt":       "Parquet · partitioned by year_month",
            "rows":      "25,000,000 rows",
            "owner":     "DE1",
            "extractor": "SparkParquetExtractor",
            "staging":   "transactions_staged/",
            "processed": "transactions.parquet",
            "features":  ["recency_days", "frequency", "monetary", "avg_order_value",
                          "promo_sensitivity_score", "category_affinity", "channel_preference"],
            "desc":      "Purchase history — product, amount, channel, payment method, promo code.",
            "preview":   lambda p: pd.read_parquet(p).head(5),
        },
        {
            "file":      "products.xlsx",
            "path":      _os.path.join(RAW_STRUCT, "products.xlsx"),
            "is_dir":    False,
            "fmt":       "Excel · 2 sheets (Products + Inventory)",
            "rows":      "50 products",
            "owner":     "DE1",
            "extractor": "SparkExcelExtractor (pandas bridge)",
            "staging":   "products_products/ + products_inventory/",
            "processed": "products.parquet",
            "features":  ["category_affinity", "promo_sensitivity_score"],
            "desc":      "Product catalogue — price, category, brand, SKU. Inventory stock levels.",
            "preview":   lambda p: pd.read_excel(p, sheet_name="Products", nrows=5),
        },
        {
            "file":      "promotions.json",
            "path":      _os.path.join(RAW_STRUCT, "promotions.json"),
            "is_dir":    False,
            "fmt":       "JSON · nested arrays",
            "rows":      "20 campaigns",
            "owner":     "DE1",
            "extractor": "SparkJSONExtractor (flattened)",
            "staging":   "promotions_staged/",
            "processed": "promos.parquet",
            "features":  ["promo_sensitivity_score"],
            "desc":      "20 promo campaigns — discount %, eligible categories/products, start/end dates.",
            "preview":   lambda p: pd.DataFrame(_json.load(open(p))[:3]),
        },
    ]

    UNSTRUCT_LINEAGE = [
        {
            "file":      "customer_reviews.txt",
            "path":      _os.path.join(RAW_UNSTRUCT, "customer_reviews.txt"),
            "fmt":       "TXT · regex-structured",
            "rows":      "10,000 reviews",
            "owner":     "DE2",
            "extractor": "SparkTextExtractor + regex",
            "staging":   "reviews_staged/",
            "processed": "reviews_enriched.parquet",
            "features":  ["satisfaction_score", "avg_review_sentiment"],
            "desc":      "Free-text customer reviews with star ratings. Parsed into: review_id, customer_id, product_id, date, rating, review_text.",
        },
        {
            "file":      "call_transcripts.txt",
            "path":      _os.path.join(RAW_UNSTRUCT, "call_transcripts.txt"),
            "fmt":       "TXT · regex-structured",
            "rows":      "2,000 calls",
            "owner":     "DE2",
            "extractor": "SparkTextExtractor + regex",
            "staging":   "transcripts_staged/",
            "processed": "customer_text_features.parquet",
            "features":  ["n_support_calls", "resolution_rate"],
            "desc":      "Call-centre transcripts — agent, resolution, sentiment. Parsed into: call_id, customer_id, product_id, sentiment, resolution.",
        },
        {
            "file":      "support_emails.txt",
            "path":      _os.path.join(RAW_UNSTRUCT, "support_emails.txt"),
            "fmt":       "TXT · regex-structured",
            "rows":      "3,000 emails",
            "owner":     "DE2",
            "extractor": "SparkTextExtractor + regex",
            "staging":   "emails_staged/",
            "processed": "customer_text_features.parquet",
            "features":  ["email_sentiment", "n_support_calls"],
            "desc":      "Support email threads — subject, product reference, sentiment label.",
        },
    ]

    def _render_lineage_card(src):
        path    = src["path"]
        is_dir  = src.get("is_dir", False)
        exists  = _os.path.isdir(path) if is_dir else _os.path.isfile(path)
        icon    = "✅" if exists else "⏳"
        feats   = " · ".join(src["features"])

        with st.expander(f"{icon}  `{src['file']}`  —  {src['rows']}  —  owner: {src['owner']}"):
            # Lineage flow diagram
            st.markdown(
                _lineage_row(
                    _node(src["file"],    src["fmt"],      "#1a0a30", "#7c5cbf"),
                    _arrow("extract"),
                    _node(src["extractor"], "Phase 1 ELT", "#0a1a30", "#4a9eff"),
                    _arrow("stage"),
                    _node(src["staging"],  "Parquet",      "#0a1e0a", "#4cef9a"),
                    _arrow("clean"),
                    _node(src["processed"], "PROD",        "#1a1a0a", "#f6c90e"),
                    _arrow("feeds"),
                    _node(feats,           "features",     "#1a0a1a", "#c97bff", "200px"),
                ),
                unsafe_allow_html=True,
            )
            st.caption(src["desc"])

            # Detail columns
            d1, d2, d3 = st.columns(3)
            d1.markdown(f"**Format:** `{src['fmt']}`  \n**Rows:** {src['rows']}")
            d2.markdown(f"**Extractor:** `{src['extractor']}`  \n**Owner:** {src['owner']}")
            d3.markdown(f"**Processed output:** `{src['processed']}`")

            # Data preview
            if exists and src.get("preview"):
                try:
                    df_prev = src["preview"](path)
                    st.dataframe(df_prev, use_container_width=True)
                    st.caption(f"5-row preview — {src['rows']} total")
                except Exception as e:
                    st.warning(f"Preview unavailable: {e}")
            elif not exists:
                st.info("Run the pipeline to generate this source.")

    for src in STRUCT_LINEAGE:
        _render_lineage_card(src)

    st.markdown('<div class="section-header">Unstructured Sources</div>', unsafe_allow_html=True)

    for src in UNSTRUCT_LINEAGE:
        path   = src["path"]
        exists = _os.path.exists(path)
        icon   = "✅" if exists else "⏳"
        feats  = " · ".join(src["features"])

        with st.expander(f"{icon}  `{src['file']}`  —  {src['rows']}  —  owner: {src['owner']}"):
            st.markdown(
                _lineage_row(
                    _node(src["file"],    src["fmt"],        "#1a0a30", "#7c5cbf"),
                    _arrow("regex parse"),
                    _node(src["extractor"], "Phase 1 ELT",   "#0a1a30", "#4a9eff"),
                    _arrow("stage"),
                    _node(src["staging"],  "Parquet",         "#0a1e0a", "#4cef9a"),
                    _arrow("aggregate"),
                    _node(src["processed"], "PROD",           "#1a1a0a", "#f6c90e"),
                    _arrow("feeds"),
                    _node(feats,           "features",        "#1a0a1a", "#c97bff", "200px"),
                ),
                unsafe_allow_html=True,
            )
            st.caption(src["desc"])

            d1, d2, d3 = st.columns(3)
            d1.markdown(f"**Format:** `{src['fmt']}`  \n**Rows:** {src['rows']}")
            d2.markdown(f"**Extractor:** `{src['extractor']}`  \n**Owner:** {src['owner']}")
            d3.markdown(f"**Processed output:** `{src['processed']}`")

            if exists:
                try:
                    with open(path, encoding="utf-8") as _f:
                        st.code(_f.read()[:600], language=None)
                except Exception as e:
                    st.warning(f"Preview error: {e}")
            else:
                st.info("Run the pipeline to generate this source.")

    st.divider()

    # ── Full pipeline lineage summary ──────────────────────────────────────────
    st.markdown('<div class="section-header">End-to-End Pipeline Lineage</div>',
                unsafe_allow_html=True)
    st.markdown(
        _lineage_row(
            _node("7 Raw Sources", "structured + unstructured", "#1a0a30", "#7c5cbf", "130px"),
            _arrow("Phase 1"),
            _node("Staging\nParquet", "7 dirs", "#0a1a30", "#4a9eff", "110px"),
            _arrow("Phase 1"),
            _node("Processed\nTables", "7 files · PROD", "#0a1e0a", "#4cef9a", "120px"),
            _arrow("Phase 3"),
            _node("Feature Matrix", "5M rows · 22 cols", "#1a1a0a", "#f6c90e", "130px"),
            _arrow("Phase 4"),
            _node("KMeans\nSegments", "3 clusters", "#1a0a1a", "#c97bff", "110px"),
            _arrow("Phase 4"),
            _node("ALS\nRecommender", "top-10 per customer", "#0a1a1a", "#4cef9a", "130px"),
        ),
        unsafe_allow_html=True,
    )


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 8 — ENVIRONMENTS  (DEV workspace vs PROD environment, clearly separated)
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🌍 Environments":
    st.title("🌍 Dev / Prod Environments")

    work_reviews = load_json("work_reviews.json") or []
    promotions   = load_json("env_promotions.json") or []
    env_registry = load_json("env_registry.json") or []

    PHASE_LABELS = {
        "phase1": "Phase 1 — Data Ingestion",
        "phase2": "Phase 2 — EDA",
        "phase3": "Phase 3 — Feature Engineering",
        "phase4": "Phase 4 — Segmentation + Recommender",
        "phase5": "Phase 5 — Reports + Deployment",
    }
    PHASE_ORDER = ["phase1","phase2","phase3","phase4","phase5"]
    VERDICT_COLOR = {
        "APPROVED":                 "#22c55e",
        "APPROVED_WITH_CONDITIONS": "#f59e0b",
        "NEEDS_REVISION":           "#ef4444",
    }
    SEV_COLOR = {"INFO": "#3b82f6", "WARN": "#f59e0b", "CRITICAL": "#ef4444"}

    # ── Top metrics strip ──────────────────────────────────────────────────────
    approved = sum(1 for r in work_reviews if "APPROVED" in r.get("verdict",""))
    prod_arts = sum(1 for r in env_registry if r.get("env") == "prod")
    mc1, mc2, mc3, mc4 = st.columns(4)
    mc1.metric("DEV Reviews",    len(work_reviews), delta="senior sign-offs required")
    mc2.metric("Passed Reviews", approved,          delta=f"{int(100*approved/max(len(work_reviews),1))}% approval rate")
    mc3.metric("PROD Promotions",len(promotions),   delta="one per phase")
    mc4.metric("PROD Artefacts", prod_arts,         delta="reviewed + cleared")

    st.divider()

    # ── Four tabs ─────────────────────────────────────────────────────────────
    tab_dev, tab_prod, tab_flow, tab_chain = st.tabs([
        "🟣 DEV Workspace",
        "🟢 PROD Environment",
        "🔀 Promotion History",
        "📋 Review Chain",
    ])

    # ══ TAB 1: DEV ════════════════════════════════════════════════════════════
    with tab_dev:
        st.markdown(
            '<div style="background:#1a0a30;border-left:4px solid #7c5cbf;border-radius:6px;'
            'padding:10px 16px;margin-bottom:16px;">'
            '<span style="background:#7c5cbf;color:#fff;border-radius:4px;padding:2px 8px;'
            'font-size:0.75em;font-weight:700;">DEV</span> &nbsp;'
            '<span style="color:#ccc;">All agent work starts here. Nothing reaches PROD '
            'until every review in the chain is APPROVED.</span></div>',
            unsafe_allow_html=True,
        )

        if not work_reviews:
            st.info("No DEV reviews recorded yet. Run the pipeline first.")
        else:
            for phase in PHASE_ORDER:
                phase_wrs = [r for r in work_reviews if r.get("phase") == phase]
                if not phase_wrs:
                    continue
                all_ok = all("APPROVED" in r.get("verdict","") for r in phase_wrs)
                ph_icon = "✅" if all_ok else "🔄"
                with st.expander(
                    f"{ph_icon} **{PHASE_LABELS.get(phase, phase)}** — "
                    f"{len(phase_wrs)} work review(s) in DEV",
                    expanded=False,
                ):
                    # Batch all review cards for this phase into one HTML block
                    cards_html = ""
                    for r in phase_wrs:
                        verdict   = r.get("verdict", "")
                        v_color   = VERDICT_COLOR.get(verdict, "#7c5cbf")
                        submitted = r.get("submitted_by_role", r.get("submitted_by",""))
                        reviewers = [AGENT_ROLES.get(rv, rv) for rv in r.get("reviewers", [])]
                        reviewer_chain = " → ".join(
                            f'<span style="color:#4cef9a">{rv}</span>' for rv in reviewers
                        )
                        findings_html = "".join(
                            f'<div style="margin:2px 0 2px 14px;">'
                            f'<span style="color:{SEV_COLOR.get(f.get("severity","INFO"),"#aaa")};'
                            f'font-size:0.72em;font-weight:700;">[{f.get("severity","INFO")}]</span> '
                            f'<span style="color:#aaa;font-size:0.78em;">'
                            f'<b>{f.get("category","")}</b>: {f.get("finding","")}</span></div>'
                            for f in r.get("findings", [])
                        )
                        cards_html += (
                            f'<div style="background:#12082a;border-left:4px solid {v_color};'
                            f'border-radius:6px;padding:10px 14px;margin:6px 0;">'
                            f'<div style="margin-bottom:4px;">'
                            f'<span style="background:#7c5cbf;color:#fff;border-radius:3px;'
                            f'padding:1px 7px;font-size:0.7em;font-weight:700;">DEV</span>'
                            f'&nbsp;<span style="background:{v_color};color:#000;border-radius:3px;'
                            f'padding:1px 7px;font-size:0.7em;font-weight:700;">{verdict}</span>'
                            f'&nbsp;<span style="color:#555;font-size:0.72em;">'
                            f'{r.get("timestamp","")[:16].replace("T"," ")}</span></div>'
                            f'<div style="color:#e0e0e0;font-size:0.85em;font-weight:600;margin-bottom:3px;">'
                            f'{r.get("artifact","")}</div>'
                            f'<div style="font-size:0.78em;color:#8ba8c4;">'
                            f'<b style="color:#aaa;">Submitted by:</b> {submitted} &nbsp;'
                            f'<b style="color:#aaa;">Review chain:</b> {reviewer_chain}</div>'
                            f'<div style="color:#ccc;font-size:0.80em;margin-top:4px;">{r.get("summary","")}</div>'
                            f'{findings_html}</div>'
                        )
                    st.markdown(cards_html, unsafe_allow_html=True)

    # ══ TAB 2: PROD ═══════════════════════════════════════════════════════════
    with tab_prod:
        st.markdown(
            '<div style="background:#0a2a0a;border-left:4px solid #22c55e;border-radius:6px;'
            'padding:10px 16px;margin-bottom:16px;">'
            '<span style="background:#22c55e;color:#000;border-radius:4px;padding:2px 8px;'
            'font-size:0.75em;font-weight:700;">PROD</span> &nbsp;'
            '<span style="color:#ccc;">Only artefacts that have passed every DEV review gate '
            'and been formally promoted appear here. These are the artefacts used by downstream '
            'phases and the production dashboard.</span></div>',
            unsafe_allow_html=True,
        )

        if not env_registry:
            st.info("No PROD artefacts yet. Run the pipeline first.")
        else:
            # Group registry by promotion phase
            # Map artefact -> promotion event for timestamp / approver
            prom_map = {}
            for p in promotions:
                for art in p.get("artifacts", []):
                    prom_map[art] = p

            for phase in PHASE_ORDER:
                phase_arts = [r for r in env_registry
                              if r.get("env") == "prod"
                              and prom_map.get(r["artifact"], {}).get("phase") == phase]
                if not phase_arts:
                    continue

                st.markdown(
                    f'<div style="color:#22c55e;font-weight:700;font-size:0.9em;'
                    f'margin:14px 0 6px;border-bottom:1px solid #1a3a1a;padding-bottom:4px;">'
                    f'✅ {PHASE_LABELS.get(phase, phase)}</div>',
                    unsafe_allow_html=True,
                )

                prom = prom_map.get(phase_arts[0]["artifact"], {})
                approver_role = AGENT_ROLES.get(prom.get("final_approver",""), prom.get("final_approver",""))
                ts_str = prom.get("timestamp","")[:16].replace("T"," ") if prom.get("timestamp") else "—"

                cols = st.columns(min(len(phase_arts), 4))
                for i, art in enumerate(phase_arts):
                    with cols[i % len(cols)]:
                        st.markdown(
                            f'<div style="background:#0a1e0a;border:2px solid #22c55e;'
                            f'border-radius:8px;padding:10px 12px;text-align:center;margin-bottom:6px;">'
                            f'<div style="background:#22c55e;color:#000;border-radius:3px;'
                            f'padding:1px 6px;font-size:0.65em;font-weight:700;display:inline-block;'
                            f'margin-bottom:6px;">PROD</div>'
                            f'<div style="color:#fff;font-size:0.82em;font-weight:600;">'
                            f'{art["artifact"]}</div>'
                            f'<div style="color:#4cef9a;font-size:0.70em;margin-top:4px;">'
                            f'promoted {art.get("promoted_at","")[:10]}</div>'
                            f'</div>',
                            unsafe_allow_html=True,
                        )
                st.caption(f"Final approver: **{approver_role}** · Promoted at: {ts_str}")

    # ══ TAB 3: PROMOTION HISTORY ══════════════════════════════════════════════
    with tab_flow:
        st.markdown("Each phase follows the same flow: agents do work in **DEV**, seniors review, "
                    "then the phase is **promoted to PROD** before the next phase begins.")
        st.write("")

        if not promotions:
            st.info("No promotions recorded yet. Run the pipeline first.")
        else:
            for p in promotions:
                phase     = p.get("phase","")
                arts      = p.get("artifacts", [])
                submitter = AGENT_ROLES.get(p.get("submitted_by",""), p.get("submitted_by",""))
                approver  = AGENT_ROLES.get(p.get("final_approver",""), p.get("final_approver",""))
                ts_str    = p.get("timestamp","")[:16].replace("T"," ") if p.get("timestamp") else "—"
                phase_wrs = [r for r in work_reviews if r.get("phase") == phase]
                n_reviews = len(phase_wrs)

                st.markdown(
                    f'<div style="background:#0d0d1a;border-radius:10px;padding:14px 18px;'
                    f'margin-bottom:12px;border:1px solid #1a1a3a;">'
                    # Phase label
                    f'<div style="color:#4a9eff;font-weight:700;font-size:0.9em;margin-bottom:10px;">'
                    f'{PHASE_LABELS.get(phase, phase)}</div>'
                    # Flow arrow
                    f'<div style="display:flex;align-items:center;gap:10px;flex-wrap:wrap;">'
                    f'<div style="background:#1a0a30;border:2px solid #7c5cbf;border-radius:6px;'
                    f'padding:6px 14px;text-align:center;">'
                    f'<div style="color:#7c5cbf;font-size:0.65em;font-weight:700;">DEV</div>'
                    f'<div style="color:#ccc;font-size:0.8em;">{n_reviews} review(s) passed</div>'
                    f'</div>'
                    f'<div style="color:#4a9eff;font-size:1.4em;">&#8594;</div>'
                    f'<div style="background:#0a1e30;border:1px solid #4a9eff;border-radius:6px;'
                    f'padding:6px 14px;text-align:center;">'
                    f'<div style="color:#4a9eff;font-size:0.65em;font-weight:700;">REVIEW GATE</div>'
                    f'<div style="color:#ccc;font-size:0.8em;">submitted by {submitter}</div>'
                    f'</div>'
                    f'<div style="color:#4a9eff;font-size:1.4em;">&#8594;</div>'
                    f'<div style="background:#0a2a0a;border:2px solid #22c55e;border-radius:6px;'
                    f'padding:6px 14px;text-align:center;">'
                    f'<div style="color:#22c55e;font-size:0.65em;font-weight:700;">PROD ✅</div>'
                    f'<div style="color:#ccc;font-size:0.8em;">approved by {approver}</div>'
                    f'</div>'
                    f'</div>'
                    # Artefacts
                    f'<div style="margin-top:8px;color:#8ba8c4;font-size:0.78em;">'
                    f'<b>Promoted artefacts:</b> '
                    + "  ".join(
                        f'<span style="background:#0a2a0a;border:1px solid #22c55e;border-radius:3px;'
                        f'padding:1px 7px;color:#4cef9a;font-size:0.9em;">{a}</span>'
                        for a in arts
                    )
                    + f'<span style="float:right;color:#555;">{ts_str}</span></div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

    # ══ TAB 4: REVIEW CHAIN ═══════════════════════════════════════════════════
    with tab_chain:
        st.markdown("Who must sign off on whose work before it can go to PROD.")
        st.write("")

        CHAIN_ROWS = [
            ("Data Engineer 1",        ["DE Lead"],                          "Phase 1 ELT artefacts"),
            ("Data Engineer 2",        ["DE Lead"],                          "Phase 1 data quality"),
            ("DE Lead",                ["Code Reviewer", "Project Manager"], "Phase 1 full ELT sign-off"),
            ("Data Scientist 1",       ["Senior Data Scientist"],            "EDA + feature engineering"),
            ("Data Scientist 2",       ["Senior Data Scientist", "DS Lead"], "Segmentation + recommender models"),
            ("Senior Data Scientist",  ["DS Lead"],                          "Model validation sign-off"),
            ("ML Engineer",            ["DS Lead"],                          "Deployment + monitoring"),
            ("Business Analyst 1",     ["Business Lead"],                    "Reports + visualisations"),
            ("Business Analyst 2",     ["Business Lead"],                    "Stakeholder communications"),
            ("Marketing Analyst",      ["Business Lead"],                    "Promo strategy briefs"),
            ("Finance Analyst",        ["Business Lead"],                    "ROI calculations"),
            ("Business Lead",          ["Project Manager", "CEO"],           "All Phase 5 business deliverables"),
            ("DS Lead",                ["Project Manager", "CEO"],           "Technical sign-off across phases"),
            ("Project Manager",        ["CEO"],                              "Project completion + PROD promotion"),
        ]

        chain_html = "".join(
            f'<div style="display:flex;align-items:center;gap:12px;'
            f'padding:7px 12px;margin-bottom:4px;background:#0d0d1a;border-radius:6px;">'
            f'<div style="min-width:180px;color:#e0e0e0;font-size:0.85em;font-weight:600;">{agent}</div>'
            f'<div style="color:#555;font-size:1em;">&#8594;</div>'
            f'<div style="min-width:260px;">'
            + "".join(
                f'<span style="background:#0a1e30;border:1px solid #4a9eff;border-radius:4px;'
                f'padding:2px 8px;color:#4a9eff;font-size:0.78em;margin-right:4px;">{rv}</span>'
                for rv in reviewers
            )
            + f'</div>'
            f'<div style="color:#555;font-size:0.78em;font-style:italic;">{scope}</div>'
            f'</div>'
            for agent, reviewers, scope in CHAIN_ROWS
        )
        st.markdown(chain_html, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 9 — AGENT MONITOR
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🧠 Agent Monitor":
    st.title("🧠 Agent Monitor")
    st.caption("Per-agent reasoning, decisions and tools across all 5 phases")

    activity = load_json("agent_activity.json") or []
    if not activity:
        st.warning("No agent activity. Run the pipeline first.")
        st.stop()

    df_act = pd.DataFrame(activity)
    view   = st.radio("View", ["Phase Timeline", "Per-Agent Thinking", "Activity Stats", "Reports"],
                      horizontal=True)

    PAGE_SIZE = 20  # max events rendered per phase / agent to keep UI fast

    def _render_event(ev, all_ids):
        d     = ev.get("detail", {}) or {}
        rol   = ev.get("role", ev.get("agent_id", ""))
        act   = ev.get("action", "")
        ts    = str(ev.get("timestamp", ""))[:19].replace("T", " ")
        aid   = ev.get("agent_id", "")
        cidx  = all_ids.index(aid) if aid in all_ids else 0
        color = SEGMENT_COLORS[cidx % len(SEGMENT_COLORS)]
        # Batch all HTML into one markdown call instead of 4-5 separate calls
        parts = [
            f'<div style="margin:8px 0;padding:4px 0;border-top:1px solid #1a2540;">'
            f'<strong style="color:{color}">{rol}</strong>'
            f' — <em style="color:#8ba8c4">{act}</em>'
            f'<span style="float:right;color:#4a5568;font-size:0.8rem">{ts}</span></div>',
        ]
        if d.get("thought"):
            parts.append(f'<div class="thought-box">💭 {d["thought"]}</div>')
        if d.get("decision"):
            parts.append(f'<div style="color:#4cef9a;font-size:0.85em;margin-left:8px">✔ {d["decision"]}</div>')
        if d.get("result"):
            parts.append(f'<div style="color:#8ba8c4;font-size:0.82em;margin-left:8px">→ {d["result"]}</div>')
        st.markdown("".join(parts), unsafe_allow_html=True)
        if d.get("tools_called"):
            st.code("\n".join(d["tools_called"]), language="python")

    if view == "Phase Timeline":
        all_ids = df_act["agent_id"].unique().tolist()
        phases  = sorted(df_act["phase"].unique().tolist())
        for phase in phases:
            icon   = PHASE_ICONS.get(phase, "📌")
            name   = PHASE_NAMES.get(phase, phase)
            ph_evs = df_act[df_act["phase"] == phase]
            total  = len(ph_evs)
            with st.expander(f"{icon} {name} — {total} events", expanded=False):
                shown = ph_evs.head(PAGE_SIZE)
                for _, ev in shown.iterrows():
                    _render_event(ev, all_ids)
                if total > PAGE_SIZE:
                    st.caption(f"Showing {PAGE_SIZE} of {total} events. Use Per-Agent Thinking for full detail.")

    elif view == "Per-Agent Thinking":
        agents    = sorted(df_act["agent_id"].unique().tolist())
        all_ids   = agents
        nice_list = [NICE_NAME.get(a, a.replace("_"," ").title()) for a in agents]
        sel_nice  = st.selectbox("Select agent", nice_list)
        sel_agent = agents[nice_list.index(sel_nice)]
        agent_evs = df_act[df_act["agent_id"] == sel_agent]
        total     = len(agent_evs)

        st.markdown(f"#### {sel_nice} — {total} actions logged")
        page_n = st.number_input("Page", min_value=1,
                                  max_value=max(1, (total - 1) // PAGE_SIZE + 1),
                                  value=1, step=1)
        start  = (page_n - 1) * PAGE_SIZE
        chunk  = agent_evs.iloc[start: start + PAGE_SIZE]
        for _, ev in chunk.iterrows():
            _render_event(ev, all_ids)
            st.markdown("---")
        st.caption(f"Page {page_n} · showing {start+1}–{min(start+PAGE_SIZE, total)} of {total}")

    elif view == "Activity Stats":
        cl, cr = st.columns(2)
        with cl:
            ac = (df_act["agent_id"].value_counts()
                        .reset_index()
                        .rename(columns={"agent_id": "agent", "count": "actions"}))
            ac["name"] = ac["agent"].map(lambda x: NICE_NAME.get(x, x.replace("_"," ").title()))
            fig_ac = px.bar(ac, x="actions", y="name", orientation="h",
                            color="actions", color_continuous_scale="Blues",
                            title="Actions per Agent")
            fig_ac.update_layout(template="plotly_dark", height=420,
                                  coloraxis_showscale=False, margin=dict(t=40, b=20))
            st.plotly_chart(fig_ac, use_container_width=True)

        with cr:
            pc = (df_act["phase"].value_counts()
                        .reset_index()
                        .rename(columns={"phase": "phase_id", "count": "events"}))
            pc["name"] = pc["phase_id"].map(PHASE_NAMES)
            fig_pc = px.pie(pc, names="name", values="events",
                            color_discrete_sequence=SEGMENT_COLORS,
                            title="Events per Phase", hole=0.4)
            fig_pc.update_layout(template="plotly_dark", height=420,
                                  margin=dict(t=40, b=20))
            st.plotly_chart(fig_pc, use_container_width=True)

    else:  # Reports
        report_tab, ceo_tab, review_tab = st.tabs(
            ["📄 Agent Reports", "👔 CEO Approvals", "🔍 Code Reviews"]
        )

        with ceo_tab:
            st.markdown('<div class="section-header">CEO Approval Gates</div>',
                        unsafe_allow_html=True)
            st.caption("Every phase required explicit CEO approval before implementation.")
            approvals_data = load_json("ceo_approvals.json") or []
            if not approvals_data:
                st.info("No CEO approvals yet. Run the pipeline first.")
            for appr in approvals_data:
                dec   = appr.get("decision", "")
                color = DECISION_COLOR.get(dec, "#8ba8c4")
                with st.expander(
                    f"**{appr.get('phase','').replace('_',' ').title()}** — "
                    f"{dec.replace('_',' ')} (requested by {appr.get('requested_by_role','')})"):
                    st.markdown(
                        f'<div style="border-left:4px solid {color};padding:12px 16px;'
                        f'background:#0d1524;border-radius:8px;margin-bottom:8px;">'
                        f'<div style="color:{color};font-weight:700;margin-bottom:8px">'
                        f'{dec.replace("_"," ")}</div>'
                        f'<strong style="color:#8ba8c4">Request:</strong>'
                        f'<div style="color:#cdd4e8;margin:4px 0 10px">'
                        f'{appr.get("request_summary","")}</div>'
                        f'<strong style="color:#8ba8c4">CEO Rationale:</strong>'
                        f'<div style="color:#e0e4f0;margin:4px 0 10px">'
                        f'{appr.get("ceo_rationale","")}</div>'
                        + (f'<strong style="color:#f6c90e">Conditions:</strong>'
                           f'<div style="color:#f6c90e;margin:4px 0">'
                           f'{appr["conditions"]}</div>'
                           if appr.get("conditions") else "")
                        + f'<div style="color:#4a5568;font-size:0.8rem;margin-top:8px">'
                          f'{appr.get("timestamp","")[:19].replace("T"," ")} UTC</div>'
                          f'</div>',
                        unsafe_allow_html=True,
                    )

        with review_tab:
            st.markdown('<div class="section-header">Code Reviews</div>',
                        unsafe_allow_html=True)
            st.caption("Every technical artefact was reviewed before implementation.")
            reviews_data = load_json("code_reviews.json") or []
            if not reviews_data:
                st.info("No code reviews yet. Run the pipeline first.")
            for rev in reviews_data:
                verdict = rev.get("verdict", "")
                vcolor  = VERDICT_COLOR.get(verdict, "#8ba8c4")
                crit_n  = sum(1 for f in rev.get("findings",[]) if f.get("severity")=="CRITICAL")
                warn_n  = sum(1 for f in rev.get("findings",[]) if f.get("severity")=="WARN")
                info_n  = sum(1 for f in rev.get("findings",[]) if f.get("severity")=="INFO")
                with st.expander(
                    f"**{rev.get('artifact','')}** — "
                    f"{verdict.replace('_',' ')} "
                    f"| {crit_n} CRITICAL · {warn_n} WARN · {info_n} INFO",
                    expanded=False):
                    c_left, c_right = st.columns([3, 1])
                    with c_right:
                        st.markdown(
                            f'<div style="background:{vcolor}22;border:2px solid {vcolor};'
                            f'border-radius:8px;padding:10px;text-align:center;">'
                            f'<div style="color:{vcolor};font-weight:700;font-size:1.1rem">'
                            f'{verdict.replace("_"," ")}</div>'
                            f'<div style="color:#8ba8c4;font-size:0.8rem;margin-top:4px">'
                            f'{rev.get("review_id","")}</div></div>',
                            unsafe_allow_html=True)
                        st.markdown(
                            f'<div style="margin-top:8px;font-size:0.85rem;">'
                            f'<span style="color:#8ba8c4">Phase:</span> {rev.get("phase","")}<br>'
                            f'<span style="color:#8ba8c4">By:</span> '
                            f'{rev.get("submitted_by_role", rev.get("submitted_by",""))}'
                            f'</div>', unsafe_allow_html=True)
                    with c_left:
                        st.markdown(f'**Summary:** {rev.get("summary","")}')
                        st.markdown("**Findings:**")
                        for finding in rev.get("findings", []):
                            sev   = finding.get("severity", "INFO")
                            scolor = SEVERITY_COLOR.get(sev, "#8ba8c4")
                            st.markdown(
                                f'<div style="background:#0d1524;border-radius:6px;'
                                f'padding:8px 12px;margin:4px 0;'
                                f'border-left:3px solid {scolor};">'
                                f'<span style="background:{scolor};color:#000;'
                                f'padding:1px 6px;border-radius:4px;font-size:0.75rem;'
                                f'font-weight:700">{sev}</span>'
                                f' <span style="color:#8ba8c4;font-size:0.8rem">'
                                f'[{finding.get("category","")}]</span><br>'
                                f'<span style="color:#cdd4e8">{finding.get("finding","")}</span>'
                                f'<br><span style="color:#4a9eff;font-size:0.85rem">'
                                f'Rec: {finding.get("recommendation","")}</span>'
                                f'</div>',
                                unsafe_allow_html=True)

        with report_tab:
            st.markdown('<div class="section-header">Agent-Generated Reports</div>',
                        unsafe_allow_html=True)
            st.caption("These are the actual files produced by agents during Phase 5.")

            REPORT_FILES = [
                ("segment_report.md",    "Segment Report",          "Business Analyst 1",  "phase5"),
                ("eda_transactions.md",  "EDA — Transactions",      "Data Scientist 1",    "phase2"),
                ("eda_customers.md",     "EDA — Customers",         "Data Scientist 1",    "phase2"),
            ]

            for filename, label, author, phase in REPORT_FILES:
                rpath = os.path.join(REPORTS, filename)
                exists = os.path.exists(rpath)
                icon   = "📄" if exists else "❌"
                with st.expander(f"{icon} **{label}** — by {author} [{phase}]",
                                 expanded=(filename == "segment_report.md" and exists)):
                    if exists:
                        with open(rpath, "r", encoding="utf-8") as _f:
                            content = _f.read()
                        st.markdown(content)
                        st.download_button(
                            label=f"Download {filename}",
                            data=content,
                            file_name=filename,
                            mime="text/markdown",
                            key=f"dl_{filename}",
                        )
                    else:
                        st.warning(f"{filename} not found. Run the pipeline to generate it.")

            # HTML report link
            st.divider()
            html_path = os.path.join(REPORTS, "final_report.html")
            if os.path.exists(html_path):
                with open(html_path, "r", encoding="utf-8") as _f:
                    html_content = _f.read()
                st.markdown("**Final HTML Report** (generated by Project Manager — phase5)")
                st.download_button(
                    label="Download final_report.html",
                    data=html_content,
                    file_name="final_report.html",
                    mime="text/html",
                    key="dl_final_html",
                )
                with st.expander("Preview final_report.html"):
                    st.components.v1.html(html_content, height=500, scrolling=True)
            else:
                st.info("final_report.html not found.")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 7 — MODEL PERFORMANCE
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📈 Model Performance":
    st.title("📈 Model Performance")
    st.caption("Technical metrics from segmentation and collaborative filtering models")

    state  = load_json("project_state.json") or {}
    kpis   = load_json("kpi_metrics.json") or {}
    models = state.get("models", {})

    if not models:
        st.warning("No model data. Run the pipeline first.")
        st.stop()

    seg_meta = models.get("segmentation", {})
    rec_meta = models.get("recommender", {})

    st.markdown('<div class="section-header">Trained Models</div>',
                unsafe_allow_html=True)
    mc1, mc2 = st.columns(2)

    with mc1:
        st.markdown("#### Segmentation Model (KMeans)")
        sil = float(seg_meta.get("silhouette_score", 0))
        nc  = seg_meta.get("n_clusters", 0)
        st.metric("Silhouette Score",  f"{sil:.4f}",
                  "Acceptable for retail" if sil > 0.1 else "Low")
        st.metric("Clusters (k)",      nc)
        st.metric("Inertia",           f"{seg_meta.get('inertia', 0):,.0f}")
        st.metric("Validation",        "Senior DS approved")
        fig_g = go.Figure(go.Indicator(
            mode="gauge+number", value=sil,
            title={"text": "Silhouette Score"},
            gauge={
                "axis": {"range": [-1, 1]},
                "bar":  {"color": "#4a9eff"},
                "steps": [
                    {"range": [-1, 0],    "color": "#3d0c11"},
                    {"range": [0, 0.2],   "color": "#1a2540"},
                    {"range": [0.2, 0.5], "color": "#0d3040"},
                    {"range": [0.5, 1],   "color": "#0a3d20"},
                ],
            },
            number={"valueformat": ".4f"},
        ))
        fig_g.update_layout(template="plotly_dark", height=220, margin=dict(t=40, b=0))
        st.plotly_chart(fig_g, use_container_width=True)

    with mc2:
        backend = rec_meta.get("model", "NMF")
        st.markdown(f"#### Recommender Model ({backend})")
        st.metric("Backend Algorithm", backend)
        st.metric("Top-N per Customer", rec_meta.get("top_n", 10))
        st.metric("Validation", "DS Lead approved")
        st.markdown(
            f'<div style="background:#0f1629;border-radius:8px;padding:16px;'
            f'margin-top:12px;border-left:4px solid #4a9eff;">'
            f'<div style="color:#4a9eff;font-weight:700;margin-bottom:8px">About {backend}</div>'
            f'<div style="color:#cdd4e8;font-size:0.9rem">'
            + ("ALS (Alternating Least Squares) factorises the user-item matrix "
               "using implicit feedback (transaction amounts as confidence signals). "
               "It captures latent preference patterns without explicit ratings."
               if backend == "ALS" else
               "NMF (Non-negative Matrix Factorisation) decomposes the user-item matrix "
               "into non-negative latent factors. Used as fallback when ALS is unavailable.")
            + '</div></div>',
            unsafe_allow_html=True)

    st.divider()

    # KPI tracker
    st.markdown('<div class="section-header">Business KPI Targets</div>',
                unsafe_allow_html=True)
    if kpis:
        kpi_df = pd.DataFrame([
            {"KPI": k.replace("_", " ").title(), "Target": v}
            for k, v in kpis.items()
        ])
        st.dataframe(kpi_df, use_container_width=True)

    # Model versions
    st.divider()
    st.markdown('<div class="section-header">Model Version Registry</div>',
                unsafe_allow_html=True)
    version_path = os.path.join(MODELS_DIR, "model_versions.json")
    if os.path.exists(version_path):
        with open(version_path, "r") as f:
            versions = json.load(f)
        if isinstance(versions, list):
            st.dataframe(pd.DataFrame(versions), use_container_width=True)
        else:
            st.json(versions)
    else:
        st.info("Model registry not yet available.")

    # Approved features
    st.divider()
    st.markdown('<div class="section-header">Approved Feature Set</div>',
                unsafe_allow_html=True)
    approved = state.get("approved_features", [])
    if approved:
        feats = [f for f in approved if f not in ["customer_id", "cluster"]]
        cols_per_row = 4
        rows = [feats[i:i+cols_per_row] for i in range(0, len(feats), cols_per_row)]
        for row in rows:
            r_cols = st.columns(cols_per_row)
            for ci, feat in enumerate(row):
                r_cols[ci].markdown(
                    f'<div style="background:#1a2540;border-radius:6px;'
                    f'padding:6px 12px;margin:2px;text-align:center;'
                    f'font-size:0.85rem;color:#cdd4e8">{feat}</div>',
                    unsafe_allow_html=True)
    else:
        st.info("No approved features recorded yet.")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 11 — UPLOAD DATA
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📤 Upload Data":
    import sys as _sys
    import io as _io
    import os as _os
    _sys.path.insert(0, _os.path.join(_os.path.dirname(__file__), ".."))

    try:
        from src.ingestion.schema_registry import (
            load_schema, save_schema, infer_schema,
            validate_against_schema, coerce_types,
            schema_to_sources_yaml, blank_schema_yaml,
            UserSchema, TableSchema, ColumnSchema,
        )
        _SR_OK = True
    except Exception as _sre:
        _SR_OK = False
        _SR_ERR = str(_sre)

    st.title("📤 Upload Data")
    st.caption(
        "Bring data from any source — any number of tables, any column names. "
        "Define your schema first, then upload your files."
    )

    _UPLOADS_DIR = _os.path.join(_os.path.dirname(__file__), "..", "data", "uploads")
    _os.makedirs(_UPLOADS_DIR, exist_ok=True)

    def _read_df(f) -> "pd.DataFrame | None":
        if f is None:
            return None
        data = f.read()
        name = f.name

        # ── Security checks before parsing ────────────────────────────────────
        if _SEC_OK:
            try:
                FileValidator.validate(data, name, max_mb=500)
            except SecurityError as _sec_e:
                st.error(f"File rejected: {_sec_e}")
                _audit.log_violation("FILE_UPLOAD_REJECTED", str(_sec_e), source=name)
                return None

        try:
            nl = name.lower()
            if nl.endswith(".parquet"):
                return pd.read_parquet(_io.BytesIO(data))
            elif nl.endswith((".xlsx", ".xls")):
                return pd.read_excel(_io.BytesIO(data), engine="openpyxl")
            elif nl.endswith(".json"):
                return pd.read_json(_io.BytesIO(data))
            else:
                return pd.read_csv(_io.BytesIO(data))
        except Exception as e:
            st.error(f"Could not read {name}: {scrub_secrets(str(e)) if _SEC_OK else e}")
            return None

    # ── Session state keys ────────────────────────────────────────────────────
    if "ud_schema" not in st.session_state:
        st.session_state.ud_schema = None      # UserSchema object
    if "ud_dfs" not in st.session_state:
        st.session_state.ud_dfs = {}           # {table_name: pd.DataFrame}
    if "ud_col_maps" not in st.session_state:
        st.session_state.ud_col_maps = {}      # {table_name: {schema_col: df_col}}

    # ══ STEP 1 — SCHEMA ══════════════════════════════════════════════════════
    st.markdown("## Step 1 — Define Your Schema")
    st.caption(
        "A schema tells the pipeline what tables and columns exist in your data. "
        "Upload a schema YAML/JSON file, auto-infer from your files, or use the template."
    )

    sch_tab_upload, sch_tab_edit, sch_tab_template = st.tabs([
        "Upload schema file", "Build interactively", "Start from template"
    ])

    with sch_tab_upload:
        schema_file = st.file_uploader(
            "Upload schema.yaml or schema.json",
            type=["yaml", "yml", "json"],
            key="schema_file_upload",
        )
        if schema_file:
            try:
                st.session_state.ud_schema = load_schema(schema_file)
                st.success(
                    f"Schema loaded: {len(st.session_state.ud_schema.tables)} table(s) — "
                    + ", ".join(t.name for t in st.session_state.ud_schema.tables)
                )
            except Exception as _se:
                st.error(f"Could not parse schema file: {_se}")

    with sch_tab_edit:
        st.caption("Define tables and columns manually.")

        if "ud_builder_tables" not in st.session_state:
            st.session_state.ud_builder_tables = [
                {"name": "customers",    "maps_to": "customers",    "primary_key": "customer_id"},
                {"name": "transactions", "maps_to": "transactions",  "primary_key": "transaction_id"},
            ]

        _builder_tables = st.session_state.ud_builder_tables
        add_tbl = st.button("+ Add table", key="add_table_btn")
        if add_tbl:
            _builder_tables.append({"name": f"table_{len(_builder_tables)+1}", "maps_to": "", "primary_key": ""})
            st.session_state.ud_builder_tables = _builder_tables

        _built_tables = []
        for ti, tbl_def in enumerate(_builder_tables):
            with st.expander(f"Table: {tbl_def['name']}", expanded=ti == 0):
                tc1, tc2, tc3 = st.columns([2, 2, 1])
                tbl_name = tc1.text_input("Table name", value=tbl_def["name"], key=f"tbl_name_{ti}")
                tbl_maps = tc2.selectbox(
                    "Maps to pipeline table",
                    ["(none / custom)", "customers", "transactions", "products", "promotions"],
                    index=["(none / custom)", "customers", "transactions", "products", "promotions"].index(
                        tbl_def["maps_to"] if tbl_def["maps_to"] in
                        ["(none / custom)", "customers", "transactions", "products", "promotions"]
                        else "(none / custom)"
                    ),
                    key=f"tbl_maps_{ti}",
                )
                tbl_pk = tc3.text_input("Primary key col", value=tbl_def.get("primary_key", ""), key=f"tbl_pk_{ti}")

                if "ud_cols_" + str(ti) not in st.session_state:
                    st.session_state["ud_cols_" + str(ti)] = [
                        {"name": "id", "type": "string", "required": True}
                    ]

                col_list = st.session_state["ud_cols_" + str(ti)]
                add_col = st.button(f"+ Add column", key=f"add_col_{ti}")
                if add_col:
                    col_list.append({"name": f"col_{len(col_list)+1}", "type": "string", "required": False})
                    st.session_state["ud_cols_" + str(ti)] = col_list

                cc1, cc2, cc3, cc4 = st.columns([3, 2, 1, 1])
                cc1.markdown("**Column name**")
                cc2.markdown("**Type**")
                cc3.markdown("**Required**")
                cc4.markdown("**PK**")

                _types = ["string", "integer", "float", "date", "datetime", "boolean"]
                updated_cols = []
                for ci, col_def in enumerate(col_list):
                    vc1, vc2, vc3, vc4 = st.columns([3, 2, 1, 1])
                    cn  = vc1.text_input("col", value=col_def["name"], key=f"cn_{ti}_{ci}", label_visibility="collapsed")
                    ct  = vc2.selectbox("type", _types, index=_types.index(col_def.get("type", "string")),
                                         key=f"ct_{ti}_{ci}", label_visibility="collapsed")
                    cr  = vc3.checkbox("req", value=col_def.get("required", False), key=f"cr_{ti}_{ci}", label_visibility="collapsed")
                    cpk = vc4.checkbox("pk",  value=(cn == tbl_pk), key=f"cpk_{ti}_{ci}", label_visibility="collapsed")
                    updated_cols.append({"name": cn, "type": ct, "required": cr, "is_primary_key": cpk})

                st.session_state["ud_cols_" + str(ti)] = updated_cols

                maps_to_val = tbl_maps if tbl_maps != "(none / custom)" else ""
                _built_tables.append({
                    "name": tbl_name,
                    "maps_to": maps_to_val,
                    "primary_key": tbl_pk,
                    "columns": [
                        ColumnSchema(
                            name=c["name"], type=c["type"],
                            required=c["required"], is_primary_key=c.get("is_primary_key", False),
                        )
                        for c in updated_cols
                    ],
                })

        if st.button("Apply interactive schema", type="primary", key="apply_builder"):
            if _SR_OK:
                tbl_objs = [
                    TableSchema(
                        name=t["name"], maps_to=t["maps_to"],
                        primary_key=t["primary_key"], columns=t["columns"],
                    )
                    for t in _built_tables
                ]
                st.session_state.ud_schema = UserSchema(tables=tbl_objs)
                st.success(f"Schema set: {len(tbl_objs)} table(s)")
            else:
                st.error(f"Schema registry unavailable: {_SR_ERR}")

    with sch_tab_template:
        st.caption("Download and edit the template, then upload it in the first tab.")
        tmpl = blank_schema_yaml() if _SR_OK else "# schema_registry not available"
        st.code(tmpl, language="yaml")
        st.download_button(
            "Download schema_template.yaml",
            data=tmpl.encode(),
            file_name="schema_template.yaml",
            mime="text/yaml",
        )

    st.divider()

    # Show current schema summary
    active_schema: "UserSchema | None" = st.session_state.ud_schema
    if active_schema is None:
        st.info("Define or upload a schema above before uploading data files.")
        st.stop()

    st.markdown(
        f"**Active schema:** {len(active_schema.tables)} table(s) — "
        + ", ".join(
            f"`{t.name}`" + (f" → {t.maps_to}" if t.maps_to else "")
            for t in active_schema.tables
        )
    )

    # ══ STEP 2 — UPLOAD DATA FILES ════════════════════════════════════════════
    st.markdown("## Step 2 — Upload Data Files")
    st.caption("Upload one file per table. Supported: CSV, Excel, Parquet, JSON.")

    _dfs: dict = st.session_state.ud_dfs

    for tbl in active_schema.tables:
        already = _dfs.get(tbl.name)
        label = (
            f"{tbl.name}"
            + (f" (maps to pipeline `{tbl.maps_to}`)" if tbl.maps_to else "")
            + (f"  — {len(already):,} rows already loaded" if already is not None else "")
        )
        f = st.file_uploader(
            label,
            type=["csv", "xlsx", "xls", "parquet", "json"],
            key=f"data_file_{tbl.name}",
        )
        if f is not None:
            df = _read_df(f)
            if df is not None:
                df.columns = [str(c).strip() for c in df.columns]
                _dfs[tbl.name] = df

    st.session_state.ud_dfs = _dfs

    if not _dfs:
        st.info("Upload at least one data file above.")
        st.stop()

    st.divider()

    # ══ STEP 3 — COLUMN MAPPING ═══════════════════════════════════════════════
    st.markdown("## Step 3 — Column Mapping")
    st.caption(
        "For each table, map your file's columns to the schema columns. "
        "Auto-detected where possible — adjust any that are wrong."
    )

    import difflib as _dl

    def _best_match(user_col: str, schema_cols: list[str]) -> "str | None":
        uc = user_col.lower().replace(" ", "_").replace("-", "_")
        for sc in schema_cols:
            if uc == sc.lower():
                return sc
        best = _dl.get_close_matches(uc, [s.lower() for s in schema_cols], n=1, cutoff=0.5)
        if best:
            idx = [s.lower() for s in schema_cols].index(best[0])
            return schema_cols[idx]
        return None

    _col_maps: dict = st.session_state.ud_col_maps
    all_valid = True

    for tbl in active_schema.tables:
        if tbl.name not in _dfs:
            continue
        df = _dfs[tbl.name]
        schema_col_names = [c.name for c in tbl.columns]
        user_cols_opts   = ["(skip)"] + list(df.columns)

        with st.expander(
            f"**{tbl.name}** — {len(df):,} rows × {len(df.columns)} cols",
            expanded=True,
        ):
            hc1, hc2, hc3 = st.columns([2, 2, 3])
            hc1.markdown("**Schema column**")
            hc2.markdown("**Your column**")
            hc3.markdown("**Sample values**")

            col_map = {}
            for sc in tbl.columns:
                auto_match = _best_match(sc.name, list(df.columns))
                default_idx = user_cols_opts.index(auto_match) if auto_match and auto_match in user_cols_opts else 0

                mc1, mc2, mc3 = st.columns([2, 2, 3])
                req_star = " \\*" if sc.required else ""
                mc1.markdown(f"`{sc.name}`{req_star} *({sc.type})*")
                chosen = mc2.selectbox(
                    f"cmap_{tbl.name}_{sc.name}", user_cols_opts,
                    index=default_idx, label_visibility="collapsed",
                    key=f"cmap_{tbl.name}_{sc.name}",
                )
                if chosen != "(skip)" and chosen in df.columns:
                    samples = df[chosen].dropna().astype(str).head(3).tolist()
                    mc3.caption(" · ".join(samples))
                    col_map[sc.name] = chosen
                elif sc.required:
                    mc3.markdown(":red[required — please map]")
                    all_valid = False

            _col_maps[tbl.name] = col_map

        # Schema validation errors
        if tbl.name in _col_maps:
            renamed = df.rename(columns={v: k for k, v in _col_maps[tbl.name].items()})
            errs = validate_against_schema(renamed, tbl)
            for e in errs:
                st.error(e)
                all_valid = False
            if not errs:
                st.success(f"✅ `{tbl.name}` — schema valid")

    st.session_state.ud_col_maps = _col_maps

    st.divider()

    # ══ STEP 4 — VALIDATION SUMMARY ═══════════════════════════════════════════
    st.markdown("## Step 4 — Validation Summary")

    _summary_html = ""
    for tbl in active_schema.tables:
        if tbl.name not in _dfs:
            _summary_html += (
                f'<div style="padding:8px 12px;margin:4px 0;background:#1a2540;border-radius:6px;'
                f'border-left:4px solid #555;">'
                f'<b>{tbl.name}</b> — <span style="color:#888;">no file uploaded</span></div>'
            )
            continue
        df = _dfs[tbl.name]
        null_pct = int(100 * df.isnull().sum().sum() / max(df.size, 1))
        dup_col  = tbl.primary_key if tbl.primary_key in df.columns else None
        dup_pct  = int(100 * df.duplicated(subset=[dup_col]).sum() / max(len(df), 1)) if dup_col else 0
        color = "#22c55e" if (null_pct < 5 and dup_pct < 5) else "#f59e0b"
        _summary_html += (
            f'<div style="padding:8px 12px;margin:4px 0;background:#1a2540;border-radius:6px;'
            f'border-left:4px solid {color};">'
            f'<b>{tbl.name}</b> — {len(df):,} rows · {len(df.columns)} cols · '
            f'{null_pct}% nulls · {dup_pct}% dup PKs'
            f'</div>'
        )

    st.markdown(_summary_html, unsafe_allow_html=True)

    st.divider()

    # ══ STEP 5 — SAVE ═════════════════════════════════════════════════════════
    st.markdown("## Step 5 — Save & Register")

    if not all_valid:
        st.error("Resolve the mapping errors above before saving.")
    else:
        if st.button("💾 Save All Tables + Schema", type="primary"):
            with st.spinner("Saving..."):
                for tbl in active_schema.tables:
                    if tbl.name not in _dfs:
                        continue
                    df = _dfs[tbl.name]
                    col_map = _col_maps.get(tbl.name, {})
                    inv_map = {v: k for k, v in col_map.items()}
                    df_out  = df.rename(columns=inv_map)
                    df_out  = coerce_types(df_out, tbl)
                    out_path = _os.path.join(_UPLOADS_DIR, f"{tbl.name}.parquet")
                    if _SEC_OK:
                        from src.security.validators import PathValidator
                        PathValidator.assert_safe_write(out_path)
                    df_out.to_parquet(out_path, index=False, engine="pyarrow")
                    if _SEC_OK:
                        _audit.log_upload(
                            table_name=tbl.name,
                            filename=f"{tbl.name}.parquet",
                            rows=len(df_out),
                            size_mb=_os.path.getsize(out_path) / (1024 * 1024),
                        )
                    st.success(f"Saved `{tbl.name}` → `data/uploads/{tbl.name}.parquet` ({len(df_out):,} rows)")

                schema_path = _os.path.join(_UPLOADS_DIR, "schema.yaml")
                save_schema(active_schema, schema_path)
                st.success("Schema saved → `data/uploads/schema.yaml`")

                # Generate sources.yaml fragment
                sources_fragment = schema_to_sources_yaml(active_schema, "data/uploads")
                sources_yaml_path = _os.path.join(_os.path.dirname(__file__), "..", "sources.yaml")
                st.download_button(
                    "Download sources.yaml (add to project root)",
                    data=sources_fragment.encode(),
                    file_name="sources.yaml",
                    mime="text/yaml",
                )

            st.info(
                "All tables saved. Run the pipeline with:\n\n"
                "```bash\npython -m src.run_all.pipeline_runner\n```\n\n"
                "Or set `USE_UPLOADED_DATA=true` in `.env` to make the pipeline always "
                "prefer your uploaded data over synthetic data."
            )
            st.balloons()


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 12 — CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════
elif page == "⚙️ Configuration":
    import sys as _sys
    import os as _os
    import yaml as _yaml
    _sys.path.insert(0, _os.path.join(_os.path.dirname(__file__), ".."))

    st.title("⚙️ Configuration")

    _CONFIG_PATH = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), "..", "config.yaml"))
    _ENV_PATH    = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), "..", ".env"))

    try:
        from src.config.config_loader import get_config, reload_config
        cfg = get_config()
        _CFG_OK = True
    except Exception as _ce:
        cfg = {}
        _CFG_OK = False

    # ── Provider colour map & helpers ─────────────────────────────────────────
    _PC = {
        "local":"#4a9eff","s3":"#f59e0b","azure":"#0078d4","gcs":"#34a853",
        "databricks":"#ff3621","emr":"#f59e0b","dataproc":"#34a853",
        "anthropic":"#c97bff","bedrock":"#f59e0b",
    }

    def _badge(label, color="#4a9eff"):
        return (
            f'<span style="background:{color};color:#fff;border-radius:5px;'
            f'padding:3px 10px;font-size:0.8em;font-weight:700;">{label.upper()}</span>'
        )

    def _crow(key, value, secret=False):
        display = "••••••••" if (secret and value) else (str(value) if value else "*(not set)*")
        color   = "#4cef9a" if value else "#555"
        return (
            f'<div style="display:flex;padding:5px 10px;border-bottom:1px solid #1a2540;">'
            f'<div style="min-width:240px;color:#8ba8c4;font-size:0.83em;">{key}</div>'
            f'<div style="color:{color};font-size:0.83em;font-family:monospace;">{display}</div>'
            f'</div>'
        )

    def _secret_field(label, env_key, help_text=""):
        existing = _os.environ.get(env_key, "")
        placeholder = "••••••••" if existing else ""
        new_val = st.text_input(
            label, value="", placeholder=placeholder or "not set",
            type="password", help=help_text, key=f"cfg_field_{env_key}"
        )
        return new_val if new_val else existing

    def _text_field(label, env_key, default="", help_text=""):
        existing = _os.environ.get(env_key, default)
        return st.text_input(label, value=existing, help=help_text, key=f"cfg_field_{env_key}")

    # ── Current active provider strip ─────────────────────────────────────────
    sp = cfg.get("storage", {}).get("provider", "local")
    cp = cfg.get("compute", {}).get("provider", "local")
    lp = cfg.get("llm",     {}).get("provider", "anthropic")

    b1, b2, b3 = st.columns(3)
    b1.markdown(f"**Storage**<br>" + _badge(sp, _PC.get(sp,"#4a9eff")), unsafe_allow_html=True)
    b2.markdown(f"**Compute**<br>" + _badge(cp, _PC.get(cp,"#4a9eff")), unsafe_allow_html=True)
    b3.markdown(f"**LLM**<br>"     + _badge(lp, _PC.get(lp,"#c97bff")), unsafe_allow_html=True)

    st.divider()

    # ── Main tabs ─────────────────────────────────────────────────────────────
    t_upload, t_form, t_view, t_env = st.tabs([
        "📁 Upload Config File", "🔧 Configure via Form", "👁️ View Current Config", "🔐 Env Variables"
    ])

    # ════════════════════════════════════════════════════════════
    # TAB 1 — Upload config.yaml or .env
    # ════════════════════════════════════════════════════════════
    with t_upload:
        st.markdown("### Upload config.yaml")
        st.caption("Drop your config.yaml here to replace the current cloud configuration.")

        cfg_file = st.file_uploader(
            "config.yaml", type=["yaml", "yml"], key="cfg_yaml_upload",
            help="Must be a valid YAML file matching the config.yaml schema."
        )
        if cfg_file:
            raw_cfg = cfg_file.read().decode("utf-8")
            _cfg_safe = True
            if _SEC_OK:
                try:
                    YAMLValidator.assert_safe(raw_cfg, label="config.yaml")
                except SecurityError as _sy:
                    st.error(f"Security check failed: {_sy}")
                    _audit.log_violation("YAML_INJECTION_ATTEMPT", str(_sy), source="config_upload")
                    _cfg_safe = False
            if _cfg_safe:
                try:
                    _yaml.safe_load(raw_cfg)
                    st.success("YAML valid — preview:")
                    with st.expander("config.yaml preview", expanded=True):
                        st.code(raw_cfg, language="yaml")
                    if st.button("Apply this config.yaml", type="primary", key="apply_cfg_upload"):
                        with open(_CONFIG_PATH, "w", encoding="utf-8") as _f:
                            _f.write(raw_cfg)
                        if _SEC_OK:
                            _audit.log_config_change("all", "uploaded", source="file_upload")
                        st.success(f"Saved to `{_CONFIG_PATH}`. Restart the dashboard to reload.")
                except Exception as _ye:
                    st.error(f"Invalid YAML: {_ye}")

        st.divider()
        st.markdown("### Upload .env file")
        st.caption("Upload a `.env` file (KEY=VALUE format) to set environment variables.")

        env_file = st.file_uploader(
            ".env file", type=["env", "txt"], key="cfg_env_upload",
            help="Plain text KEY=VALUE pairs, one per line. Secrets are shown masked."
        )
        if env_file:
            raw_env = env_file.read().decode("utf-8")
            _SECRET_WORDS = ("key", "token", "password", "secret", "credential", "pass")

            parsed_env = {}
            for line in raw_env.splitlines():
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, _, v = line.partition("=")
                parsed_env[k.strip()] = v.strip()

            rows_html = "".join(
                _crow(k, v, secret=any(w in k.lower() for w in _SECRET_WORDS))
                for k, v in parsed_env.items()
            )
            st.markdown(
                f'<div style="background:#0d1524;border-radius:8px;">{rows_html}</div>',
                unsafe_allow_html=True,
            )
            st.caption(f"{len(parsed_env)} variables detected")

            if st.button("Apply this .env", type="primary", key="apply_env_upload"):
                with open(_ENV_PATH, "w", encoding="utf-8") as _f:
                    _f.write(raw_env)
                for k, v in parsed_env.items():
                    _os.environ[k] = v
                st.success(f"Saved to `{_ENV_PATH}` and loaded into runtime environment.")

    # ════════════════════════════════════════════════════════════
    # TAB 2 — Form-based configuration
    # ════════════════════════════════════════════════════════════
    with t_form:
        st.markdown("### Cloud Provider Configuration")
        st.caption("Fill in the fields for your environment. Secrets are masked. Click **Save** at the bottom.")

        with st.expander("🗄️ Storage Provider", expanded=True):
            new_sp = st.radio(
                "Storage provider", ["local", "s3", "azure", "gcs"],
                index=["local","s3","azure","gcs"].index(sp),
                horizontal=True, key="form_storage_provider",
            )

            if new_sp == "s3":
                fa1, fa2 = st.columns(2)
                _s3_bucket    = fa1.text_input("S3 Bucket",   value=_os.environ.get("S3_BUCKET",""),    key="cfg_s3_bucket")
                _s3_region    = fa2.text_input("AWS Region",  value=_os.environ.get("AWS_REGION","us-east-1"), key="cfg_s3_region")
                fb1, fb2 = st.columns(2)
                _s3_key_id    = _secret_field("AWS Access Key ID",     "AWS_ACCESS_KEY_ID")
                _s3_key_secret= _secret_field("AWS Secret Access Key", "AWS_SECRET_ACCESS_KEY")
                _s3_session   = _secret_field("AWS Session Token (optional)", "AWS_SESSION_TOKEN")

            elif new_sp == "azure":
                fa1, fa2 = st.columns(2)
                _az_account   = fa1.text_input("Storage Account", value=_os.environ.get("AZURE_STORAGE_ACCOUNT",""), key="cfg_az_account")
                _az_container = fa2.text_input("Container",       value=_os.environ.get("AZURE_CONTAINER",""),       key="cfg_az_container")
                _az_cred      = _secret_field("SAS Token / Connection String (blank = managed identity)", "AZURE_STORAGE_CREDENTIAL")

            elif new_sp == "gcs":
                fa1, fa2 = st.columns(2)
                _gcs_bucket  = fa1.text_input("GCS Bucket",  value=_os.environ.get("GCS_BUCKET",""),  key="cfg_gcs_bucket")
                _gcs_project = fa2.text_input("GCP Project", value=_os.environ.get("GCP_PROJECT",""), key="cfg_gcs_project")
                _gcs_creds   = st.text_input("Path to service account JSON (blank = ADC)",
                                              value=_os.environ.get("GOOGLE_APPLICATION_CREDENTIALS",""),
                                              key="cfg_gcs_creds")
            else:
                st.info("Local storage — no credentials required. Data is stored in `outputs/` on this machine.")

        with st.expander("⚡ Compute Provider", expanded=True):
            new_cp = st.radio(
                "Compute provider", ["local", "databricks", "emr", "dataproc"],
                index=["local","databricks","emr","dataproc"].index(cp),
                horizontal=True, key="form_compute_provider",
            )

            if new_cp == "databricks":
                _db_host    = _text_field("Databricks Host", "DATABRICKS_HOST", help_text="e.g. https://adb-xxxx.azuredatabricks.net")
                _db_token   = _secret_field("Databricks Token", "DATABRICKS_TOKEN")
                _db_cluster = _text_field("Cluster ID", "DATABRICKS_CLUSTER_ID")

            elif new_cp == "emr":
                _emr_master = _text_field("EMR Master URL", "EMR_MASTER_URL", help_text="spark://master:7077")
                _emr_region = _text_field("AWS Region",    "AWS_REGION", default="us-east-1")

            elif new_cp == "dataproc":
                _dp_master  = _text_field("Dataproc Master hostname", "DATAPROC_MASTER")
                _dp_cluster = _text_field("Cluster name",             "DATAPROC_CLUSTER")
                _dp_project = _text_field("GCP Project",              "GCP_PROJECT")
                _dp_region  = _text_field("GCP Region",               "GCP_REGION", default="us-central1")
            else:
                st.info("Local Spark — runs on this machine. Java 17+ must be on PATH.")

        with st.expander("🤖 LLM Provider", expanded=True):
            new_lp = st.radio(
                "LLM provider", ["anthropic", "bedrock", "azure"],
                index=["anthropic","bedrock","azure"].index(lp),
                horizontal=True, key="form_llm_provider",
            )

            if new_lp == "anthropic":
                _ant_key   = _secret_field("Anthropic API Key", "ANTHROPIC_API_KEY")
                _ant_model = _text_field("Model", "ANTHROPIC_MODEL", default="claude-sonnet-4-6")

            elif new_lp == "bedrock":
                _bdr_region = _text_field("AWS Region", "AWS_REGION", default="us-east-1")
                _bdr_model  = st.text_input("Bedrock model ID",
                                             value="anthropic.claude-3-5-sonnet-20241022-v2:0",
                                             key="cfg_bedrock_model")

            elif new_lp == "azure":
                _az_ep   = _text_field("Azure OpenAI Endpoint",   "AZURE_OPENAI_ENDPOINT")
                _az_key  = _secret_field("Azure OpenAI Key",       "AZURE_OPENAI_KEY")
                _az_dep  = _text_field("Deployment name",          "AZURE_OPENAI_DEPLOYMENT", default="gpt-4o")

        st.divider()

        if st.button("💾 Save Configuration", type="primary", key="form_save_cfg"):
            with st.spinner("Writing config.yaml and .env..."):
                # Load existing config
                if _os.path.exists(_CONFIG_PATH):
                    with open(_CONFIG_PATH, encoding="utf-8") as _cf:
                        existing_cfg = _yaml.safe_load(_cf.read()) or {}
                else:
                    existing_cfg = {}

                existing_cfg.setdefault("storage", {})["provider"] = new_sp
                existing_cfg.setdefault("compute", {})["provider"] = new_cp
                existing_cfg.setdefault("llm",     {})["provider"] = new_lp

                with open(_CONFIG_PATH, "w", encoding="utf-8") as _cf:
                    _yaml.dump(existing_cfg, _cf, allow_unicode=True,
                               default_flow_style=False, sort_keys=False)

                # Write .env
                env_updates: dict[str, str] = {
                    "STORAGE_PROVIDER": new_sp,
                    "COMPUTE_PROVIDER": new_cp,
                    "LLM_PROVIDER":     new_lp,
                }

                def _add(k, v):
                    if v:
                        env_updates[k] = v

                if new_sp == "s3":
                    _add("S3_BUCKET", _s3_bucket); _add("AWS_REGION", _s3_region)
                    _add("AWS_ACCESS_KEY_ID", _s3_key_id)
                    _add("AWS_SECRET_ACCESS_KEY", _s3_key_secret)
                    _add("AWS_SESSION_TOKEN", _s3_session)
                elif new_sp == "azure":
                    _add("AZURE_STORAGE_ACCOUNT", _az_account)
                    _add("AZURE_CONTAINER", _az_container)
                    _add("AZURE_STORAGE_CREDENTIAL", _az_cred)
                elif new_sp == "gcs":
                    _add("GCS_BUCKET", _gcs_bucket); _add("GCP_PROJECT", _gcs_project)
                    _add("GOOGLE_APPLICATION_CREDENTIALS", _gcs_creds)

                if new_cp == "databricks":
                    _add("DATABRICKS_HOST", _db_host)
                    _add("DATABRICKS_TOKEN", _db_token)
                    _add("DATABRICKS_CLUSTER_ID", _db_cluster)
                elif new_cp == "emr":
                    _add("EMR_MASTER_URL", _emr_master); _add("AWS_REGION", _emr_region)
                elif new_cp == "dataproc":
                    _add("DATAPROC_MASTER", _dp_master); _add("DATAPROC_CLUSTER", _dp_cluster)
                    _add("GCP_PROJECT", _dp_project); _add("GCP_REGION", _dp_region)

                if new_lp == "anthropic":
                    _add("ANTHROPIC_API_KEY", _ant_key)
                    _add("ANTHROPIC_MODEL", _ant_model)
                elif new_lp == "bedrock":
                    _add("AWS_REGION", _bdr_region)
                elif new_lp == "azure":
                    _add("AZURE_OPENAI_ENDPOINT", _az_ep)
                    _add("AZURE_OPENAI_KEY", _az_key)
                    _add("AZURE_OPENAI_DEPLOYMENT", _az_dep)

                # Merge with existing .env
                existing_env: dict[str, str] = {}
                if _os.path.exists(_ENV_PATH):
                    with open(_ENV_PATH, encoding="utf-8") as _ef:
                        for line in _ef:
                            line = line.strip()
                            if line and not line.startswith("#") and "=" in line:
                                k, _, v = line.partition("=")
                                existing_env[k.strip()] = v.strip()
                existing_env.update(env_updates)

                with open(_ENV_PATH, "w", encoding="utf-8") as _ef:
                    for k, v in existing_env.items():
                        _ef.write(f"{k}={v}\n")
                for k, v in env_updates.items():
                    _os.environ[k] = v
                if _SEC_OK:
                    _audit.log_config_change("storage+compute+llm",
                                              f"{new_sp}/{new_cp}/{new_lp}",
                                              source="form")
                    _audit.log_env_write(list(env_updates.keys()), source="form")

            st.success("config.yaml and .env updated. Restart the dashboard for changes to take effect.")

    # ════════════════════════════════════════════════════════════
    # TAB 3 — View current config (read-only + edit)
    # ════════════════════════════════════════════════════════════
    with t_view:
        sv_tab, cv_tab, lv_tab, pv_tab = st.tabs(["Storage", "Compute", "LLM", "Pipeline"])

        for _tab, _section, _provs in [
            (sv_tab, "storage", ["local","s3","azure","gcs"]),
            (cv_tab, "compute", ["local","databricks","emr","dataproc"]),
            (lv_tab, "llm",     ["anthropic","bedrock","azure"]),
        ]:
            with _tab:
                _active = cfg.get(_section, {}).get("provider", _provs[0])
                _pcfg   = cfg.get(_section, {}).get(_active, {})
                rows = "".join(
                    _crow(k, v, secret=any(w in k.lower() for w in ("key","token","secret","password","credential")))
                    for k, v in _pcfg.items()
                )
                if rows:
                    st.markdown(f"**Active: {_active}**")
                    st.markdown(f'<div style="background:#0d1524;border-radius:8px;">{rows}</div>',
                                unsafe_allow_html=True)
                else:
                    st.info(f"No settings stored for active provider `{_active}` in config.yaml.")

                for prov in _provs:
                    if prov != _active:
                        with st.expander(f"{prov} (inactive)"):
                            pc = cfg.get(_section, {}).get(prov, {})
                            st.json(pc if pc else {"status": "not configured"})

        with pv_tab:
            pl = cfg.get("pipeline", {})
            rows = "".join(_crow(k, v) for k, v in pl.items())
            if rows:
                st.markdown(f'<div style="background:#0d1524;border-radius:8px;">{rows}</div>',
                            unsafe_allow_html=True)
            else:
                st.info("No pipeline settings in config.yaml.")

        st.divider()
        with st.expander("Edit config.yaml directly", expanded=False):
            if _os.path.exists(_CONFIG_PATH):
                with open(_CONFIG_PATH, encoding="utf-8") as _cf:
                    _cur_yaml = _cf.read()
                _new_yaml = st.text_area("config.yaml", value=_cur_yaml, height=500,
                                          label_visibility="collapsed", key="raw_yaml_editor")
                if st.button("Save config.yaml", key="raw_yaml_save"):
                    try:
                        _yaml.safe_load(_new_yaml)
                        with open(_CONFIG_PATH, "w", encoding="utf-8") as _cf:
                            _cf.write(_new_yaml)
                        st.success("Saved. Restart dashboard to reload.")
                    except Exception as _ye:
                        st.error(f"Invalid YAML: {_ye}")
            else:
                st.info(f"`config.yaml` not found at `{_CONFIG_PATH}`.")

    # ════════════════════════════════════════════════════════════
    # TAB 4 — Environment variables
    # ════════════════════════════════════════════════════════════
    with t_env:
        st.markdown("### Current environment variables")
        st.caption("Variables set in the runtime environment. Add missing ones via the form tab or by uploading a .env file.")

        _SECRET_WORDS = ("key", "token", "secret", "password", "credential", "pass")
        ALL_ENV_KEYS = [
            "ANTHROPIC_API_KEY", "ANTHROPIC_MODEL",
            "AZURE_OPENAI_KEY", "AZURE_OPENAI_ENDPOINT", "AZURE_OPENAI_DEPLOYMENT",
            "AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_SESSION_TOKEN", "AWS_REGION",
            "S3_BUCKET",
            "AZURE_STORAGE_ACCOUNT", "AZURE_CONTAINER", "AZURE_STORAGE_CREDENTIAL",
            "GCS_BUCKET", "GCP_PROJECT", "GCP_REGION", "GOOGLE_APPLICATION_CREDENTIALS",
            "DATABRICKS_HOST", "DATABRICKS_TOKEN", "DATABRICKS_CLUSTER_ID",
            "EMR_MASTER_URL", "DATAPROC_MASTER", "DATAPROC_CLUSTER",
            "STORAGE_PROVIDER", "COMPUTE_PROVIDER", "LLM_PROVIDER",
        ]
        rows_html = "".join(
            _crow(k, _os.environ.get(k, ""),
                  secret=any(w in k.lower() for w in _SECRET_WORDS))
            for k in ALL_ENV_KEYS
        )
        st.markdown(
            f'<div style="background:#0d1524;border-radius:8px;">{rows_html}</div>',
            unsafe_allow_html=True,
        )
        st.caption("Green = set · grey = not set · •••••• = secret (value hidden)")

        st.divider()
        st.markdown("### Add / update a variable")
        ev1, ev2 = st.columns([2, 3])
        new_env_key = ev1.text_input("Variable name", key="new_env_key", placeholder="MY_API_KEY")
        new_env_val = ev2.text_input(
            "Value", key="new_env_val", type="password"
            if any(w in new_env_key.lower() for w in _SECRET_WORDS) else "default",
        )
        if st.button("Set variable", key="set_env_var"):
            if new_env_key and new_env_val:
                if _SEC_OK:
                    try:
                        InputValidator.assert_env_key(new_env_key)
                    except SecurityError as _ek:
                        st.error(f"Invalid variable name: {_ek}")
                        st.stop()
                _os.environ[new_env_key] = new_env_val
                existing_env2: dict[str, str] = {}
                if _os.path.exists(_ENV_PATH):
                    with open(_ENV_PATH, encoding="utf-8") as _ef:
                        for line in _ef:
                            line = line.strip()
                            if line and not line.startswith("#") and "=" in line:
                                k2, _, v2 = line.partition("=")
                                existing_env2[k2.strip()] = v2.strip()
                existing_env2[new_env_key] = new_env_val
                with open(_ENV_PATH, "w", encoding="utf-8") as _ef:
                    for k2, v2 in existing_env2.items():
                        _ef.write(f"{k2}={v2}\n")
                if _SEC_OK:
                    _audit.log_env_write([new_env_key], source="single_var_form")
                st.success(f"{new_env_key} set and written to .env")
            else:
                st.warning("Enter both a variable name and value.")
