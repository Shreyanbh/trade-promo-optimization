import os
import json
import pandas as pd
from src.config.settings import PATHS
from src.utils.file_helpers import write_json
from src.utils.logger import get_logger

log = get_logger(__name__)


def feed_dashboard(
    segment_profiles: pd.DataFrame | None,
    recommendations: pd.DataFrame | None,
    kpi_metrics: dict,
    project_state_snapshot: dict,
    activity_stream: list[dict],
) -> dict[str, str]:
    out = PATHS["reports"]
    os.makedirs(out, exist_ok=True)
    paths = {}

    if segment_profiles is not None:
        path = os.path.join(out, "segment_summary.json")
        write_json(segment_profiles.reset_index().to_dict(orient="records"), path)
        paths["segment_summary"] = path

    if recommendations is not None:
        path = os.path.join(out, "recommendations_sample.parquet")
        recommendations.to_parquet(path, index=False)
        paths["recommendations"] = path

    path = os.path.join(out, "kpi_metrics.json")
    write_json(kpi_metrics, path)
    paths["kpi_metrics"] = path

    path = os.path.join(out, "project_state.json")
    write_json(project_state_snapshot, path)
    paths["project_state"] = path

    path = os.path.join(out, "agent_activity.json")
    write_json(activity_stream, path)
    paths["agent_activity"] = path

    log.info(f"Dashboard data written to {out}")
    return paths
