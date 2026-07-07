import os
import json
import pandas as pd
from jinja2 import Template
from src.config.settings import PATHS
from src.utils.logger import get_logger

log = get_logger(__name__)

REPORT_TEMPLATE = """
<!DOCTYPE html><html><head><meta charset="utf-8">
<title>Trade Promo Optimization Report</title>
<style>
  body { font-family: Arial, sans-serif; margin: 40px; color: #333; }
  h1 { color: #1a73e8; } h2 { color: #444; border-bottom: 1px solid #ddd; }
  table { border-collapse: collapse; width: 100%; }
  th, td { border: 1px solid #ccc; padding: 8px; text-align: left; }
  th { background: #f0f4f8; }
  .metric { display: inline-block; margin: 8px 16px 8px 0; padding: 12px 20px;
            background: #e8f0fe; border-radius: 6px; font-size: 1.1em; }
</style></head><body>
<h1>{{ project_name }}</h1>
<p><em>Generated: {{ timestamp }}</em></p>

<h2>Project Milestones</h2>
<table><tr><th>Phase</th><th>Status</th></tr>
{% for phase, status in milestones.items() %}
<tr><td>{{ phase }}</td><td>{{ status }}</td></tr>
{% endfor %}
</table>

<h2>Customer Segments</h2>
{% if cluster_profiles %}
<table><tr>
{% for col in cluster_profiles[0].keys() %}<th>{{ col }}</th>{% endfor %}
</tr>
{% for row in cluster_profiles %}
<tr>{% for v in row.values() %}<td>{{ v }}</td>{% endfor %}</tr>
{% endfor %}
</table>
{% else %}<p>Not yet available.</p>{% endif %}

<h2>Model Performance</h2>
{% for name, meta in models.items() %}
<h3>{{ name }}</h3>
{% for k, v in meta.items() %}
<span class="metric"><strong>{{ k }}</strong>: {{ v }}</span>
{% endfor %}
{% endfor %}

<h2>Business KPIs</h2>
{% for k, v in kpis.items() %}
<span class="metric"><strong>{{ k }}</strong>: {{ v }}</span>
{% endfor %}
</body></html>
"""


def generate_report(
    milestones: dict,
    cluster_profiles: pd.DataFrame | None,
    models: dict,
    kpis: dict,
) -> str:
    from datetime import datetime
    os.makedirs(PATHS["reports"], exist_ok=True)
    profiles_data = []
    if cluster_profiles is not None and not cluster_profiles.empty:
        profiles_data = cluster_profiles.reset_index().round(3).to_dict(orient="records")

    html = Template(REPORT_TEMPLATE).render(
        project_name="Trade Promo Optimization — Customer Recommendation & Segmentation",
        timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        milestones=milestones,
        cluster_profiles=profiles_data,
        models=models,
        kpis=kpis,
    )
    path = os.path.join(PATHS["reports"], "final_report.html")
    with open(path, "w") as f:
        f.write(html)
    log.info(f"Report generated -> {path}")
    return path
