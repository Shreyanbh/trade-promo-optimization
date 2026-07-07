import os
import json
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from src.config.settings import PATHS
from src.utils.logger import get_logger

log = get_logger(__name__)


class EDARunner:
    def __init__(self):
        self.viz_dir = PATHS["visualizations"]
        self.report_dir = PATHS["reports"]
        os.makedirs(self.viz_dir, exist_ok=True)
        os.makedirs(self.report_dir, exist_ok=True)

    def run(self, df: pd.DataFrame, dataset_name: str) -> dict:
        log.info(f"Running EDA on {dataset_name}")
        summary = {
            "dataset_name": dataset_name,
            "shape":        list(df.shape),
            "columns":      list(df.columns),
            "dtypes":       df.dtypes.astype(str).to_dict(),
            "null_rates":   (df.isnull().mean() * 100).round(2).to_dict(),
            "describe":     df.describe(include="all").to_dict(),
        }

        self._plot_missing(df, dataset_name)
        num_df = df.select_dtypes(include="number")
        if not num_df.empty:
            self._plot_distributions(num_df, dataset_name)
            self._plot_correlation(num_df, dataset_name)

        report_path = os.path.join(self.report_dir, f"eda_{dataset_name}.md")
        self._write_md_report(summary, report_path)
        summary["report_path"] = report_path
        log.info(f"EDA complete -> {report_path}")
        return summary

    def _plot_missing(self, df: pd.DataFrame, name: str) -> None:
        null_rates = df.isnull().mean().sort_values(ascending=False)
        if null_rates.sum() == 0:
            return
        fig, ax = plt.subplots(figsize=(10, 4))
        null_rates[null_rates > 0].plot(kind="bar", ax=ax, color="salmon")
        ax.set_title(f"Missing Value Rates — {name}")
        ax.set_ylabel("Null Rate")
        fig.tight_layout()
        fig.savefig(os.path.join(self.viz_dir, f"eda_{name}_missing.png"), dpi=100)
        plt.close(fig)

    def _plot_distributions(self, df: pd.DataFrame, name: str) -> None:
        cols = df.columns[:6]
        fig, axes = plt.subplots(1, len(cols), figsize=(4 * len(cols), 4))
        if len(cols) == 1:
            axes = [axes]
        for ax, col in zip(axes, cols):
            df[col].dropna().hist(bins=30, ax=ax, color="steelblue", edgecolor="white")
            ax.set_title(col)
        fig.suptitle(f"Distributions — {name}")
        fig.tight_layout()
        fig.savefig(os.path.join(self.viz_dir, f"eda_{name}_distributions.png"), dpi=100)
        plt.close(fig)

    def _plot_correlation(self, df: pd.DataFrame, name: str) -> None:
        if df.shape[1] < 2:
            return
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
        ax.set_title(f"Correlation Matrix — {name}")
        fig.tight_layout()
        fig.savefig(os.path.join(self.viz_dir, f"eda_{name}_correlation.png"), dpi=100)
        plt.close(fig)

    def _write_md_report(self, summary: dict, path: str) -> None:
        lines = [
            f"# EDA Report: {summary['dataset_name']}",
            f"\n## Shape\n- Rows: {summary['shape'][0]}, Columns: {summary['shape'][1]}",
            f"\n## Columns\n" + ", ".join(f"`{c}`" for c in summary["columns"]),
            "\n## Null Rates",
        ]
        for col, rate in summary["null_rates"].items():
            if rate > 0:
                lines.append(f"- `{col}`: {rate:.1f}%")
        lines.append("\n## Descriptive Statistics\n```")
        lines.append(pd.DataFrame(summary["describe"]).to_string())
        lines.append("```")
        with open(path, "w") as f:
            f.write("\n".join(lines))
