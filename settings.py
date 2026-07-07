import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

PATHS = {
    "raw_data":        str(ROOT / "data" / "raw"),
    "staging":         str(ROOT / "data" / "staging"),
    "processed_data":  str(ROOT / "data" / "processed"),      # PROD (reviewed + promoted)
    "external_data":   str(ROOT / "data" / "external"),
    "models":          str(ROOT / "outputs" / "models"),
    "reports":         str(ROOT / "outputs" / "reports"),
    "visualizations":  str(ROOT / "outputs" / "visualizations"),
}

# Dev environment: agents write here first; reviewed before promotion to PROD (PATHS above)
ENVIRONMENTS = {
    "dev": {
        "processed":    str(ROOT / "data" / "dev" / "processed"),
        "staging":      str(ROOT / "data" / "dev" / "staging"),
        "models":       str(ROOT / "outputs" / "dev" / "models"),
        "reports":      str(ROOT / "outputs" / "dev" / "reports"),
    },
    "prod": {
        "processed":    str(ROOT / "data" / "processed"),     # same as PATHS["processed_data"]
        "staging":      str(ROOT / "data" / "staging"),
        "models":       str(ROOT / "outputs" / "models"),
        "reports":      str(ROOT / "outputs" / "reports"),
    },
}

# Review authority chain: who must sign off on each role's work before it goes to PROD
REVIEW_CHAIN = {
    "data_engineer_1":    ["de_lead"],
    "data_engineer_2":    ["de_lead"],
    "de_lead":            ["code_reviewer", "project_manager"],
    "data_scientist_1":   ["senior_data_scientist"],
    "data_scientist_2":   ["senior_data_scientist", "ds_lead"],
    "senior_data_scientist": ["ds_lead"],
    "ml_engineer":        ["ds_lead"],
    "business_analyst_1": ["business_lead"],
    "business_analyst_2": ["business_lead"],
    "marketing_analyst":  ["business_lead"],
    "finance_analyst":    ["business_lead"],
    "business_lead":      ["project_manager", "ceo"],
    "ds_lead":            ["project_manager", "ceo"],
    "project_manager":    ["ceo"],
}

MODEL_PARAMS = {
    "kmeans": {"n_clusters": 5, "random_state": 42, "n_init": 10, "max_iter": 300},
    "dbscan": {"eps": 0.5, "min_samples": 5},
    "nmf":    {"n_components": 20, "random_state": 42, "max_iter": 200},
    "als":    {"factors": 50, "iterations": 20, "regularization": 0.01},
}

FEATURE_COLS = {
    "rfm":       ["recency", "frequency", "monetary"],
    "clv":       ["clv_score"],
    "promo":     ["promo_sensitivity_score"],
    "segmentation_features": [
        "recency", "frequency", "monetary",
        "clv_score", "promo_sensitivity_score",
        "avg_basket_size", "purchase_frequency",
    ],
}

ANTHROPIC_MODEL = "claude-sonnet-4-6"
LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"

PROJECT_NAME = "Trade Promo Optimization — Customer Recommendation & Segmentation"
