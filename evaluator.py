import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_score
from src.utils.logger import get_logger

log = get_logger(__name__)


def evaluate_segmentation(X: np.ndarray, labels: np.ndarray) -> dict:
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    valid = labels != -1
    sil = silhouette_score(X[valid], labels[valid]) if valid.sum() > 1 and n_clusters > 1 else 0.0
    metrics = {
        "n_clusters": n_clusters,
        "silhouette_score": round(float(sil), 4),
        "noise_points": int((labels == -1).sum()),
    }
    log.info(f"Segmentation metrics: {metrics}")
    return metrics


def precision_at_k(recommended: list, relevant: set, k: int) -> float:
    hits = sum(1 for item in recommended[:k] if item in relevant)
    return hits / k


def recall_at_k(recommended: list, relevant: set, k: int) -> float:
    hits = sum(1 for item in recommended[:k] if item in relevant)
    return hits / len(relevant) if relevant else 0.0


def ndcg_at_k(recommended: list, relevant: set, k: int) -> float:
    dcg = sum(
        1 / np.log2(i + 2)
        for i, item in enumerate(recommended[:k])
        if item in relevant
    )
    idcg = sum(1 / np.log2(i + 2) for i in range(min(len(relevant), k)))
    return dcg / idcg if idcg > 0 else 0.0


def evaluate_recommender(
    recommendations: pd.DataFrame,
    ground_truth: pd.DataFrame,
    k: int = 10,
) -> dict:
    precisions, recalls, ndcgs = [], [], []
    for cid, rec_group in recommendations.groupby("customer_id"):
        rec_items = rec_group["product_id"].tolist()
        rel_items = set(ground_truth[ground_truth["customer_id"] == cid]["product_id"])
        if not rel_items:
            continue
        precisions.append(precision_at_k(rec_items, rel_items, k))
        recalls.append(recall_at_k(rec_items, rel_items, k))
        ndcgs.append(ndcg_at_k(rec_items, rel_items, k))

    metrics = {
        f"precision@{k}": round(np.mean(precisions), 4) if precisions else 0.0,
        f"recall@{k}":    round(np.mean(recalls), 4) if recalls else 0.0,
        f"ndcg@{k}":      round(np.mean(ndcgs), 4) if ndcgs else 0.0,
        "n_users_evaluated": len(precisions),
    }
    log.info(f"Recommender metrics: {metrics}")
    return metrics
