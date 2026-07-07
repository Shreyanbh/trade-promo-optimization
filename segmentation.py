import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from src.config.settings import MODEL_PARAMS
from src.utils.logger import get_logger

log = get_logger(__name__)

FEATURE_COLS_EXCLUDE = ["customer_id"]


class SegmentationModel:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.labels_ = None
        self.cluster_profiles_ = None
        self.silhouette_ = None
        self.algorithm = None

    def _get_features(self, df: pd.DataFrame) -> np.ndarray:
        cols = [c for c in df.columns if c not in FEATURE_COLS_EXCLUDE]
        num_df = df[cols].select_dtypes(include="number")
        return self.scaler.fit_transform(num_df), list(num_df.columns)

    def fit_kmeans(self, df: pd.DataFrame, n_clusters: int = None) -> "SegmentationModel":
        X, feat_cols = self._get_features(df)
        self.algorithm = "kmeans"

        if n_clusters is None:
            n_clusters = self._select_k(X)

        params = MODEL_PARAMS["kmeans"].copy()
        params["n_clusters"] = n_clusters
        self.model = KMeans(**params)
        self.labels_ = self.model.fit_predict(X)

        if len(set(self.labels_)) > 1:
            self.silhouette_ = round(silhouette_score(X, self.labels_), 4)
        else:
            self.silhouette_ = 0.0

        log.info(f"KMeans k={n_clusters} | silhouette={self.silhouette_}")
        self.cluster_profiles_ = self._build_profiles(df, feat_cols)
        return self

    def fit_dbscan(self, df: pd.DataFrame) -> "SegmentationModel":
        X, feat_cols = self._get_features(df)
        self.algorithm = "dbscan"
        params = MODEL_PARAMS["dbscan"]
        self.model = DBSCAN(**params)
        self.labels_ = self.model.fit_predict(X)

        valid = self.labels_ != -1
        if valid.sum() > 1 and len(set(self.labels_[valid])) > 1:
            self.silhouette_ = round(silhouette_score(X[valid], self.labels_[valid]), 4)
        else:
            self.silhouette_ = 0.0

        log.info(f"DBSCAN | n_clusters={len(set(self.labels_)) - (1 if -1 in self.labels_ else 0)} | silhouette={self.silhouette_}")
        self.cluster_profiles_ = self._build_profiles(df, feat_cols)
        return self

    def _select_k(self, X: np.ndarray) -> int:
        best_k, best_score = 2, -1
        for k in range(2, 9):
            km = KMeans(n_clusters=k, random_state=42, n_init=5)
            labels = km.fit_predict(X)
            score = silhouette_score(X, labels)
            if score > best_score:
                best_score, best_k = score, k
        log.info(f"Auto-selected k={best_k} (silhouette={best_score:.4f})")
        return best_k

    def _build_profiles(self, df: pd.DataFrame, feat_cols: list[str]) -> pd.DataFrame:
        tmp = df.copy()
        tmp["cluster"] = self.labels_
        num_cols = [c for c in feat_cols if c in tmp.columns]
        profiles = tmp.groupby("cluster")[num_cols].mean().round(3)
        profiles["size"] = tmp.groupby("cluster").size()
        return profiles

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        X, _ = self._get_features(df)
        return self.model.predict(X)
