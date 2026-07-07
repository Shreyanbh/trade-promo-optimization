import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.decomposition import NMF
from src.config.settings import MODEL_PARAMS
from src.utils.logger import get_logger

log = get_logger(__name__)


class CollaborativeFilterRecommender:
    def __init__(self):
        self.model = None
        self.user_factors = None
        self.item_factors = None
        self.user_index: dict = {}
        self.item_index: dict = {}
        self.item_index_inv: dict = {}

    def _build_matrix(self, transactions: pd.DataFrame) -> csr_matrix:
        users = transactions["customer_id"].unique().tolist()
        items = transactions["product_id"].unique().tolist()
        self.user_index = {u: i for i, u in enumerate(users)}
        self.item_index = {it: i for i, it in enumerate(items)}
        self.item_index_inv = {i: it for it, i in self.item_index.items()}

        rows = transactions["customer_id"].map(self.user_index)
        cols = transactions["product_id"].map(self.item_index)
        data = transactions["amount"].values
        return csr_matrix((data, (rows, cols)), shape=(len(users), len(items)))

    def fit(self, transactions: pd.DataFrame) -> "CollaborativeFilterRecommender":
        log.info("Building user-item matrix for NMF recommender …")
        matrix = self._build_matrix(transactions)

        try:
            import implicit
            als = implicit.als.AlternatingLeastSquares(
                factors=MODEL_PARAMS["als"]["factors"],
                iterations=MODEL_PARAMS["als"]["iterations"],
                regularization=MODEL_PARAMS["als"]["regularization"],
            )
            # implicit >= 0.5 takes user-item matrix directly
            als.fit(matrix)
            self.user_factors = als.user_factors   # shape: (n_users, factors)
            self.item_factors = als.item_factors   # shape: (n_items, factors)
            self.model = als
            self._backend = "als"
            log.info("Recommender fitted with ALS (implicit)")
        except ImportError:
            params = MODEL_PARAMS["nmf"]
            nmf = NMF(**params)
            self.user_factors = nmf.fit_transform(matrix)
            self.item_factors = nmf.components_.T
            self.model = nmf
            self._backend = "nmf"
            log.info("Recommender fitted with NMF (sklearn fallback)")

        return self

    def recommend(self, customer_id: str, top_n: int = 10) -> list[dict]:
        if customer_id not in self.user_index:
            return []
        uid = self.user_index[customer_id]
        scores = self.user_factors[uid] @ self.item_factors.T
        top_items = np.argsort(scores)[::-1][:top_n]
        return [
            {"product_id": self.item_index_inv[i], "score": float(scores[i])}
            for i in top_items
        ]

    def recommend_batch(self, customer_ids: list[str], top_n: int = 10) -> pd.DataFrame:
        rows = []
        for cid in customer_ids:
            for rec in self.recommend(cid, top_n):
                rows.append({"customer_id": cid, **rec})
        return pd.DataFrame(rows)
