"""
SparkFeatureEngineer — PySpark feature engineering at 5M-customer scale.

Computes the same feature matrix as feature_engineer.py but fully in PySpark:
  - RFM (recency, frequency, monetary, avg_order_value, max_basket_size)
  - CLV proxy (frequency * avg_order_value / normalization)
  - Promo sensitivity (promo transactions / total transactions)
  - Category affinity (spend fraction per category, pivoted to columns)
  - Channel preference (most-used channel, encoded)

Returns the full 5M-row Spark DataFrame.  The caller is responsible for
sampling before handing off to sklearn / implicit.
"""
import os

from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F
from pyspark.sql.window import Window

from src.config.settings import PATHS
from src.utils.logger import get_logger

log = get_logger("spark_feature_engineer")

CATEGORIES     = ["beverages", "snacks", "dairy", "produce", "household"]
CHANNELS       = ["in-store", "online", "mobile-app", "click-and-collect"]
REFERENCE_DATE = "2024-01-01"
STAGING        = PATHS["staging"]
PROCESSED      = PATHS["processed_data"]


class SparkFeatureEngineer:
    """Builds 5M-row feature matrix entirely in PySpark."""

    def build_feature_matrix(self,
                              spark:        SparkSession,
                              customers_df: DataFrame,
                              txn_df:       DataFrame,
                              products_df:  DataFrame,
                              promos_df:    DataFrame) -> DataFrame:
        """
        Returns a Spark DataFrame with one row per customer.
        Writes result to data/processed/feature_matrix_spark/ (Parquet).
        """
        log.info("SparkFeatureEngineer: building feature matrix...")

        # ── 1. RFM ───────────────────────────────────────────────────────────
        rfm = self._rfm(txn_df)
        log.info("  RFM computed")

        # ── 2. CLV proxy ─────────────────────────────────────────────────────
        rfm = rfm.withColumn(
            "clv_score",
            F.round(F.col("frequency") * F.col("avg_order_value") / 100.0, 4)
        )

        # ── 3. Promo sensitivity ─────────────────────────────────────────────
        promo_sens = self._promo_sensitivity(txn_df, promos_df)
        log.info("  Promo sensitivity computed")

        # ── 4. Category affinity ─────────────────────────────────────────────
        cat_aff = self._category_affinity(txn_df, products_df)
        log.info("  Category affinity computed")

        # ── 5. Channel preference (most-used channel, one-hot) ───────────────
        chan_df = self._channel_preference(txn_df)
        log.info("  Channel preference computed")

        # ── 6. Join everything onto customers ────────────────────────────────
        cust = customers_df.select(
            "customer_id", "region", "age", "loyalty_tier", "income_bracket"
        )

        fm = (
            cust
            .join(rfm,       on="customer_id", how="left")
            .join(promo_sens, on="customer_id", how="left")
            .join(cat_aff,   on="customer_id", how="left")
            .join(chan_df,   on="customer_id", how="left")
            .fillna(0.0, subset=[
                "recency_days","frequency","monetary","avg_order_value",
                "max_basket_size","clv_score","promo_sensitivity_score",
            ] + [f"spend_pct_{c}" for c in CATEGORIES]
              + [f"channel_{c.replace('-','_')}" for c in CHANNELS])
        )

        # ── 7. Encode loyalty tier as ordinal ────────────────────────────────
        tier_map = F.create_map(
            F.lit("bronze"),   F.lit(1),
            F.lit("silver"),   F.lit(2),
            F.lit("gold"),     F.lit(3),
            F.lit("platinum"), F.lit(4),
        )
        fm = fm.withColumn("loyalty_score", tier_map[F.col("loyalty_tier")])

        # ── 8. Persist ───────────────────────────────────────────────────────
        out_path = os.path.join(PROCESSED, "feature_matrix_spark")
        fm.write.mode("overwrite").parquet(out_path)
        n = fm.count()
        log.info(f"  Feature matrix: {n:,} rows, {len(fm.columns)} cols -> {out_path}/")
        return fm

    # ── Private helpers ───────────────────────────────────────────────────────

    def _rfm(self, txn_df: DataFrame) -> DataFrame:
        ref = F.to_date(F.lit(REFERENCE_DATE))
        return (
            txn_df
            .withColumn("txn_date", F.to_date("date"))
            .groupBy("customer_id")
            .agg(
                F.datediff(ref, F.max("txn_date")).alias("recency_days"),
                F.count("transaction_id").alias("frequency"),
                F.sum("amount").alias("monetary"),
                F.round(F.avg("amount"), 4).alias("avg_order_value"),
                F.max("amount").alias("max_basket_size"),
            )
        )

    def _promo_sensitivity(self, txn_df: DataFrame, promos_df: DataFrame) -> DataFrame:
        # promos_df has: promo_id, product_id, discount_pct, start_date, end_date
        # Broadcast promos (small) across the large transactions table
        # Non-equijoin: date BETWEEN start_date AND end_date
        pt = (
            txn_df.alias("t")
            .join(
                F.broadcast(promos_df.alias("p")),
                on=(
                    (F.col("t.product_id") == F.col("p.product_id")) &
                    (F.to_date(F.col("t.date")) >= F.to_date(F.col("p.start_date"))) &
                    (F.to_date(F.col("t.date")) <= F.to_date(F.col("p.end_date")))
                ),
                how="left"
            )
            .withColumn("is_promo", F.when(F.col("p.promo_id").isNotNull(), 1).otherwise(0))
            .groupBy("t.customer_id")
            .agg(
                F.sum("is_promo").alias("promo_purchases"),
                F.count("t.transaction_id").alias("total_purchases"),
            )
            .withColumn(
                "promo_sensitivity_score",
                F.round(F.col("promo_purchases") / F.col("total_purchases"), 4)
            )
            .select("customer_id", "promo_sensitivity_score")
        )
        return pt

    def _category_affinity(self, txn_df: DataFrame, products_df: DataFrame) -> DataFrame:
        # Join transactions with product categories, compute spend per category
        cat_spend = (
            txn_df
            .join(
                F.broadcast(products_df.select("product_id", "category")),
                on="product_id", how="left"
            )
            .groupBy("customer_id", "category")
            .agg(F.sum("amount").alias("cat_spend"))
        )

        # Pivot: one column per category
        pivoted = (
            cat_spend
            .groupBy("customer_id")
            .pivot("category", CATEGORIES)
            .agg(F.sum("cat_spend"))
            .fillna(0.0)
        )
        # Rename to spend_pct_ (we'll turn absolute spend into fractions)
        for c in CATEGORIES:
            if c in pivoted.columns:
                pivoted = pivoted.withColumnRenamed(c, f"_raw_{c}")

        # Compute row total then fractions
        total_expr = sum(F.col(f"_raw_{c}") for c in CATEGORIES if f"_raw_{c}" in pivoted.columns)
        pivoted = pivoted.withColumn("_cat_total", total_expr)
        for c in CATEGORIES:
            col = f"_raw_{c}"
            if col in pivoted.columns:
                pivoted = pivoted.withColumn(
                    f"spend_pct_{c}",
                    F.round(F.col(col) / (F.col("_cat_total") + 1e-9), 4)
                ).drop(col)
        return pivoted.drop("_cat_total")

    def _channel_preference(self, txn_df: DataFrame) -> DataFrame:
        chan_counts = (
            txn_df
            .groupBy("customer_id", "channel")
            .count()
        )
        # One-hot encode the top channel
        for ch in CHANNELS:
            safe = ch.replace("-", "_")
            chan_counts = chan_counts.withColumn(
                f"channel_{safe}",
                F.when(F.col("channel") == F.lit(ch), F.col("count")).otherwise(F.lit(0))
            )
        result = (
            chan_counts
            .groupBy("customer_id")
            .agg(*[F.sum(f"channel_{ch.replace('-','_')}").alias(f"channel_{ch.replace('-','_')}")
                   for ch in CHANNELS])
        )
        return result


def to_pandas_sample(spark_df: DataFrame, n: int = 200_000, seed: int = 42) -> "pd.DataFrame":
    """Sample n rows from a Spark DataFrame and return as pandas."""
    total = spark_df.count()
    frac  = min(1.0, n / max(total, 1))
    log.info(f"Sampling {n:,} / {total:,} rows ({frac*100:.1f}%) for ML")
    return spark_df.sample(fraction=frac, seed=seed).limit(n).toPandas()


def top_active_customers(spark_df: DataFrame, txn_df: DataFrame,
                          n: int = 100_000) -> "pd.DataFrame":
    """Return the n most active customers (by frequency) as pandas."""
    top_ids = (
        txn_df
        .groupBy("customer_id")
        .count()
        .orderBy(F.desc("count"))
        .limit(n)
        .select("customer_id")
    )
    return spark_df.join(top_ids, on="customer_id", how="inner").toPandas()
