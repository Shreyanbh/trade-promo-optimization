"""
SparkDataGenerator — generates 5M-customer source datasets using PySpark.

Structured (Parquet, partitioned for efficient reads):
  data/raw/structured/customers/   — 5M rows, partitioned by region
  data/raw/structured/transactions/ — 25M rows, partitioned by year_month

Small sources (kept as original formats):
  data/raw/structured/products.xlsx
  data/raw/structured/promotions.json

Unstructured (TXT — same parser regardless of scale):
  data/raw/unstructured/customer_reviews.txt
  data/raw/unstructured/call_transcripts.txt
  data/raw/unstructured/support_emails.txt
"""
import os
import json
from datetime import datetime, timedelta

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import IntegerType, LongType

from src.config.settings import PATHS
from src.utils.logger import get_logger

log = get_logger("spark_data_generator")

RAW_STRUCT   = os.path.join(PATHS["raw_data"], "structured")
RAW_UNSTRUCT = os.path.join(PATHS["raw_data"], "unstructured")

REGIONS  = ["north", "south", "east", "west"]
LOYALTY  = ["bronze", "silver", "gold", "platinum"]
INCOME   = ["<30k", "30-50k", "50-80k", "80-120k", ">120k"]
CHANNELS = ["in-store", "online", "mobile-app", "click-and-collect"]
PAYMENT  = ["card", "cash", "digital-wallet", "loyalty-points"]
CATEGORIES = ["beverages", "snacks", "dairy", "produce", "household"]
BRANDS     = ["FreshCo", "NaturePlus", "HomeEssentials", "DailyBest", "PureChoice"]

_SPARK: SparkSession = None


def get_spark() -> SparkSession:
    global _SPARK
    if _SPARK is None or _SPARK._sc._jvm is None:
        import sys
        # Windows: PySpark workers look for 'python3'; point them to the current interpreter
        os.environ.setdefault("PYSPARK_PYTHON", sys.executable)
        os.environ.setdefault("PYSPARK_DRIVER_PYTHON", sys.executable)
        _SPARK = (
            SparkSession.builder
            .appName("TradePromoOptimization")
            .config("spark.driver.memory", "4g")
            .config("spark.executor.memory", "4g")
            .config("spark.sql.shuffle.partitions", "50")
            .config("spark.default.parallelism", "8")
            .config("spark.sql.execution.arrow.pyspark.enabled", "false")
            .config("spark.pyspark.python",        sys.executable)
            .config("spark.pyspark.driver.python", sys.executable)
            .getOrCreate()
        )
        _SPARK.sparkContext.setLogLevel("WARN")
    return _SPARK


def generate_all_spark(n_customers: int = 5_000_000,
                       n_transactions: int = 25_000_000,
                       n_products: int = 50,
                       n_promos: int = 20,
                       force: bool = False) -> dict:
    """Generate all raw source files. Returns dict of paths."""
    os.makedirs(RAW_STRUCT, exist_ok=True)
    os.makedirs(RAW_UNSTRUCT, exist_ok=True)

    spark = get_spark()
    paths = {}

    paths["customers_parquet"]  = _customers(spark, n_customers, force)
    paths["transactions_parquet"] = _transactions(spark, n_transactions, n_customers, n_products, force)
    paths["products_xlsx"]      = _products_xlsx(n_products, force)
    paths["promotions_json"]    = _promotions_json(n_products, n_promos, force)
    paths["reviews_txt"]        = _reviews_txt(min(n_customers, 10_000), force)
    paths["transcripts_txt"]    = _transcripts_txt(min(n_customers // 5000, 2_000), force)
    paths["emails_txt"]         = _emails_txt(min(n_customers // 2000, 3_000), force)

    log.info(f"Generated {len(paths)} raw source paths "
             f"({n_customers:,} customers, {n_transactions:,} transactions)")
    return paths


# ── Large structured sources (PySpark) ───────────────────────────────────────

def _customers(spark: SparkSession, n: int, force: bool) -> str:
    path = os.path.join(RAW_STRUCT, "customers")
    if os.path.exists(path) and not force:
        cnt = spark.read.parquet(path).count()
        log.info(f"  customers Parquet exists: {cnt:,} rows")
        return path

    log.info(f"  Generating {n:,} customers with PySpark...")
    regions = F.array(*[F.lit(r) for r in REGIONS])
    loyalty = F.array(*[F.lit(l) for l in LOYALTY])
    incomes = F.array(*[F.lit(i) for i in INCOME])

    df = (
        spark.range(n)
        .withColumn("customer_id",    F.format_string("C%07d", F.col("id")))
        .withColumn("region",         F.element_at(regions, (F.rand(1) * 4).cast(IntegerType()) + 1))
        .withColumn("age",            (F.rand(2) * 57 + 18).cast(IntegerType()))
        .withColumn("loyalty_tier",   F.element_at(loyalty, (F.rand(3) * 4).cast(IntegerType()) + 1))
        .withColumn("income_bracket", F.element_at(incomes, (F.rand(4) * 5).cast(IntegerType()) + 1))
        .withColumn("signup_date",    F.date_add(F.lit("2018-01-01"),
                                                  (F.rand(5) * 2190).cast(IntegerType())))
        .withColumn("email",          F.concat(F.lit("customer_"),
                                                F.lower(F.col("customer_id")),
                                                F.lit("@email.com")))
        .drop("id")
    )

    (df.repartition(8, "region")
       .write.partitionBy("region")
       .mode("overwrite")
       .parquet(path))
    actual = spark.read.parquet(path).count()
    log.info(f"  Wrote {actual:,} customers -> {path}/ (Parquet, partitioned by region)")
    return path


def _transactions(spark: SparkSession, n: int, n_customers: int,
                   n_products: int, force: bool) -> str:
    path = os.path.join(RAW_STRUCT, "transactions")
    if os.path.exists(path) and not force:
        cnt = spark.read.parquet(path).count()
        log.info(f"  transactions Parquet exists: {cnt:,} rows")
        return path

    log.info(f"  Generating {n:,} transactions with PySpark...")
    channels = F.array(*[F.lit(c) for c in CHANNELS])
    payments = F.array(*[F.lit(p) for p in PAYMENT])

    df = (
        spark.range(n)
        .withColumn("transaction_id",
                    F.format_string("TXN%09d", F.col("id")))
        .withColumn("customer_id",
                    F.format_string("C%07d", (F.rand(10) * n_customers).cast(LongType())))
        .withColumn("product_id",
                    F.format_string("P%03d", (F.rand(11) * n_products).cast(IntegerType())))
        .withColumn("quantity",      (F.rand(12) * 5 + 1).cast(IntegerType()))
        .withColumn("unit_price",    F.round(F.rand(13) * 49 + 1, 2))
        .withColumn("amount",        F.round(F.col("quantity") * F.col("unit_price"), 2))
        .withColumn("date",          F.date_add(F.lit("2023-01-01"),
                                                  (F.rand(14) * 365).cast(IntegerType())))
        .withColumn("channel",       F.element_at(channels, (F.rand(15) * 4).cast(IntegerType()) + 1))
        .withColumn("payment_method",F.element_at(payments, (F.rand(16) * 4).cast(IntegerType()) + 1))
        .withColumn("promo_code_applied",
                    F.when(F.rand(17) < 0.3,
                           F.format_string("PROMO%03d", (F.rand(18) * 20 + 1).cast(IntegerType())))
                     .otherwise(F.lit("")))
        .withColumn("year_month",    F.date_format("date", "yyyy-MM"))
        .drop("id")
    )

    (df.repartition(50)
       .write.partitionBy("year_month")
       .mode("overwrite")
       .parquet(path))
    actual = spark.read.parquet(path).count()
    log.info(f"  Wrote {actual:,} transactions -> {path}/ (Parquet, partitioned by year_month)")
    return path


# ── Small structured sources (pandas, keep same format) ───────────────────────

def _products_xlsx(n_products: int, force: bool) -> str:
    import numpy as np
    import pandas as pd
    import string
    path = os.path.join(RAW_STRUCT, "products.xlsx")
    if os.path.exists(path) and not force:
        return path
    rng = np.random.default_rng(42)
    cats = [rng.choice(CATEGORIES) for _ in range(n_products)]
    pids = [f"P{i:03d}" for i in range(n_products)]
    SUBCATS = {"beverages": ["hot drinks","soft drinks","juices"],
               "snacks": ["crisps","biscuits","confectionery"],
               "dairy": ["milk","cheese","yogurt"],
               "produce": ["fruits","vegetables","salads"],
               "household": ["cleaning","personal care","kitchen"]}
    products = pd.DataFrame([{
        "product_id":   pid, "product_name": f"Product_{pid[1:]}",
        "category":     cat, "subcategory":  rng.choice(SUBCATS[cat]),
        "brand":        rng.choice(BRANDS),
        "price":        round(float(rng.uniform(1, 50)), 2),
        "cost_price":   round(float(rng.uniform(0.5, 25)), 2),
        "sku":          "SKU-" + "".join(rng.choice(list(string.ascii_uppercase), 4)),
    } for pid, cat in zip(pids, cats)])
    inventory = pd.DataFrame([{
        "product_id":        pid,
        "stock_level":       int(rng.integers(0, 500)),
        "reorder_threshold": int(rng.integers(20, 100)),
        "warehouse_location": f"WH{int(rng.integers(1,4))}-AISLE{int(rng.integers(1,10))}",
    } for pid in pids])
    with pd.ExcelWriter(path, engine="openpyxl") as w:
        products.to_excel(w, sheet_name="Products", index=False)
        inventory.to_excel(w, sheet_name="Inventory", index=False)
    log.info(f"  Wrote {path} (Excel, 2 sheets)")
    return path


def _promotions_json(n_products: int, n: int, force: bool) -> str:
    import numpy as np
    path = os.path.join(RAW_STRUCT, "promotions.json")
    if os.path.exists(path) and not force:
        return path
    rng  = np.random.default_rng(42)
    pids = [f"P{i:03d}" for i in range(n_products)]
    base = datetime(2023, 1, 1)
    promos = []
    for i in range(n):
        start = base + timedelta(days=int(rng.integers(0, 300)))
        cats  = list(rng.choice(CATEGORIES, size=int(rng.integers(1, 4)), replace=False))
        prods = [str(p) for p in rng.choice(pids, size=int(rng.integers(3, 10)), replace=False)]
        promos.append({
            "promo_id":     f"PROMO{i+1:03d}",
            "name":         f"Promotion {i+1}",
            "type":         rng.choice(["percentage", "fixed", "bogo"]),
            "discount_value": round(float(rng.uniform(5, 40)), 1),
            "start_date":   start.strftime("%Y-%m-%d"),
            "end_date":     (start + timedelta(days=int(rng.integers(7, 45)))).strftime("%Y-%m-%d"),
            "eligible_categories": cats,
            "eligible_products":   prods,
        })
    with open(path, "w") as f:
        json.dump(promos, f, indent=2)
    log.info(f"  Wrote {path} ({n} promos, JSON)")
    return path


# ── Unstructured sources (scaled TXT) ─────────────────────────────────────────

_POS = ["Absolutely love this product! Fast delivery and great quality.",
        "Exceeded my expectations. Will definitely reorder soon.",
        "Fantastic value for money. Outstanding quality.",
        "Perfect product. Arrived on time and exactly as described.",
        "Really impressed. This has become my go-to brand.",
        "Five stars all around. Superb quality and packaging."]
_NEG = ["Very disappointed. Product didn't match the description.",
        "Poor quality for the price. Expected much better.",
        "Arrived damaged and the packaging was completely crushed.",
        "Not what I ordered. Customer service was unhelpful.",
        "Terrible experience. Will not be purchasing again."]
_NEU = ["Decent product. Does what it says on the tin.",
        "Average quality. Nothing special but gets the job done.",
        "It's okay. Not as good as the previous version.",
        "Product is fine. Delivery took a bit longer than expected."]


def _reviews_txt(n: int, force: bool) -> str:
    import numpy as np
    path = os.path.join(RAW_UNSTRUCT, "customer_reviews.txt")
    if os.path.exists(path) and not force:
        return path
    rng  = np.random.default_rng(42)
    pids = [f"P{i:03d}" for i in range(50)]
    base = datetime(2023, 1, 1)
    lines = []
    for i in range(n):
        cid    = f"C{int(rng.integers(0, 5_000_000)):07d}"
        pid    = rng.choice(pids)
        rating = int(rng.integers(1, 6))
        d      = (base + timedelta(days=int(rng.integers(0, 365)))).strftime("%Y-%m-%d")
        text   = rng.choice(_POS if rating >= 4 else (_NEG if rating <= 2 else _NEU))
        lines += [f"REVIEW [R{i+1:06d}] | CUSTOMER: {cid} | PRODUCT: {pid} | DATE: {d} | RATING: {rating}/5",
                  f'"{text}"', "---", ""]
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    log.info(f"  Wrote {path} ({n:,} reviews, TXT)")
    return path


def _transcripts_txt(n: int, force: bool) -> str:
    import numpy as np
    n = max(n, 60)
    path = os.path.join(RAW_UNSTRUCT, "call_transcripts.txt")
    if os.path.exists(path) and not force:
        return path
    rng   = np.random.default_rng(42)
    pids  = [f"P{i:03d}" for i in range(50)]
    agents = ["Sarah", "Michael", "Emily", "James", "Priya", "Tom"]
    issues = [("damaged product", "Replacement order created."),
              ("missing item", "Refund issued."),
              ("billing query", "Billing corrected, credit applied."),
              ("delivery delay", "Customer updated with new ETA."),
              ("product complaint", "Return label sent, refund processed.")]
    sents  = ["NEGATIVE", "NEUTRAL", "POSITIVE"]
    base   = datetime(2023, 1, 1)
    lines  = []
    for i in range(n):
        cid     = f"C{int(rng.integers(0, 5_000_000)):07d}"
        pid     = rng.choice(pids)
        issue, res = rng.choice(issues)
        sent    = rng.choice(sents, p=[0.45, 0.35, 0.20])
        d       = (base + timedelta(days=int(rng.integers(0, 365)))).strftime("%Y-%m-%d")
        lines  += [f"CALL [T{i+1:06d}] | DATE: {d} | AGENT: {rng.choice(agents)} | CUSTOMER: {cid}",
                   f"PRODUCT REFERENCED: {pid}",
                   f"Issue: {issue.title()}",
                   f"RESOLUTION: {res}",
                   f"SENTIMENT: {sent}", "---", ""]
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    log.info(f"  Wrote {path} ({n:,} transcripts, TXT)")
    return path


def _emails_txt(n: int, force: bool) -> str:
    import numpy as np
    n = max(n, 100)
    path = os.path.join(RAW_UNSTRUCT, "support_emails.txt")
    if os.path.exists(path) and not force:
        return path
    rng  = np.random.default_rng(42)
    pids = [f"P{i:03d}" for i in range(50)]
    subjects = ["Order Inquiry","Missing Delivery","Return Request",
                "Billing Dispute","Product Feedback","Account Problem"]
    base  = datetime(2023, 1, 1)
    lines = []
    for i in range(n):
        cid   = f"C{int(rng.integers(0, 5_000_000)):07d}"
        pid   = rng.choice(pids)
        subj  = rng.choice(subjects)
        d     = (base + timedelta(days=int(rng.integers(0, 365)))).strftime("%Y-%m-%d")
        senti = rng.choice(["POSITIVE","NEUTRAL","NEGATIVE"], p=[0.2, 0.45, 0.35])
        lines += [f"EMAIL [E{i+1:06d}] | DATE: {d} | CUSTOMER: {cid} | SUBJECT: {subj}",
                  f"PRODUCT: {pid}", "---",
                  f"SENTIMENT_LABEL: {senti}",
                  f"RESPONSE: Issue acknowledged.", "===", ""]
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    log.info(f"  Wrote {path} ({n:,} emails, TXT)")
    return path
