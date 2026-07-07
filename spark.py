import os
import sys
import logging

from pyspark.sql import SparkSession

from src.utils.logger import get_logger

log = get_logger("cloud.spark")

_SPARK: dict = {}


def get_spark(config: dict) -> SparkSession:
    provider = config.get("provider", "local")

    if provider in _SPARK:
        return _SPARK[provider]

    log.info(f"Initialising SparkSession — provider: {provider}")

    if provider == "local":
        spark = _build_local(config)

    elif provider == "databricks":
        spark = _build_databricks(config)

    elif provider == "emr":
        spark = _build_emr(config)

    elif provider == "dataproc":
        spark = _build_dataproc(config)

    else:
        raise ValueError(
            f"Unknown Spark provider '{provider}'. "
            "Valid values: 'local' | 'databricks' | 'emr' | 'dataproc'"
        )

    _SPARK[provider] = spark
    return spark


def stop_spark(provider: str = "local") -> None:
    session = _SPARK.pop(provider, None)
    if session:
        log.info(f"Stopping SparkSession — provider: {provider}")
        session.stop()


# ---------------------------------------------------------------------------
# Provider builders
# ---------------------------------------------------------------------------

def _build_local(config: dict) -> SparkSession:
    driver_mem = config.get("driver_memory", "4g")
    executor_mem = config.get("executor_memory", "4g")

    os.environ["PYSPARK_PYTHON"] = sys.executable
    os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable

    spark = (
        SparkSession.builder
        .appName(config.get("app_name", "TradePromoOptimization"))
        .master("local[*]")
        .config("spark.driver.memory", driver_mem)
        .config("spark.executor.memory", executor_mem)
        .config("spark.sql.shuffle.partitions", "50")
        .config("spark.sql.execution.arrow.pyspark.enabled", "false")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")
    return spark


def _build_databricks(config: dict) -> SparkSession:
    try:
        from databricks.connect import DatabricksSession

        host = config.get("host", "")
        token = config.get("token", "")
        cluster_id = config.get("cluster_id", "")

        log.info(f"Connecting to Databricks cluster: {cluster_id} @ {host}")
        spark = (
            DatabricksSession.builder
            .remote(host=host, token=token, cluster_id=cluster_id)
            .getOrCreate()
        )
        return spark

    except ImportError:
        connect_url = config.get("connect_url", "")
        if connect_url:
            log.warning(
                "databricks-connect not installed — falling back to Spark Connect URL"
            )
            spark = (
                SparkSession.builder
                .remote(connect_url)
                .getOrCreate()
            )
            return spark

        raise ImportError(
            "databricks-connect is not installed and no 'connect_url' was provided. "
            "Install it with: pip install databricks-connect"
        )


def _build_emr(config: dict) -> SparkSession:
    master_url = config.get("master_url", "yarn")
    app_name = config.get("app_name", "TradePromoOptimization-EMR")

    log.info(f"Connecting to EMR master: {master_url}")

    spark = (
        SparkSession.builder
        .appName(app_name)
        .master(master_url)
        .config("spark.sql.shuffle.partitions", "50")
        .getOrCreate()
    )
    return spark


def _build_dataproc(config: dict) -> SparkSession:
    master_url = config.get("master_url", "")
    app_name = config.get("app_name", "TradePromoOptimization-Dataproc")
    project_id = config.get("project_id", "")
    region = config.get("region", "")
    cluster_name = config.get("cluster_name", "")

    if not master_url and cluster_name:
        master_url = f"spark://{cluster_name}-m:7077"
        log.info(
            f"Derived Dataproc master URL from cluster_name: {master_url} "
            f"(project={project_id}, region={region})"
        )
    else:
        log.info(f"Connecting to Dataproc master: {master_url}")

    spark = (
        SparkSession.builder
        .appName(app_name)
        .master(master_url)
        .config("spark.sql.shuffle.partitions", "50")
        .getOrCreate()
    )
    return spark
