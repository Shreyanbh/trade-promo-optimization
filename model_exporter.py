import os
import joblib
from datetime import datetime
from src.config.settings import PATHS
from src.utils.logger import get_logger

log = get_logger(__name__)


def export_model(model, name: str, metadata: dict = None) -> str:
    os.makedirs(PATHS["models"], exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(PATHS["models"], f"{name}_{ts}.pkl")
    payload = {"model": model, "metadata": metadata or {}, "name": name, "timestamp": ts}
    joblib.dump(payload, path)
    log.info(f"Model exported -> {path}")
    return path


def load_model(path: str) -> dict:
    payload = joblib.load(path)
    log.info(f"Model loaded from {path}")
    return payload
