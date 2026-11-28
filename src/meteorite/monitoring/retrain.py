"""HTTP call used by the daily retraining DAG."""
import json
import logging
from urllib.request import Request, urlopen

logger = logging.getLogger(__name__)


def call_retrain(api_url: str) -> None:
    """Request a single training run and log its registry outcome."""
    request = Request(f"{api_url.rstrip('/')}/train", method="POST")
    with urlopen(request, timeout=300) as response:
        result = json.load(response)
    logger.info(
        "Training completed: run_id=%s model_version=%s promoted=%s",
        result["run_id"], result["model_version"], result["promoted"],
    )
