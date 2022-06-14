import logging
import os
from pathlib import Path

import requests

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def trigger(event, context):
    """Triggered by a change to a Cloud Storage bucket.
    Calls the model inference server for each new file in the bucket.
    Args:
         event (dict): Event payload.
         context (google.cloud.functions.Context): Metadata for the event.
    """
    logger.info(f"Event: {event}")
    if "bucket" not in event and "name" not in event:
        raise ValueError("No bucket or name in event")

    src_path = f"gs://{event['bucket']}/{event['name']}"
    model_name = Path(event["name"]).parts[0]
    logger.info(f"Extracted model_name: {model_name} from {src_path}")

    available_models = os.environ.get("MODELS").split(" ")
    if model_name not in available_models:
        raise ValueError(f"{model_name} not available in {available_models}")

    url = f"{os.environ.get('INFERENCE_HOST')}/predictions/{model_name}"
    logger.info(f"Sending request to {url}")
    response = requests.post(url, data={"uri": src_path})
    logger.info(f"Received response code: {response.status_code}")
