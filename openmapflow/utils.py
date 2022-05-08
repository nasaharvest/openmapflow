from pathlib import Path
from typing import List

import pandas as pd

from .config import GCLOUD_PROJECT_ID


def try_txt_read(file_path: Path) -> List[str]:
    try:
        return pd.read_csv(file_path, sep="\n", header=None)[0].tolist()
    except FileNotFoundError:
        return []


def colab_gee_gcloud_login():
    import google
    import ee

    print("Logging into Google Cloud")
    google.colab.auth.authenticate_user()
    print("Logging into Earth Engine")
    SCOPES = [
        "https://www.googleapis.com/auth/cloud-platform",
        "https://www.googleapis.com/auth/earthengine",
    ]
    CREDENTIALS, _ = google.auth.default(default_scopes=SCOPES)
    ee.Initialize(CREDENTIALS, project=GCLOUD_PROJECT_ID)
