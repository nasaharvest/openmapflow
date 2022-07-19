import os

import ee
import numpy as np
import pandas as pd


def colab_gee_gcloud_login(project_id: str, google):
    os.environ["GOOGLE_CLOUD_PROJECT"] = project_id
    print("Logging into Google Cloud")
    google.colab.auth.authenticate_user()
    print("Logging into Earth Engine")
    SCOPES = [
        "https://www.googleapis.com/auth/cloud-platform",
        "https://www.googleapis.com/auth/earthengine",
    ]
    CREDENTIALS, _ = google.auth.default(default_scopes=SCOPES)
    ee.Initialize(CREDENTIALS, project=project_id)


def confirmation(message, default="y"):
    print(message)
    if default == "y":
        return input("Confirm [y]/n: ").lower() != "n"
    else:
        return input("Confirm y/[n]: ").lower() == "y"


def to_date(d):
    if type(d) == np.datetime64:
        return d.astype("M8[D]").astype("O")
    elif type(d) == str:
        return pd.to_datetime(d).date()
    else:
        return d.date()
