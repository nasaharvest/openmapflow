from pathlib import Path

import pandas as pd

from openmapflow.config import PROJECT, PROJECT_ROOT, DataPaths
from openmapflow.constants import COUNTRY, START
from openmapflow.utils import to_date


def generate_model_name(val_df: pd.DataFrame, start_month: str) -> str:
    country = val_df[COUNTRY].iloc[0]
    start_year = to_date(val_df[START].iloc[0]).year
    project = PROJECT.replace("-example", "")
    return f"{country}_{project}_{start_year}_{start_month}"


def model_path_from_name(model_name: str) -> Path:
    return PROJECT_ROOT / DataPaths.MODELS / f"{model_name}.pt"
