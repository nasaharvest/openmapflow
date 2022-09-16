from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd

from openmapflow.config import PROJECT, PROJECT_ROOT, DataPaths
from openmapflow.constants import COUNTRY, EO_DATA, MONTHS, START
from openmapflow.utils import to_date, str_to_np


def generate_model_name(val_df: pd.DataFrame, start_month: str) -> str:
    country = val_df[COUNTRY].iloc[0]
    start_year = to_date(val_df[START].iloc[0]).year
    project = PROJECT.replace("-example", "")
    return f"{country}_{project}_{start_year}_{start_month}"


def model_path_from_name(model_name: str) -> Path:
    return PROJECT_ROOT / DataPaths.MODELS / f"{model_name}.pt"


def upsample_df(
    df: pd.DataFrame, class_col: str, upsample_ratio: float
) -> pd.DataFrame:
    positive = df[class_col].astype(bool)
    negative = ~df[class_col].astype(bool)
    if len(df[positive]) > len(df[negative]):
        minority_label = "negative"
        minority = negative
        majority = positive
    else:
        minority_label = "positive"
        minority = positive
        majority = negative

    original_size = len(df[minority])
    upsampled_amount = round(len(df[majority]) * upsample_ratio) - len(df[minority])
    new_size = original_size + upsampled_amount
    if upsampled_amount < 0:
        print("Warning: upsample_minority_ratio is too high")
        return df

    upsampled_points = df[minority].sample(
        n=upsampled_amount, replace=True, random_state=42
    )
    print(
        f"Upsampling: {minority_label} class from {original_size} to {new_size} "
        + f"using upsampling ratio: {upsample_ratio}"
    )
    return df.append(upsampled_points, ignore_index=True)


def get_x_y(
    df: pd.DataFrame, label_col: str, start_month: str, input_months: int
) -> Tuple[List[np.ndarray], List[float]]:
    i = MONTHS.index(start_month)

    def to_numpy(x: str):
        return str_to_np(x)[i : i + input_months, :]  # noqa

    return df[EO_DATA].progress_apply(to_numpy).to_list(), df[label_col].to_list()
