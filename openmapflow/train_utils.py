from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

from openmapflow.config import PROJECT, PROJECT_ROOT, DataPaths
from openmapflow.constants import CLASS_PROB, COUNTRY, EO_DATA, MONTHS, START
from openmapflow.utils import str_to_np, to_date, tqdm


def generate_model_name(val_df: pd.DataFrame, start_month: Optional[str] = None) -> str:
    """Generate a model name based on the validation data."""
    model_name = ""
    try:
        model_name += val_df[COUNTRY].iloc[0] + "_"
    except KeyError:
        pass

    model_name += PROJECT.replace("-example", "")
    model_name += f"_{to_date(val_df[START].iloc[0]).year}"
    if start_month:
        model_name += f"_{start_month}"
    return model_name


def model_path_from_name(model_name: str) -> Path:
    """Get the path to a model from its name."""
    return PROJECT_ROOT / DataPaths.MODELS / f"{model_name}.pt"


def upsample_df(
    df: pd.DataFrame, label_col: str = CLASS_PROB, upsample_ratio: float = 1.0
) -> pd.DataFrame:
    """Upsample a dataframe to have more balanced classes."""
    label_classes = df[label_col].unique()
    if len(label_classes) != 2:
        raise ValueError(
            f"Can only upsample binary classes. Found {len(label_classes)} classe in {label_col}"
        )

    positive = df[label_col].astype(bool)
    negative = ~df[label_col].astype(bool)
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
    df: pd.DataFrame,
    label_col: str = CLASS_PROB,
    start_month: str = "February",
    input_months: int = 12,
) -> Tuple[List[np.ndarray], List[float]]:
    """Get the X and y data from a dataframe."""
    i = MONTHS.index(start_month)

    def to_numpy(x: str):
        return str_to_np(x)[i : i + input_months, :]  # noqa

    tqdm.pandas()
    return df[EO_DATA].progress_apply(to_numpy).to_list(), df[label_col].to_list()
