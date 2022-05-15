import pickle

import numpy as np
import pandas as pd

try:
    import torch
    from torch import Tensor
    from torch.utils.data import Dataset
except ImportError:
    print("PyTorch must be installed to use the PyTorch dataset.")


from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, cast

from cropharvest.countries import BBox
from dateutil.relativedelta import relativedelta
from tqdm import tqdm

from openmapflow.constants import CLASS_PROB, END, FEATURE_PATH, LAT, LON, MONTHS, START

IS_POSITIVE_CLASS = "is_positive_class"
IS_LOCAL = "is_local"


def _is_local(df: pd.DataFrame, target_bbox: BBox) -> Union[bool, pd.Series]:
    if target_bbox:
        return (
            (df[LAT] >= target_bbox.min_lat)
            & (df[LAT] <= target_bbox.max_lat)
            & (df[LON] >= target_bbox.min_lon)
            & (df[LON] <= target_bbox.max_lon)
        )
    return True


def _compute_num_timesteps(
    start_month_index: int, input_months: int, start_col: pd.Series, end_col: pd.Series
) -> List[int]:
    df_start_date = pd.to_datetime(start_col).apply(
        lambda dt: dt.replace(month=start_month_index + 1)
    )
    df_candidate_end_date = df_start_date.apply(
        lambda dt: dt + relativedelta(months=+input_months)
    )
    df_data_end_date = pd.to_datetime(end_col)
    df_end_date = pd.DataFrame({"1": df_data_end_date, "2": df_candidate_end_date}).min(
        axis=1
    )
    # Pick min available end date
    timesteps = (
        ((df_end_date - df_start_date) / np.timedelta64(1, "M"))
        .round()
        .unique()
        .astype(int)
    )
    return [int(t) for t in timesteps]


def _df_stats(df: pd.DataFrame, subset: str) -> Dict[str, Union[float, int]]:
    dataset_info: Dict[str, Union[float, int]] = {}
    if df[IS_LOCAL].any():
        dataset_info[f"local_{subset}_original_size"] = len(df[df[IS_LOCAL]])
        dataset_info[f"local_{subset}_positive_class_percentage"] = round(
            len(df[df[IS_LOCAL] & df["is_positive_class"]]) / len(df[df[IS_LOCAL]]),
            4,
        )

    if not df[IS_LOCAL].all():
        dataset_info[f"global_{subset}_original_size"] = len(df[~df[IS_LOCAL]])
        dataset_info[f"global_{subset}_positive_class_percentage"] = round(
            len(df[~df[IS_LOCAL] & df["is_positive_class"]]) / len(df[~df[IS_LOCAL]]),
            4,
        )
    return dataset_info


def _upsample_df(df: pd.DataFrame, upsample_minority_ratio: float) -> pd.DataFrame:
    positive = df[IS_LOCAL] & df[IS_POSITIVE_CLASS]
    negative = df[IS_LOCAL] & ~df[IS_POSITIVE_CLASS]
    if len(df[positive]) > len(df[negative]):
        minority_label = "negative"
        minority = negative
        majority = positive
    else:
        minority_label = "positive"
        minority = positive
        majority = negative

    original_size = len(df[minority])
    additional_size = len(df[majority]) * (upsample_minority_ratio) - len(df[minority])
    upsampled_size = original_size + additional_size
    if additional_size < 0:
        print("Warning: upsample_minority_ratio is too high")
        return df

    upsampled_points = df[minority].sample(
        n=additional_size, replace=True, random_state=42
    )
    print(
        f"Upsampling: {minority_label} from {original_size} to {upsampled_size} "
        + f"using ratio upsampling ratio: {upsample_minority_ratio}"
    )
    return df.append(upsampled_points, ignore_index=True)


class PyTorchDataset(Dataset):
    """
    Used for training and evaluating PyTorch based models.

    Args:
        df (pd.DataFrame): LabeledDataset style data frame consisting of
            - A coordinate
            - A binary label for that coordinate (y)
            - A path to earth observation data for that coordinate (X)
        subset (str): One of "training", "validation", or "testing"
            default: "training"
        start_month (str): The month the earth observation data should start
            default: "January"
        input_months (int): The number of months of earth observation data to use
            default: 12
        up_to_year (int): If specified, only use data up to this year
            default: None
        cache (bool): Whether to cache the dataset for faster loading
            default: True
        upsample_minority_ratio (Optional[float]): Upsample the minority class
            such that the minority class / majority class ratio == upsample_minority_ratio
            default: None
        target_bbox (BBox): A bounding box to indicate which examples are local (within bbox)
            default: None
        prob_threshold (float): The probability threshold for what is a positive and negative class
            default: 0.5
    """

    def __init__(
        self,
        df: pd.DataFrame,
        subset: str = "training",
        start_month: str = "January",
        input_months: int = 12,
        up_to_year: Optional[int] = None,
        cache: bool = True,
        upsample_minority_ratio: Optional[float] = None,
        target_bbox: BBox = None,
        probability_threshold: float = 0.5,
    ) -> None:

        assert subset in ["training", "validation", "testing"]
        self.start_month_index = MONTHS.index(start_month)
        self.input_months = input_months

        df = df.copy()  # To avoid indexing errors
        if up_to_year is not None:
            df = df[pd.to_datetime(df[START]).dt.year <= up_to_year]
        df[IS_POSITIVE_CLASS] = df[CLASS_PROB] >= probability_threshold
        df[IS_LOCAL] = _is_local(df, target_bbox)
        self.num_timesteps = _compute_num_timesteps(
            start_month_index=self.start_month_index,
            input_months=self.input_months,
            start_col=df[START],
            end_col=df[END],
        )
        self.dataset_info = _df_stats(df, subset)
        if upsample_minority_ratio:
            df = _upsample_df(df, upsample_minority_ratio)
        self.df = df

        # Set parameters needed for __getitem__
        self.probability_threshold = probability_threshold
        self.target_bbox = target_bbox

        # Cache dataset if necessary
        self.x: Optional[Tensor] = None
        self.y: Optional[Tensor] = None
        self.weights: Optional[Tensor] = None
        self.cache = False
        if cache:
            self.x, self.y, self.weights = self.to_array()
            self.cache = cache

    def __len__(self) -> int:
        return len(self.df)

    def to_array(self) -> Tuple[Tensor, Tensor, Tensor]:
        if self.x is not None:
            assert self.y is not None
            assert self.weights is not None
            return self.x, self.y, self.weights
        else:
            x_list: List[Tensor] = []
            y_list: List[Tensor] = []
            weight_list: List[Tensor] = []
            print("Loading data into memory")
            for i in tqdm(range(len(self)), desc="Caching files"):
                x, y, weight = self[i]
                x_list.append(x)
                y_list.append(y)
                weight_list.append(weight)

            return torch.stack(x_list), torch.stack(y_list), torch.stack(weight_list)

    def __getitem__(self, index: int) -> Tuple[Tensor, Tensor, Tensor]:

        if (self.cache) & (self.x is not None):
            # if we upsample, the caching might not have happened yet
            return (
                cast(Tensor, self.x)[index],
                cast(Tensor, self.y)[index],
                cast(Tensor, self.weights)[index],
            )

        row = self.df.iloc[index]

        # first, we load up the target file
        with Path(row[FEATURE_PATH]).open("rb") as f:
            target_datainstance = pickle.load(f)

        x = target_datainstance.labelled_array
        x = x[
            self.start_month_index : self.start_month_index  # noqa: E203
            + self.input_months
        ]

        # If x is a partial time series, pad it to full length
        if x.shape[0] < self.input_months:
            x = np.concatenate(
                [x, np.full((self.input_months - x.shape[0], x.shape[1]), np.nan)]
            )

        return (
            torch.from_numpy(x).float(),
            torch.tensor(int(row[IS_POSITIVE_CLASS])).float(),
            torch.tensor(int(row[IS_LOCAL])).float(),
        )
