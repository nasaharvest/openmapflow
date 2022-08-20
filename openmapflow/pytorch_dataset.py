from typing import Dict, List, Optional, Tuple, Union, cast

import numpy as np
import pandas as pd
import torch
from dateutil.relativedelta import relativedelta
from torch import Tensor
from torch.utils.data import Dataset

from openmapflow.bbox import BBox
from openmapflow.constants import CLASS_PROB, END, EO_DATA, LAT, LON, MONTHS, START
from openmapflow.utils import tqdm

IS_POSITIVE_CLASS = "is_positive_class"
IS_LOCAL = "is_local"


def _is_local(df: pd.DataFrame, bbox: Optional[BBox]) -> Union[bool, pd.Series]:
    if bbox is None:
        return True
    return (
        (df[LAT] >= bbox.min_lat)
        & (df[LAT] <= bbox.max_lat)
        & (df[LON] >= bbox.min_lon)
        & (df[LON] <= bbox.max_lon)
    )


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


def _df_stats(
    df: pd.DataFrame, subset: str, upsample_ratio
) -> Dict[str, Union[float, int]]:
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


def _upsample_df(df: pd.DataFrame, upsample_ratio: float) -> pd.DataFrame:
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

        for col in [CLASS_PROB, END, LAT, LON, START, EO_DATA]:
            if col not in df.columns:
                raise ValueError(f"{col} is not a column in the dataframe")

        if subset not in ["training", "validation", "testing"]:
            raise ValueError(
                f"{subset} must be in ['training', 'validation', 'testing']"
            )

        if start_month not in MONTHS:
            raise ValueError(
                f"{start_month} is not a valid start month. Should be one of {MONTHS}"
            )

        if input_months < 1:
            raise ValueError(
                f"{input_months} is not a valid input months. Should be > 0"
            )

        if upsample_minority_ratio is not None and upsample_minority_ratio <= 0.0:
            raise ValueError(
                f"{upsample_minority_ratio} is not a valid upsample_minority_ratio. Should be > 0"
            )

        if probability_threshold < 0 or probability_threshold > 1:
            raise ValueError(
                f"{probability_threshold} is not a valid probability threshold."
                + "Should be between 0 and 1"
            )

        df = df.copy()

        self.input_months = input_months
        self.start_month_index = MONTHS.index(start_month)
        self.end_month_index = self.start_month_index + input_months

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
        self.dataset_info = _df_stats(df, subset, upsample_minority_ratio)
        if upsample_minority_ratio:
            df = _upsample_df(df, upsample_minority_ratio)
        self.df = df

        # Set parameters needed for __getitem__
        self.probability_threshold = probability_threshold
        self.target_bbox = target_bbox

        # Cache dataset if necessary
        self.x: Optional[Tensor] = None
        self.y: Optional[Tensor] = None
        self.is_local: Optional[Tensor] = None
        self.cache = False
        if cache:
            self.x, self.y, self.is_local = self.to_array()
            self.cache = True

    def __len__(self) -> int:
        return len(self.df)

    def to_array(self) -> Tuple[Tensor, Tensor, Tensor]:
        if self.x is not None and self.y is not None and self.is_local is not None:
            return self.x, self.y, self.is_local
        x_list: List[Tensor] = []
        y_list: List[Tensor] = []
        is_local_list: List[Tensor] = []
        print("Loading data into memory")
        for i in tqdm(range(len(self)), desc="Caching files"):
            x, y, is_local = self[i]
            x_list.append(x)
            y_list.append(y)
            is_local_list.append(is_local)
        return torch.stack(x_list), torch.stack(y_list), torch.stack(is_local_list)

    def _pad_if_necessary(self, x: np.ndarray) -> np.ndarray:
        padding_necessary = x.shape[0] < self.input_months
        if padding_necessary:
            return np.concatenate(
                [x, np.full((self.input_months - x.shape[0], x.shape[1]), np.nan)]
            )
        return x

    def __getitem__(self, index: int) -> Tuple[Tensor, Tensor, Tensor]:
        if self.cache:
            return (
                cast(Tensor, self.x)[index],
                cast(Tensor, self.y)[index],
                cast(Tensor, self.is_local)[index],
            )

        label_row = self.df.iloc[index]
        x = label_row[EO_DATA]
        x = x[self.start_month_index : self.end_month_index]  # noqa E203
        x = self._pad_if_necessary(x)
        return (
            torch.from_numpy(x).float(),
            torch.tensor(int(label_row[IS_POSITIVE_CLASS])).float(),
            torch.tensor(int(label_row[IS_LOCAL])).float(),
        )
