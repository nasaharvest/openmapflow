import numpy as np
import pandas as pd
import pickle

try:
    import torch
    from torch.utils.data import Dataset
    from torch import Tensor
except ImportError:
    print("PyTorch must be installed to use the PyTorch dataset.")


from cropharvest.countries import BBox
from dateutil.relativedelta import relativedelta
from pathlib import Path
from tqdm import tqdm
from typing import cast, Tuple, Optional, List, Dict, Union

from .constants import CLASS_PROB, FEATURE_PATH, LAT, LON, START, END, MONTHS


class PyTorchDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        subset: str = "training",
        cache: bool = True,
        upsample: bool = False,
        target_bbox: BBox = None,
        start_month: str = "January",
        probability_threshold: float = 0.5,
        input_months: int = 12,
        up_to_year: Optional[int] = None,
    ) -> None:

        assert subset in ["training", "validation", "testing"]

        df = df.copy()  # To avoid indexing errors

        if subset == "training" and up_to_year is not None:
            df = df[pd.to_datetime(df[START]).dt.year <= up_to_year]

        self.start_month_index = MONTHS.index(start_month)
        self.input_months = input_months

        df["is_positive_class"] = df[CLASS_PROB] >= probability_threshold
        if target_bbox:
            df["is_local"] = (
                (df[LAT] >= target_bbox.min_lat)
                & (df[LAT] <= target_bbox.max_lat)
                & (df[LON] >= target_bbox.min_lon)
                & (df[LON] <= target_bbox.max_lon)
            )
        else:
            df["is_local"] = True

        local_positive_class = len(df[df["is_local"] & df["is_positive_class"]])
        local_negative_class = len(df[df["is_local"] & ~df["is_positive_class"]])
        local_difference = np.abs(local_positive_class - local_negative_class)

        self.num_timesteps = self._compute_num_timesteps(
            start_col=df[START], end_col=df[END]
        )

        dataset_info: Dict[str, Union[float, int]] = {}
        if df["is_local"].any():
            dataset_info[f"local_{subset}_original_size"] = len(df[df["is_local"]])
            dataset_info[f"local_{subset}_positive_class_percentage"] = round(
                local_positive_class / len(df[df["is_local"]]), 4
            )

        if not df["is_local"].all():
            dataset_info[f"global_{subset}_original_size"] = len(df[~df["is_local"]])
            dataset_info[f"global_{subset}_positive_class_percentage"] = round(
                len(df[~df["is_local"] & df["is_positive_class"]])
                / len(df[~df["is_local"]]),
                4,
            )

        if upsample:
            dataset_info[f"{subset}_upsampled_size"] = len(df) + local_difference

        self.dataset_info = dataset_info

        if upsample:
            if local_positive_class > local_negative_class:
                arrow = "<-"
                df = df.append(
                    df[df["is_local"] & ~df["is_positive_class"]].sample(
                        n=local_difference, replace=True, random_state=42
                    ),
                    ignore_index=True,
                )
            elif local_positive_class < local_negative_class:
                arrow = "->"
                df = df.append(
                    df[df["is_local"] & df["is_positive_class"]].sample(
                        n=local_difference, replace=True, random_state=42
                    ),
                    ignore_index=True,
                )

            print(
                f"Upsampling: local positive class{arrow}negative class: "
                + f"{local_positive_class}{arrow}{local_negative_class}"
            )

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

    def _compute_num_timesteps(
        self, start_col: pd.Series, end_col: pd.Series
    ) -> List[int]:
        df_start_date = pd.to_datetime(start_col).apply(
            lambda dt: dt.replace(month=self.start_month_index + 1)
        )
        df_candidate_end_date = df_start_date.apply(
            lambda dt: dt + relativedelta(months=+self.input_months)
        )
        df_data_end_date = pd.to_datetime(end_col)
        df_end_date = pd.DataFrame(
            {"1": df_data_end_date, "2": df_candidate_end_date}
        ).min(axis=1)
        # Pick min available end date
        timesteps = (
            ((df_end_date - df_start_date) / np.timedelta64(1, "M"))
            .round()
            .unique()
            .astype(int)
        )
        return [int(t) for t in timesteps]

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
            torch.tensor(int(row["is_positive_class"])).float(),
            torch.tensor(int(row["is_local"])).float(),
        )
