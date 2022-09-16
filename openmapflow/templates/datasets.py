"""
File for storing references to datasets.
"""
from datetime import date
from typing import List

import pandas as pd

from openmapflow.config import PROJECT_ROOT, DataPaths
from openmapflow.constants import CLASS_PROB, COUNTRY, END, LAT, LON, START, SUBSET
from openmapflow.datasets import GeowikiLandcover2017
from openmapflow.label_utils import train_val_test_split
from openmapflow.labeled_dataset import LabeledDataset, create_datasets

label_col = CLASS_PROB


# -----------------------------------------------------------------------------
# Example custom dataset to be used as reference
# -----------------------------------------------------------------------------
class TogoCrop2019(LabeledDataset):
    def load_labels(self) -> pd.DataFrame:
        # Read in raw label file
        df = pd.read_csv(PROJECT_ROOT / DataPaths.RAW_LABELS / "Togo_2019.csv")

        # Rename coordinate columns to be used for getting Earth observation data
        df.rename(columns={"latitude": LAT, "longitude": LON}, inplace=True)

        # Set start and end date for Earth observation data
        df[START], df[END] = date(2019, 1, 1), date(2020, 12, 31)

        # Set consistent label column
        df[label_col] = df["crop"].astype(float)

        # Split labels into train, validation, and test sets
        df[SUBSET] = train_val_test_split(index=df.index, val=0.2, test=0.2)

        # Set country column for later analysis
        df[COUNTRY] = "Togo"

        return df


datasets: List[LabeledDataset] = [GeowikiLandcover2017(), TogoCrop2019()]

if __name__ == "__main__":
    create_datasets(datasets)
