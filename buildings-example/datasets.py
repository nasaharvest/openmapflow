"""
File for storing references to datasets.
"""
from datetime import date
from typing import List

import pandas as pd

from openmapflow.config import PROJECT_ROOT, DataPaths
from openmapflow.constants import END, LAT, LON, START, SUBSET
from openmapflow.label_utils import train_val_test_split
from openmapflow.labeled_dataset import LabeledDataset, create_datasets

raw_dir = PROJECT_ROOT / DataPaths.RAW_LABELS


class UgandaBuildings2020(LabeledDataset):
    def load_labels(self):
        df = pd.read_csv(raw_dir / "Uganda_177_buildings_confidence_0.9.csv")
        df.rename(columns={"latitude": LAT, "longitude": LON}, inplace=True)
        df[START] = date(2020, 1, 1)
        df[END] = date(2021, 12, 31)
        df["is_building"] = 1.0
        df[SUBSET] = train_val_test_split(index=df.index, val=0.1, test=0.1)
        return df


class UgandaParks2020(LabeledDataset):
    def load_labels(self):
        df = pd.read_csv(raw_dir / "Uganda_parks_2020.csv")
        df["is_building"] = 0.0
        return df


class GeowikiLandcover2017(LabeledDataset):
    def load_labels(self):
        df = pd.read_csv(raw_dir / "loc_all_2.txt", sep="\t")
        df = df[(df.sumcrop / 100) > 0.5].copy()
        df.rename(columns={"loc_cent_Y": LAT, "loc_cent_X": LON}, inplace=True)
        df[START] = date(2017, 1, 1)
        df[END] = date(2018, 12, 31)
        df["is_building"] = 0.0
        df[SUBSET] = train_val_test_split(index=df.index, val=0.05, test=0.05)
        return df


datasets: List[LabeledDataset] = [
    UgandaBuildings2020(),
    UgandaParks2020(),
    GeowikiLandcover2017(),
]

if __name__ == "__main__":
    create_datasets(datasets)
