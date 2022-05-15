"""
File for storing references to datasets.
"""
from typing import List

from openmapflow.labeled_dataset import LabeledDataset
from openmapflow.raw_labels import RawLabels

datasets: List[LabeledDataset] = [
    LabeledDataset(
        dataset="Uganda_buildings_2020",
        country="Uganda",
        raw_labels=(
            RawLabels(
                filename="177_buildings_confidence_0.9.csv",
                latitude_col="latitude",
                longitude_col="longitude",
                class_prob=1.0,
                start_year=2020,
                x_y_from_centroid=False,
                train_val_test=(0.8, 0.1, 0.1),
            ),
        ),
    ),
]
