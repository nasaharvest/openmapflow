"""
File for storing references to datasets.
"""
from typing import List

from openmapflow.features import create_features
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
    LabeledDataset(
        dataset="geowiki_landcover_2017",
        country="global",
        raw_labels=(
            RawLabels(
                filename="loc_all_2.txt",
                longitude_col="loc_cent_X",
                latitude_col="loc_cent_Y",
                class_prob=0.0,
                start_year=2017,
                x_y_from_centroid=False,
                train_val_test=(0.9, 0.05, 0.05),
                filter_df=lambda df: df[(df.sumcrop / 100) > 0.5],
            ),
        ),
    ),
]

if __name__ == "__main__":
    create_features(datasets)
