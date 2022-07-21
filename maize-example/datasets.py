"""
File for storing references to datasets.
"""
from typing import List

from openmapflow.labeled_dataset import (
    CustomLabeledDataset,
    LabeledDataset,
    create_datasets,
)
from openmapflow.raw_labels import RawLabels

datasets: List[LabeledDataset] = [
    CustomLabeledDataset(
        dataset="Kenya_non_crop_2019",
        country="Kenya",
        raw_labels=(
            RawLabels(
                filename="noncrop_labels_v2.zip",
                class_prob=0,
                start_year=2019,
                transform_crs_from=32636,
                train_val_test=(0.6, 0.2, 0.2),
            ),
            RawLabels(
                filename="2019_gepro_noncrop.zip",
                class_prob=0,
                start_year=2019,
                transform_crs_from=32636,
                train_val_test=(0.6, 0.2, 0.2),
            ),
            RawLabels(
                filename="noncrop_water_kenya_gt.zip",
                class_prob=0,
                start_year=2019,
                train_val_test=(0.6, 0.2, 0.2),
            ),
            RawLabels(
                filename="noncrop_kenya_gt.zip",
                class_prob=0,
                start_year=2019,
                train_val_test=(0.6, 0.2, 0.2),
            ),
            RawLabels(
                filename="kenya_non_crop_test_polygons.zip",
                class_prob=0,
                start_year=2019,
                train_val_test=(0.6, 0.2, 0.2),
            ),
        ),
    ),
    CustomLabeledDataset(
        dataset="ref_african_crops_kenya_01_labels",
        country="Kenya",
        raw_labels=tuple(
            RawLabels(
                filename=f"ref_african_crops_kenya_01_labels_0{i}/labels.geojson",
                latitude_col="Latitude",
                longitude_col="Longitude",
                class_prob=lambda df: df["Crop1"] == "Maize",
                start_date_col="Planting Date",
                x_y_from_centroid=False,
                train_val_test=(0.6, 0.2, 0.2),
            )
            for i in [0, 1, 2]
        ),
    ),
]

if __name__ == "__main__":
    create_datasets(datasets)
