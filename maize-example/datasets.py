"""
File for storing references to datasets.
"""
from typing import List

from openmapflow.labeled_dataset import LabeledDataset
from openmapflow.processor import Processor

datasets: List[LabeledDataset] = [
    LabeledDataset(
        dataset="Kenya_non_crop_2019",
        country="Kenya",
        processors=(
            Processor(
                filename="noncrop_labels_v2.zip",
                class_prob=0,
                start_year=2019,
                transform_crs_from=32636,
            ),
            Processor(
                filename="2019_gepro_noncrop.zip",
                class_prob=0,
                start_year=2019,
                transform_crs_from=32636,
            ),
            Processor(
                filename="noncrop_water_kenya_gt.zip", class_prob=0, start_year=2019
            ),
            Processor(filename="noncrop_kenya_gt.zip", class_prob=0, start_year=2019),
            Processor(
                filename="kenya_non_crop_test_polygons.zip",
                class_prob=0,
                start_year=2019,
            ),
        ),
    ),
    LabeledDataset(
        dataset="ref_african_crops_kenya_01_labels",
        country="Kenya",
        processors=(
            Processor(
                filename=f"ref_african_crops_kenya_01_labels_0{i}/labels.geojson",
                latitude_col="Latitude",
                longitude_col="Longitude",
                class_prob=lambda df: df["Crop1"] == "Maize",
                plant_date_col="Planting Date",
                x_y_from_centroid=False,
            )
            for i in [0, 1, 2]
        ),
    ),
]
