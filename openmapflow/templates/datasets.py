"""
File for storing references to datasets.
"""
from typing import List

# from openmapflow.datasets import geowiki_landcover_2017
from openmapflow.labeled_dataset import LabeledDataset, create_datasets

# from openmapflow.labeled_dataset_custom import CustomLabeledDataset
# from openmapflow.raw_labels import RawLabels

datasets: List[LabeledDataset] = [
    # --------------------------------------------------------------------------
    # Example ExistingLabeledDataset
    # --------------------------------------------------------------------------
    # geowiki_landcover_2017,
    #
    #
    # --------------------------------------------------------------------------
    # Example CustomLabeledDataset
    # --------------------------------------------------------------------------
    # CustomLabeledDataset(
    #     dataset="example_dataset",
    #     country="Togo",
    #     raw_labels=(
    #         RawLabels(
    #             filename="Togo_2019.csv",
    #             longitude_col="longitude",
    #             latitude_col="latitude",
    #             class_prob=lambda df: df["crop"],
    #             start_year=2019,
    #             x_y_from_centroid=False,
    #         ),
    #     ),
    # ),
]

if __name__ == "__main__":
    create_datasets(datasets)
