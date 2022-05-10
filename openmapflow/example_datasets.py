"""
File for storing references to datasets.
"""
from typing import List
from openmapflow.labeled_dataset import LabeledDataset

# from openmapflow.processor import Processor

datasets: List[LabeledDataset] = [
    # --------------------------------------------------------------------------
    # Example LabeledDataset (remove once you have your own)
    # --------------------------------------------------------------------------
    # LabeledDataset(
    #     dataset="example_dataset",
    #     country="Togo",
    #     processors=(
    #         Processor(
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
