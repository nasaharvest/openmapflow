from typing import List
from openmapflow import LabeledDataset, Processor

datasets: List[LabeledDataset] = [
    LabeledDataset(
        dataset="Togo_2019",
        country="Togo",
        processors=(
            Processor(
                filename="crop_merged_v2.zip",
                class_prob=1.0,
                train_val_test=(0.8, 0.2, 0.0),
                start_year=2019,
            ),
            Processor(
                filename="noncrop_merged_v2.zip",
                class_prob=0.0,
                train_val_test=(0.8, 0.2, 0.0),
                start_year=2019,
            ),
            Processor(
                filename="random_sample_hrk.zip",
                class_prob=lambda df: df["hrk-label"],
                transform_crs_from=32631,
                train_val_test=(0.0, 0.0, 1.0),
                start_year=2019,
            ),
            Processor(
                filename="random_sample_cn.zip",
                class_prob=lambda df: df["cn_labels"],
                train_val_test=(0.0, 0.0, 1.0),
                start_year=2019,
            ),
            Processor(
                filename="BB_random_sample_1k.zip",
                class_prob=lambda df: df["bb_label"],
                train_val_test=(0.0, 0.0, 1.0),
                start_year=2019,
            ),
            Processor(
                filename="random_sample_bm.zip",
                class_prob=lambda df: df["bm_labels"],
                train_val_test=(0.0, 0.0, 1.0),
                start_year=2019,
            ),
        ),
    ),
]
