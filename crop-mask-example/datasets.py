from typing import List

from openmapflow.features import create_features
from openmapflow.labeled_dataset import LabeledDataset
from openmapflow.raw_labels import RawLabels

datasets: List[LabeledDataset] = [
    LabeledDataset(
        dataset="geowiki_landcover_2017",
        country="global",
        raw_labels=(
            RawLabels(
                filename="loc_all_2.txt",
                longitude_col="loc_cent_X",
                latitude_col="loc_cent_Y",
                class_prob=lambda df: df.sumcrop / 100,
                start_year=2017,
                x_y_from_centroid=False,
            ),
        ),
    ),
    LabeledDataset(
        dataset="Togo_2019",
        country="Togo",
        raw_labels=(
            RawLabels(
                filename="crop_merged_v2.zip",
                class_prob=1.0,
                train_val_test=(0.8, 0.2, 0.0),
                start_year=2019,
            ),
            RawLabels(
                filename="noncrop_merged_v2.zip",
                class_prob=0.0,
                train_val_test=(0.8, 0.2, 0.0),
                start_year=2019,
            ),
            RawLabels(
                filename="random_sample_hrk.zip",
                class_prob=lambda df: df["hrk-label"],
                transform_crs_from=32631,
                train_val_test=(0.0, 0.0, 1.0),
                start_year=2019,
            ),
            RawLabels(
                filename="random_sample_cn.zip",
                class_prob=lambda df: df["cn_labels"],
                train_val_test=(0.0, 0.0, 1.0),
                start_year=2019,
            ),
            RawLabels(
                filename="BB_random_sample_1k.zip",
                class_prob=lambda df: df["bb_label"],
                train_val_test=(0.0, 0.0, 1.0),
                start_year=2019,
            ),
            RawLabels(
                filename="random_sample_bm.zip",
                class_prob=lambda df: df["bm_labels"],
                train_val_test=(0.0, 0.0, 1.0),
                start_year=2019,
            ),
        ),
    ),
]

if __name__ == "__main__":
    create_features(datasets)
