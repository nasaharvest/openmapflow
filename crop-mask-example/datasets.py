from typing import List

from openmapflow.datasets import geowiki_landcover_2017, togo_crop_2019
from openmapflow.labeled_dataset import LabeledDataset, create_datasets

datasets: List[LabeledDataset] = [geowiki_landcover_2017, togo_crop_2019]

if __name__ == "__main__":
    create_datasets(datasets)
