from typing import List

from openmapflow.datasets import GeowikiLandcover2017, TogoCrop2019
from openmapflow.labeled_dataset import LabeledDataset, create_datasets

datasets: List[LabeledDataset] = [GeowikiLandcover2017(), TogoCrop2019()]

if __name__ == "__main__":
    create_datasets(datasets)
