from typing import List

from openmapflow.constants import CLASS_PROB
from openmapflow.datasets import GeowikiLandcover2017, TogoCrop2019
from openmapflow.labeled_dataset import LabeledDataset, create_datasets

label_col = CLASS_PROB
datasets: List[LabeledDataset] = [GeowikiLandcover2017(), TogoCrop2019()]

if __name__ == "__main__":
    create_datasets(datasets)
