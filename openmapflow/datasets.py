import pandas as pd

from openmapflow.constants import CLASS_PROB
from openmapflow.labeled_dataset import LabeledDataset

gcloud_url = "https://storage.googleapis.com/harvest-public-assets/openmapflow/datasets"

label_col = CLASS_PROB


class GeowikiLandcover2017(LabeledDataset):
    def load_labels(self):
        # Source: https://github.com/nasaharvest/crop-mask/blob/master/datasets.py
        df = pd.read_csv(f"{gcloud_url}/crop/geowiki_landcover_2017.csv")
        df = df[df[label_col] != 0.5].copy()
        return df


class TogoCrop2019(LabeledDataset):
    def load_labels(self):
        # Source: https://github.com/nasaharvest/crop-mask/blob/master/datasets.py
        df = pd.read_csv(f"{gcloud_url}/crop/Togo_2019.csv")
        df = df[df[label_col] != 0.5].copy()
        return df


class KenyaCrop201819(LabeledDataset):
    def load_labels(self):
        # Source: https://github.com/nasaharvest/crop-mask/blob/master/datasets.py
        return pd.read_csv(f"{gcloud_url}/crop/Kenya_2018_2019.csv")


datasets = [GeowikiLandcover2017(), TogoCrop2019(), KenyaCrop201819()]
