from openmapflow.labeled_dataset import LabeledDataset

import pandas as pd

gcloud_url = "https://storage.googleapis.com/harvest-public-assets/openmapflow/datasets"

class GeowikiLandcover2017(LabeledDataset):
    """Source: https://github.com/nasaharvest/crop-mask/blob/master/datasets.py"""
    def load_labels(self):
        return pd.read_csv(f"{gcloud_url}/crop/geowiki_landcover_2017.csv")
            
class TogoCrop2019(LabeledDataset):
    """Source: https://github.com/nasaharvest/crop-mask/blob/master/datasets.py"""
    def load_labels(self):
        return pd.read_csv(f"{gcloud_url}/crop/Togo_2019.csv")

class KenyaCrop201819(LabeledDataset):
    """Source: https://github.com/nasaharvest/crop-mask/blob/master/datasets.py"""
    def load_labels(self):
        return pd.read_csv(f"{gcloud_url}/crop/Kenya_2018_2019.csv")

datasets = [GeowikiLandcover2017(), TogoCrop2019(), KenyaCrop201819()]
