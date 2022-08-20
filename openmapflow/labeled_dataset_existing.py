from dataclasses import dataclass
from urllib.request import Request, urlopen

import pandas as pd
from tqdm import tqdm

from openmapflow.labeled_dataset import LabeledDataset


@dataclass
class ExistingLabeledDataset(LabeledDataset):

    download_url: str = ""
    chunk_size: int = 1024

    def __post_init__(self):
        super().__post_init__()

    def create_dataset(self):
        if not self.df_path.exists():
            self.df_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.df_path, "wb") as fh:
                with urlopen(Request(self.download_url)) as response:
                    total = response.length // self.chunk_size
                    with tqdm(total=total, desc=f"Downloading {self.dataset}") as pbar:
                        for chunk in iter(lambda: response.read(self.chunk_size), ""):
                            if not chunk:
                                break
                            pbar.update(1)
                            fh.write(chunk)

        df = pd.read_csv(self.df_path)
        return self.summary(df)
