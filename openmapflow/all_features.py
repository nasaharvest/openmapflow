import pandas as pd
import pickle

from .config import full_paths
from .utils import try_txt_read


class AllFeatures:
    def __init__(self) -> pd.DataFrame:
        duplicates_data = try_txt_read(full_paths["duplicates"])
        features = []
        files = list(full_paths["features"].glob("*.pkl"))
        print("------------------------------")
        print("Loading all features...")
        non_duplicated_files = []
        for p in files:
            if p.stem not in duplicates_data:
                non_duplicated_files.append(p)
                with p.open("rb") as f:
                    features.append(pickle.load(f))
        df = pd.DataFrame([feat.__dict__ for feat in features])
        df["filename"] = non_duplicated_files
        self.df = df

    def check_empty(self) -> str:
        """
        Some exported tif data may have nan values
        """
        empties = self.df[self.df["labelled_array"].isnull()]
        num_empty = len(empties)
        if num_empty > 0:
            return f"\u2716 Found {num_empty} empty features"
        else:
            return "\u2714 Found no empty features"

    def check_duplicates(self) -> str:
        """
        Can happen when not all tifs have been downloaded and different labels are matched to same tif
        """
        cols_to_check = ["instance_lon", "instance_lat", "source_file"]
        duplicates = self.df[self.df.duplicated(subset=cols_to_check)]
        num_dupes = len(duplicates)
        if num_dupes > 0:
            return f"\u2716 Found {num_dupes} duplicates"
        else:
            return "\u2714 No duplicates found"
