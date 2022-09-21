"""
File for storing references to datasets.
"""
from datetime import date
from typing import List

import numpy as np
import pandas as pd
from pandas.compat._optional import import_optional_dependency

from openmapflow.config import PROJECT_ROOT, DataPaths
from openmapflow.constants import COUNTRY, END, LAT, LON, START, SUBSET
from openmapflow.label_utils import (
    get_lat_lon_from_centroid,
    read_zip,
    train_val_test_split,
)
from openmapflow.labeled_dataset import LabeledDataset, create_datasets
from openmapflow.utils import to_date

raw_dir = PROJECT_ROOT / DataPaths.RAW_LABELS
label_col = "is_maize"


class KenyaNonCrop2019(LabeledDataset):
    def load_labels(self):
        df1 = read_zip(raw_dir / "Kenya_non_crop_2019/noncrop_labels_v2.zip")
        df1 = df1[df1.geometry.notna()].copy()
        df1[LAT], df1[LON] = get_lat_lon_from_centroid(df1.geometry, src_crs=32636)

        df2 = read_zip(raw_dir / "Kenya_non_crop_2019/2019_gepro_noncrop.zip")
        df2[LAT], df2[LON] = get_lat_lon_from_centroid(df2.geometry, src_crs=32636)

        df3 = read_zip(raw_dir / "Kenya_non_crop_2019/noncrop_water_kenya_gt.zip")
        df3[LAT], df3[LON] = get_lat_lon_from_centroid(df3.geometry)

        df4 = read_zip(raw_dir / "Kenya_non_crop_2019/noncrop_kenya_gt.zip")
        df4[LAT], df4[LON] = get_lat_lon_from_centroid(df4.geometry)

        df5 = read_zip(raw_dir / "Kenya_non_crop_2019/kenya_non_crop_test_polygons.zip")
        df5[LAT], df5[LON] = get_lat_lon_from_centroid(df5.geometry)

        df = pd.concat([df1, df2, df3, df4, df5])
        df[START], df[END] = date(2019, 1, 1), date(2020, 12, 31)
        df[SUBSET] = train_val_test_split(index=df.index, val=0.2, test=0.2)
        df[label_col] = 0.0
        df[COUNTRY] = "Kenya"
        return df


class RefAfricanCropsKenya01Labels(LabeledDataset):
    def load_labels(self) -> pd.DataFrame:
        rack01 = "ref_african_crops_kenya_01_labels"
        gpd = import_optional_dependency("geopandas")

        df1 = gpd.read_file(raw_dir / rack01 / f"{rack01}_00/labels.geojson")
        df2 = gpd.read_file(raw_dir / rack01 / f"{rack01}_01/labels.geojson")
        df3 = gpd.read_file(raw_dir / rack01 / f"{rack01}_02/labels.geojson")
        df = pd.concat([df1, df2, df3])
        df.rename(columns={"Latitude": LAT, "Longitude": LON}, inplace=True)
        df[START] = np.vectorize(to_date)(df["Planting Date"])
        df[START] = np.vectorize(lambda d: d.replace(month=1, day=1))(df[START])
        df[END] = np.vectorize(lambda d: d.replace(year=d.year + 1, month=12, day=31))(
            df[START]
        )
        df[SUBSET] = train_val_test_split(index=df.index, val=0.2, test=0.2)
        df[label_col] = (df["Crop1"] == "Maize").astype(float)
        df[COUNTRY] = "Kenya"
        return df


datasets: List[LabeledDataset] = [KenyaNonCrop2019(), RefAfricanCropsKenya01Labels()]

if __name__ == "__main__":
    create_datasets(datasets)
