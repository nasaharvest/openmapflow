import re
import tempfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from cropharvest.countries import BBox
from cropharvest.engineer import Engineer
from cropharvest.eo import EarthEngineExporter
from cropharvest.eo.eo import get_cloud_tif_list
from cropharvest.utils import memoized
from google.cloud import storage
from tqdm import tqdm

from openmapflow.config import PROJECT_ROOT, BucketNames
from openmapflow.config import DataPaths as dp
from openmapflow.constants import (
    CLASS_PROB,
    COUNTRY,
    DATASET,
    END,
    EO_DATA,
    EO_FILE,
    EO_LAT,
    EO_LON,
    EO_STATUS,
    EO_STATUS_COMPLETE,
    EO_STATUS_DUPLICATE,
    EO_STATUS_EXPORT_FAILED,
    EO_STATUS_EXPORTING,
    EO_STATUS_MISSING_VALUES,
    LABEL_DUR,
    LABELER_NAMES,
    LAT,
    LON,
    NUM_LABELERS,
    SOURCE,
    START,
    SUBSET,
    TIF_PATHS,
)
from openmapflow.raw_labels import RawLabels

temp_dir = tempfile.gettempdir()


def find_nearest(array, value: float) -> Tuple[float, int]:
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx], idx


def distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    haversince formula, inspired by:
    https://stackoverflow.com/questions/41336756/find-the-closest-latitude-and-longitude/41337005
    """
    p = 0.017453292519943295
    a = (
        0.5
        - np.cos((lat2 - lat1) * p) / 2
        + np.cos(lat1 * p) * np.cos(lat2 * p) * (1 - np.cos((lon2 - lon1) * p)) / 2
    )
    return 12742 * np.arcsin(np.sqrt(a))


def distance_point_from_center(lat_idx: int, lon_idx: int, tif) -> int:
    x_dist = np.abs((len(tif.x) - 1) / 2 - lon_idx)
    y_dist = np.abs((len(tif.y) - 1) / 2 - lat_idx)
    return x_dist + y_dist


def bbox_from_str(s: str) -> BBox:
    """
    Generate bbox from str
    """
    decimals_in_p = re.findall(r"=-?\d*\.?\d*", Path(s).name)
    coords = [float(d[1:]) for d in decimals_in_p[0:4]]
    bbox = BBox(
        min_lat=coords[0],
        min_lon=coords[1],
        max_lat=coords[2],
        max_lon=coords[3],
        name=s,
    )
    return bbox


@memoized
def generate_bbox_from_paths() -> Dict[Path, BBox]:
    cloud_tif_uris = [uri for uri in get_cloud_tif_list(BucketNames.LABELED_TIFS)]
    return {
        Path(uri): bbox_from_str(uri)
        for uri in tqdm(cloud_tif_uris, desc="Generating BBoxes from paths")
    }


def get_tif_paths(path_to_bbox, lat, lon, start_date, end_date, pbar):
    candidate_paths = []
    for p, bbox in path_to_bbox.items():
        if (
            bbox.contains(lat=lat, lon=lon)
            and f"dates={start_date}_{end_date}" in p.stem
        ):
            candidate_paths.append(p)
    pbar.update(1)
    return candidate_paths


def match_labels_to_tifs(labels: pd.DataFrame) -> pd.Series:
    # Add a bounday to get additional tifs
    bbox_for_labels = BBox(
        min_lon=labels[LON].min() - 1.0,
        min_lat=labels[LAT].min() - 1.0,
        max_lon=labels[LON].max() + 1.0,
        max_lat=labels[LAT].max() + 1.0,
    )
    # Get all tif paths and bboxes
    path_to_bbox = {
        p: bbox
        for p, bbox in generate_bbox_from_paths().items()
        if bbox_for_labels.contains_bbox(bbox)
    }

    # Match labels to tif files
    # Faster than going through bboxes
    with tqdm(total=len(labels), desc="Matching labels to tif paths") as pbar:
        tif_paths = np.vectorize(get_tif_paths, otypes=[np.ndarray])(
            path_to_bbox=path_to_bbox,
            lat=labels[LAT],
            lon=labels[LON],
            start_date=labels[START],
            end_date=labels[END],
            pbar=pbar,
        )
    return tif_paths


def find_matching_point(
    start: str, tif_paths: List[Path], label_lon: float, label_lat: float, tif_bucket
) -> Tuple[np.ndarray, float, float, str]:
    """
    Given a label coordinate (y) this functions finds the associated satellite data (X)
    by looking through one or multiple tif files.
    Each tif file contains satellite data for a grid of coordinates.
    So the function finds the closest grid coordinate to the label coordinate.
    Additional value is given to a grid coordinate that is close to the center of the tif.
    """
    start_date = datetime.strptime(start, "%Y-%m-%d")
    tif_slope_tuples = []
    for p in tif_paths:
        blob = tif_bucket.blob(str(p))
        local_path = Path(f"{temp_dir}/{p.name}")
        if not local_path.exists():
            blob.download_to_filename(str(local_path))
        tif_slope_tuples.append(
            Engineer.load_tif(
                str(local_path), start_date=start_date, num_timesteps=None
            )
        )
        if local_path.exists():
            local_path.unlink()

    if len(tif_slope_tuples) > 1:
        min_distance_from_point = np.inf
        min_distance_from_center = np.inf
        for i, tif_slope_tuple in enumerate(tif_slope_tuples):
            tif, slope = tif_slope_tuple
            lon, lon_idx = find_nearest(tif.x, label_lon)
            lat, lat_idx = find_nearest(tif.y, label_lat)
            distance_from_point = distance(label_lat, label_lon, lat, lon)
            distance_from_center = distance_point_from_center(lat_idx, lon_idx, tif)
            if (distance_from_point < min_distance_from_point) or (
                distance_from_point == min_distance_from_point
                and distance_from_center < min_distance_from_center
            ):
                closest_lon = lon
                closest_lat = lat
                min_distance_from_center = distance_from_center
                min_distance_from_point = distance_from_point
                labelled_np = tif.sel(x=lon).sel(y=lat).values
                average_slope = slope
                source_file = tif_paths[i].name
    else:
        tif, slope = tif_slope_tuples[0]
        closest_lon = find_nearest(tif.x, label_lon)[0]
        closest_lat = find_nearest(tif.y, label_lat)[0]
        labelled_np = tif.sel(x=closest_lon).sel(y=closest_lat).values
        average_slope = slope
        source_file = tif_paths[0].name

    labelled_np = Engineer.calculate_ndvi(labelled_np)
    labelled_np = Engineer.remove_bands(labelled_np)
    labelled_np = Engineer.fillna(labelled_np, average_slope)

    return labelled_np, closest_lon, closest_lat, source_file


def get_label_timesteps(labels):
    diff = pd.to_datetime(labels[END]) - pd.to_datetime(labels[START])
    return (diff / np.timedelta64(1, "M")).round().astype(int)


@dataclass
class LabeledDataset:
    """
    A labeled dataset represents a DataFrame where each row consists of:
    - A coordinate
    - A binary label for that coordinate (y)
    - The earth observation data for that coordinate (X)
    Together labels (y) and the associated earth observation data (X) can be used
    to train and evaluate a macine learning model a model.

    Args:
        dataset (str): The name of the dataset.
        country (str): The country of the dataset (can be 'global' for global datasets).
        raw_labels (Tuple[RawLabels, ...]): A list of raw labels used to create the dataset.
    """

    dataset: str = ""
    country: str = ""
    raw_labels: Tuple[RawLabels, ...] = ()

    def __post_init__(self):
        self.raw_dir = PROJECT_ROOT / dp.RAW_LABELS / self.dataset
        self.df_path = PROJECT_ROOT / "data" / "features_csv" / (self.dataset + ".csv")
        self._cached_labels_csv = None

    def summary(self, df=None):
        if df is None:
            df = self.load_df(allow_processing=False, fail_if_missing_features=False)
        text = f"{self.dataset} "
        timesteps = get_label_timesteps(df).unique()
        text += f"(Timesteps: {','.join([str(int(t)) for t in timesteps])})\n"
        text += "----------------------------------------------------------------------------\n"
        train_val_test_counts = df[SUBSET].value_counts()
        for subset, labels_in_subset in train_val_test_counts.items():
            features_in_subset = df[df[SUBSET] == subset][EO_DATA].notnull().sum()
            if labels_in_subset != features_in_subset:
                text += (
                    f"\u2716 {subset}: {labels_in_subset} labels, "
                    + f"but {features_in_subset} features\n"
                )

            else:
                positive_class_percentage = (
                    df[df[SUBSET] == subset][CLASS_PROB] > 0.5
                ).sum() / labels_in_subset
                text += (
                    f"\u2714 {subset} amount: {labels_in_subset}, "
                    + f"positive class: {positive_class_percentage:.1%}\n"
                )

        return text

    def create_df(self):
        """
        Creates a single processed labels file from a list of raw labels.
        """
        df = pd.DataFrame({})
        already_processed = []
        if self.df_path.exists():
            df = pd.read_csv(self.df_path)
            already_processed = df[SOURCE].unique()

        new_labels: List[pd.DataFrame] = []
        for p in self.raw_labels:
            if p.filename not in str(already_processed):
                new_labels.append(p.process(self.raw_dir))

        if len(new_labels) == 0:
            return df

        df = pd.concat([df] + new_labels)

        # Combine duplicate labels
        df[NUM_LABELERS] = 1

        def join_if_exists(values):
            if all((isinstance(v, str) for v in values)):
                return ",".join(values)
            return ""

        df = df.groupby([LON, LAT, START, END], as_index=False, sort=False).agg(
            {
                SOURCE: lambda sources: ",".join(sources.unique()),
                CLASS_PROB: "mean",
                NUM_LABELERS: "sum",
                SUBSET: "first",
                LABEL_DUR: join_if_exists,
                LABELER_NAMES: join_if_exists,
                EO_DATA: "first",
                EO_LAT: "first",
                EO_LON: "first",
                EO_FILE: "first",
            }
        )
        df[COUNTRY] = self.country
        df[DATASET] = self.dataset

        df = df.reset_index(drop=True)
        df.to_csv(self.df_path, index=False)
        return df

    def load_df(
        self,
        allow_processing: bool = False,
        fail_if_missing_features: bool = False,
    ) -> pd.DataFrame:
        if allow_processing:
            df = self.create_df()
            self._cached_df = df
        elif self._cached_labels_csv is not None:
            df = self._cached_df
        elif self.df_path.exists():
            df = pd.read_csv(self.df_path)
            self._cached_df = df
        else:
            raise FileNotFoundError(f"{self.df_path} does not exist")

        df = df[
            (df[EO_STATUS] != EO_STATUS_MISSING_VALUES)
            & (df[EO_STATUS] != EO_STATUS_EXPORT_FAILED)
            & (df[EO_STATUS] != EO_STATUS_DUPLICATE)
            & (df[CLASS_PROB] != 0.5)
        ].copy()

        df[EO_DATA] = df[EO_DATA].apply(lambda x: np.array(eval(x)) if x else None)

        if fail_if_missing_features and not df[EO_DATA].all():
            raise ValueError(f"{self.dataset} has missing features")
        return df

    def create_features(self, disable_gee_export: bool = False) -> str:
        """
        Features are the (X, y) pairs that are used to train and evaluate a machine learning model.
        In this case,
        - X is the earth observation data for a lat lon coordinate over a 24 month time series
        - y is the binary class label for that coordinate
        To create the features:
        1. Obtain the labels
        2. Check if the features already exist
        3. Use the label coordinates to match to the associated satellite data (X)
        4. If the satellite data is missing, download it using Google Earth Engine
        5. Create the features (X, y)
        """
        print("------------------------------")
        print(self.dataset)

        # -------------------------------------------------
        # STEP 1: Obtain the labels
        # -------------------------------------------------
        df = self.load_df(allow_processing=True)

        # -------------------------------------------------
        # STEP 2: Check if features already exist
        # -------------------------------------------------
        if df[EO_DATA].isnull().sum() == 0:
            return self.summary(df=df)

        # -------------------------------------------------
        # STEP 3: Match labels to tif files (X)
        # -------------------------------------------------
        df[TIF_PATHS] = ""
        tifs = match_labels_to_tifs(df[df[EO_DATA].isnull()])
        df.loc[df[EO_DATA].isnull(), TIF_PATHS] = tifs
        tifs_found = df[df[EO_DATA].isnull()][TIF_PATHS].str.len() > 0

        df_with_no_tifs = df[df[EO_DATA].isnull()].loc[~tifs_found]
        df_with_tifs_but_no_features = df[df[EO_DATA].isnull()].loc[tifs_found]

        # -------------------------------------------------
        # STEP 4: If no matching tif, download it
        # -------------------------------------------------
        already_exporting = df_with_no_tifs[
            df_with_no_tifs[EO_STATUS] == EO_STATUS_EXPORTING
        ]
        if len(already_exporting) > 0:
            confirm = (
                input(
                    f"{len(already_exporting)} labels were already set to {EO_STATUS_EXPORTING} ,"
                    + " add to failed export list? y/[n]: "
                )
                or "n"
            )
            if confirm.lower() == "y":
                df.loc[already_exporting.index, EO_STATUS] = EO_STATUS_EXPORT_FAILED
                df_with_no_tifs = df_with_no_tifs.loc[~already_exporting.index]

        if len(df_with_no_tifs) > 0:
            print(f"{len(df_with_no_tifs )} labels not matched")

            if not disable_gee_export:
                df_with_no_tifs[START] = pd.to_datetime(df_with_no_tifs[START]).dt.date
                df_with_no_tifs[END] = pd.to_datetime(df_with_no_tifs[END]).dt.date
                EarthEngineExporter(
                    check_ee=True,
                    check_gcp=True,
                    dest_bucket=BucketNames.LABELED_TIFS,
                ).export_for_labels(labels=df_with_no_tifs)
                df.loc[df_with_no_tifs.index, EO_STATUS] = EO_STATUS_EXPORTING

        # -------------------------------------------------
        # STEP 5: Create the features (X, y)
        # -------------------------------------------------
        if len(df_with_tifs_but_no_features) > 0:
            tif_bucket = storage.Client().bucket(BucketNames.LABELED_TIFS)

            def set_df(i, start, tif_paths, lon, lat, pbar):
                (eo_data, eo_lon, eo_lat, eo_file) = find_matching_point(
                    start=start,
                    tif_paths=tif_paths,
                    label_lon=lon,
                    label_lat=lat,
                    tif_bucket=tif_bucket,
                )
                pbar.update(1)
                if eo_data is None:
                    df.at[i, EO_STATUS] = EO_STATUS_MISSING_VALUES
                    return

                df.at[i, EO_DATA] = eo_data.tolist()
                df.at[i, EO_LAT] = eo_lat
                df.at[i, EO_LON] = eo_lon
                df.at[i, EO_FILE] = eo_file
                df.at[i, EO_STATUS] = EO_STATUS_COMPLETE

            with tqdm(
                total=len(df_with_tifs_but_no_features),
                desc="Matching labels to tif paths",
            ) as pbar:
                np.vectorize(set_df)(
                    i=df_with_tifs_but_no_features.index,
                    start=df_with_tifs_but_no_features[START],
                    tif_paths=df_with_tifs_but_no_features[TIF_PATHS],
                    lon=df_with_tifs_but_no_features[LON],
                    lat=df_with_tifs_but_no_features[LAT],
                    pbar=pbar,
                )

            df.to_csv(self.df_path, index=False)
        return self.summary(df=df)
