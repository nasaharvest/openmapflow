import re
import tempfile
import warnings
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from urllib.request import Request, urlopen

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
    MATCHING_EO_FILES,
    NUM_LABELERS,
    SOURCE,
    START,
    SUBSET,
)
from openmapflow.raw_labels import RawLabels

temp_dir = tempfile.gettempdir()


def _find_nearest(array, value: float) -> Tuple[float, int]:
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx], idx


def _distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
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


def _distance_point_from_center(lat_idx: int, lon_idx: int, tif) -> int:
    x_dist = np.abs((len(tif.x) - 1) / 2 - lon_idx)
    y_dist = np.abs((len(tif.y) - 1) / 2 - lat_idx)
    return x_dist + y_dist


def bbox_from_str(s: str) -> BBox:
    """Generate bbox from str"""
    if not (
        s.find("min_lat") < s.find("min_lon") < s.find("max_lat") < s.find("max_lon")
    ):
        raise ValueError(
            f"Bbox string must go min_lat={{}}_min_lon={{}}_max_lat={{}}_max_lon={{}}, got {s}"
        )
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
def _generate_bbox_from_paths() -> Dict[Path, BBox]:
    cloud_eo_uris = get_cloud_tif_list(BucketNames.LABELED_EO)
    return {
        Path(uri): bbox_from_str(uri)
        for uri in tqdm(cloud_eo_uris, desc="Generating BBoxes from paths")
    }


def _get_tif_paths(
    path_to_bbox: Dict,
    lat: float,
    lon: float,
    start_date: str,
    end_date: str,
    pbar=None,
):
    candidate_paths = []
    for p, bbox in path_to_bbox.items():
        if (
            bbox.contains(lat=lat, lon=lon)
            and f"dates={start_date}_{end_date}" in p.stem
        ):
            candidate_paths.append(p)
    if pbar:
        pbar.update(1)
    return candidate_paths


def _match_labels_to_eo_files(labels: pd.DataFrame) -> pd.Series:
    # Add a bounday to get additional tifs
    bbox_for_labels = BBox(
        min_lon=labels[LON].min() - 1.0,
        min_lat=labels[LAT].min() - 1.0,
        max_lon=labels[LON].max() + 1.0,
        max_lat=labels[LAT].max() + 1.0,
    )
    # Get all eo file paths and bboxes
    path_to_bbox = {
        p: bbox
        for p, bbox in _generate_bbox_from_paths().items()
        if bbox_for_labels.contains_bbox(bbox)
    }

    # Faster than going through bboxes
    with tqdm(
        total=len(labels), desc="Matching labels to earth observation paths"
    ) as pbar:
        eo_file_paths = np.vectorize(_get_tif_paths, otypes=[np.ndarray])(
            path_to_bbox=path_to_bbox,
            lat=labels[LAT],
            lon=labels[LON],
            start_date=labels[START],
            end_date=labels[END],
            pbar=pbar,
        )
    return eo_file_paths


def _find_matching_point(
    start: str, eo_paths: List[Path], label_lon: float, label_lat: float, tif_bucket
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
    for p in eo_paths:
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
            lon, lon_idx = _find_nearest(tif.x, label_lon)
            lat, lat_idx = _find_nearest(tif.y, label_lat)
            distance_from_point = _distance(label_lat, label_lon, lat, lon)
            distance_from_center = _distance_point_from_center(lat_idx, lon_idx, tif)
            if (distance_from_point < min_distance_from_point) or (
                distance_from_point == min_distance_from_point
                and distance_from_center < min_distance_from_center
            ):
                closest_lon = lon
                closest_lat = lat
                min_distance_from_center = distance_from_center
                min_distance_from_point = distance_from_point
                eo_data = tif.sel(x=lon).sel(y=lat).values
                average_slope = slope
                eo_file = eo_paths[i].name
    else:
        tif, slope = tif_slope_tuples[0]
        closest_lon = _find_nearest(tif.x, label_lon)[0]
        closest_lat = _find_nearest(tif.y, label_lat)[0]
        eo_data = tif.sel(x=closest_lon).sel(y=closest_lat).values
        average_slope = slope
        eo_file = eo_paths[0].name

    eo_data = Engineer.calculate_ndvi(eo_data)
    eo_data = Engineer.remove_bands(eo_data)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        eo_data = Engineer.fillna(eo_data, average_slope)

    return eo_data, closest_lon, closest_lat, eo_file


def get_label_timesteps(labels: pd.DataFrame):
    if START not in labels.columns or END not in labels.columns:
        raise ValueError("Labels must have start and end columns")

    diff = pd.to_datetime(labels[END]) - pd.to_datetime(labels[START])
    return (diff / np.timedelta64(1, "M")).round().astype(int)


def _to_np(x: str) -> Optional[np.ndarray]:
    try:
        return np.array(eval(x))
    except TypeError:
        return None


def _clean_df_condition(df: pd.DataFrame) -> pd.Series:
    return (
        (df[EO_STATUS] != EO_STATUS_MISSING_VALUES)
        & (df[EO_STATUS] != EO_STATUS_EXPORT_FAILED)
        & (df[EO_STATUS] != EO_STATUS_DUPLICATE)
        & (df[CLASS_PROB] != 0.5)
    )


def _label_eo_counts(df: pd.DataFrame) -> str:
    df = df[_clean_df_condition(df)]
    label_counts = df[SUBSET].value_counts()
    eo_counts = df[df[EO_DATA].notnull()][SUBSET].value_counts()
    text = ""
    for subset in ["training", "validation", "testing"]:
        if subset not in label_counts:
            continue
        labels_in_subset = label_counts.get(subset, 0)
        features_in_subset = eo_counts.get(subset, 0)
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
    label_type: str = "binary"
    license: str = ""
    source: str = ""

    def __post_init__(self):
        self.df_path = PROJECT_ROOT / dp.DATASETS / (self.dataset + ".csv")

    def summary(self, df: pd.DataFrame) -> str:
        timesteps = get_label_timesteps(df).unique()
        eo_status_str = str(df[EO_STATUS].value_counts()).rsplit("\n", 1)[0]
        return (
            f"{self.dataset} (Timesteps: {','.join([str(int(t)) for t in timesteps])})\n"
            + "----------------------------------------------------------------------------\n"
            + eo_status_str
            + "\n"
            + _label_eo_counts(df)
            + "\n"
        )

    def load_df(self, skip_to_np: bool = False) -> pd.DataFrame:
        """Load dataset (labels + earth observation data) as a DataFrame"""
        if not self.df_path.exists():
            print(self.create_dataset())
        df = pd.read_csv(self.df_path)
        df = df[_clean_df_condition(df)].copy()
        if df[EO_DATA].isnull().any():
            raise ValueError(
                f"{self.dataset} has missing earth observation data, "
                + "run openmapflow create-datasets"
            )
        if not skip_to_np:
            df[EO_DATA] = df[EO_DATA].apply(_to_np)
        return df

    def create_dataset(self):
        raise NotImplementedError


@dataclass
class CustomLabeledDataset(LabeledDataset):

    raw_labels: Tuple[RawLabels, ...] = ()

    def __post_init__(self):
        super().__post_init__()
        self.raw_dir = PROJECT_ROOT / dp.RAW_LABELS / self.dataset

    def _create_df_if_not_exist(self):
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
                EO_STATUS: "first",
            }
        )
        df[COUNTRY] = self.country
        df[DATASET] = self.dataset

        df = df.reset_index(drop=True)
        df.to_csv(self.df_path, index=False)
        return df

    def _duplicates_check(self, df: pd.DataFrame):
        clean_df = df[_clean_df_condition(df)]
        duplicates = clean_df.duplicated(subset=[EO_LAT, EO_LON, EO_FILE])
        if duplicates.sum() > 0:
            print(f"Found {duplicates.sum()} duplicates")
            duplicates_index = clean_df.index[duplicates]
            df.loc[duplicates_index, EO_STATUS] = EO_STATUS_DUPLICATE
            df.loc[duplicates_index, EO_LAT] = None
            df.loc[duplicates_index, EO_LON] = None
            df.loc[duplicates_index, EO_FILE] = None
            df.to_csv(self.df_path, index=False)
        return df

    def create_dataset(self) -> str:
        """
        A dataset consists of (X, y) pairs that are used to train and evaluate
        a machine learning model. In this case,
        - X is the earth observation data for a lat lon coordinate over a 24 month time series
        - y is the binary class label for that coordinate
        To create a dataset:
        1. Obtain the labels
        2. Check if the eath observation data already exists
        3. Use the label coordinates to match to the associated eath observation data (X)
        4. If the eath observation data is missing, download it using Google Earth Engine
        5. Create the dataset
        """
        # ---------------------------------------------------------------------
        # STEP 1: Obtain the labels
        # ---------------------------------------------------------------------
        df = self._create_df_if_not_exist()

        # ---------------------------------------------------------------------
        # STEP 2: Check if earth observation data already available
        # ---------------------------------------------------------------------
        no_eo = _clean_df_condition(df) & (df[EO_DATA].isnull())
        if no_eo.sum() == 0:
            df = self._duplicates_check(df)
            return self.summary(df)

        print(self.summary(df))

        # ---------------------------------------------------------------------
        # STEP 3: Match labels to earth observation files
        # ---------------------------------------------------------------------
        df[MATCHING_EO_FILES] = ""
        df.loc[no_eo, MATCHING_EO_FILES] = _match_labels_to_eo_files(df[no_eo])

        eo_files_found = df[no_eo][MATCHING_EO_FILES].str.len() > 0
        df_with_no_eo_files = df[no_eo].loc[~eo_files_found]
        df_with_eo_files = df[no_eo].loc[eo_files_found]

        # ---------------------------------------------------------------------
        # STEP 4: If no matching earth observation file, download it
        # ---------------------------------------------------------------------
        already_getting_eo = df_with_no_eo_files[EO_STATUS] == EO_STATUS_EXPORTING
        if already_getting_eo.sum() > 0:
            confirm = (
                input(
                    f"{already_getting_eo.sum()} labels were already set to {EO_STATUS_EXPORTING} ,"
                    + " have they failed on EarthEngine? y/[n]: "
                )
                or "n"
            )
            if confirm.lower() == "y":
                df.loc[already_getting_eo, EO_STATUS] = EO_STATUS_EXPORT_FAILED
                df_with_no_eo_files = df_with_no_eo_files.loc[~already_getting_eo]

        if len(df_with_no_eo_files) > 0:
            print(f"{len(df_with_no_eo_files)} labels not matched")
            df_with_no_eo_files[START] = pd.to_datetime(
                df_with_no_eo_files[START]
            ).dt.date
            df_with_no_eo_files[END] = pd.to_datetime(df_with_no_eo_files[END]).dt.date
            EarthEngineExporter(
                check_ee=True, check_gcp=True, dest_bucket=BucketNames.LABELED_EO
            ).export_for_labels(labels=df_with_no_eo_files)
            df.loc[df_with_no_eo_files.index, EO_STATUS] = EO_STATUS_EXPORTING

        # ---------------------------------------------------------------------
        # STEP 5: Create the dataset (earth observation data, label)
        # ---------------------------------------------------------------------
        if len(df_with_eo_files) > 0:
            tif_bucket = storage.Client().bucket(BucketNames.LABELED_EO)

            def set_df(i, start, eo_paths, lon, lat, pbar):
                (eo_data, eo_lon, eo_lat, eo_file) = _find_matching_point(
                    start=start,
                    eo_paths=eo_paths,
                    label_lon=lon,
                    label_lat=lat,
                    tif_bucket=tif_bucket,
                )
                pbar.update(1)
                if eo_data is None:
                    print(
                        "Earth observation file could not be loaded, "
                        + f"setting status to: {EO_STATUS_MISSING_VALUES}"
                    )
                    df.at[i, EO_STATUS] = EO_STATUS_MISSING_VALUES
                elif (
                    (df[EO_FILE] == eo_file)
                    & (df[EO_LAT] == eo_lat)
                    & (df[EO_LON] == eo_lon)
                ).any():
                    print(
                        "Earth observation coordinate already used, "
                        + f"setting status to {EO_STATUS_DUPLICATE}"
                    )
                    df.at[i, EO_STATUS] = EO_STATUS_DUPLICATE
                else:
                    df.at[i, EO_DATA] = eo_data.tolist()
                    df.at[i, EO_LAT] = eo_lat
                    df.at[i, EO_LON] = eo_lon
                    df.at[i, EO_FILE] = eo_file
                    df.at[i, EO_STATUS] = EO_STATUS_COMPLETE

            with tqdm(
                total=len(df_with_eo_files),
                desc="Extracting matched earth observation data",
            ) as pbar:
                np.vectorize(set_df)(
                    i=df_with_eo_files.index,
                    start=df_with_eo_files[START],
                    eo_paths=df_with_eo_files[MATCHING_EO_FILES],
                    lon=df_with_eo_files[LON],
                    lat=df_with_eo_files[LAT],
                    pbar=pbar,
                )

            df.drop(columns=[MATCHING_EO_FILES], inplace=True)
            df.to_csv(self.df_path, index=False)
        return self.summary(df)


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


def create_datasets(datasets: List[LabeledDataset]):
    report = "DATASET REPORT (autogenerated, do not edit directly)"
    for d in datasets:
        summary = d.create_dataset()
        print(summary)
        report += "\n\n" + summary

    with (PROJECT_ROOT / dp.REPORT).open("w") as f:
        f.write(report)
