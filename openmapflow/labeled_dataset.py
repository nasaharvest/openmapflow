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
    ALREADY_EXISTS,
    CLASS_PROB,
    COUNTRY,
    DATASET,
    END,
    FEATURE_FILENAME,
    FEATURE_PATH,
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
from openmapflow.features import create_feature
from openmapflow.raw_labels import RawLabels
from openmapflow.utils import try_txt_read

temp_dir = tempfile.gettempdir()

missing_data = try_txt_read(PROJECT_ROOT / dp.MISSING)
unexported = try_txt_read(PROJECT_ROOT / dp.UNEXPORTED)
duplicates_data = try_txt_read(PROJECT_ROOT / dp.DUPLICATES)


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
        if bbox.contains(lat, lon) and f"dates={start_date}_{end_date}" in p.stem:
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
            path_to_bbox,
            labels[LAT],
            labels[LON],
            labels[START],
            labels[END],
            pbar,
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


def create_pickled_labeled_dataset(labels: pd.DataFrame):
    tif_bucket = storage.Client().bucket(BucketNames.LABELED_TIFS)
    for label in tqdm(
        labels.to_dict(orient="records"), desc="Creating pickled instances"
    ):
        (labelled_array, tif_lon, tif_lat, tif_file) = find_matching_point(
            start=label[START],
            tif_paths=label[TIF_PATHS],
            label_lon=label[LON],
            label_lat=label[LAT],
            tif_bucket=tif_bucket,
        )

        if labelled_array is None:
            missing_data_file = PROJECT_ROOT / dp.MISSING
            if not missing_data_file.exists():
                missing_data_file.touch()

            with open(missing_data_file, "a") as f:
                f.write("\n" + label[FEATURE_FILENAME])
            continue

        create_feature(label[FEATURE_PATH], labelled_array, tif_lon, tif_lat, tif_file)


def get_label_timesteps(labels):
    diff = pd.to_datetime(labels[END]) - pd.to_datetime(labels[START])
    return (diff / np.timedelta64(1, "M")).round().astype(int)


@dataclass
class LabeledDataset:
    """
    A labeled dataset represents a DataFrame where each row consists of:
    - A coordinate
    - A binary label for that coordinate (y)
    - A path to earth observation data for that coordinate (X)
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
        self.labels_path = PROJECT_ROOT / dp.PROCESSED_LABELS / (self.dataset + ".csv")
        self._cached_labels_csv = None

    def summary(self, df=None, unexported_check=True):
        if df is None:
            df = self.load_labels(
                allow_processing=False, fail_if_missing_features=False
            )
        text = f"{self.dataset} "
        timesteps = get_label_timesteps(df).unique()
        text += f"(Timesteps: {','.join([str(int(t)) for t in timesteps])})\n"
        text += "----------------------------------------------------------------------------\n"
        train_val_test_counts = df[SUBSET].value_counts()
        newly_unexported = []
        for subset, labels_in_subset in train_val_test_counts.items():
            features_in_subset = df[df[SUBSET] == subset][ALREADY_EXISTS].sum()
            if labels_in_subset != features_in_subset:
                text += (
                    f"\u2716 {subset}: {labels_in_subset} labels, "
                    + f"but {features_in_subset} features\n"
                )
                if not unexported_check:
                    continue
                labels_with_no_feature = df[
                    (df[SUBSET] == subset) & ~df[ALREADY_EXISTS]
                ]
                newly_unexported += labels_with_no_feature[FEATURE_FILENAME].tolist()

            else:
                positive_class_percentage = (
                    df[df[SUBSET] == subset][CLASS_PROB] > 0.5
                ).sum() / labels_in_subset
                text += (
                    f"\u2714 {subset} amount: {labels_in_subset}, "
                    + f"positive class: {positive_class_percentage:.1%}\n"
                )
        if not unexported_check or len(newly_unexported) == 0:
            return text

        add_to_file = input(
            f"Found {len(newly_unexported)} missing features. "
            + "These may have failed on EarthEngine. Add to unexported list? (y/[n]): "
        )
        if add_to_file.lower() == "y":
            with (PROJECT_ROOT / dp.UNEXPORTED).open("w") as f:
                f.write("\n".join(unexported + newly_unexported))

        return text

    def create_processed_labels(self):
        """
        Creates a single processed labels file from a list of raw labels.
        """
        df = pd.DataFrame({})
        already_processed = []
        if self.labels_path.exists():
            df = pd.read_csv(self.labels_path)
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
        df = df.groupby([LON, LAT, START, END], as_index=False, sort=False).agg(
            {
                SOURCE: lambda sources: ",".join(sources.unique()),
                CLASS_PROB: "mean",
                NUM_LABELERS: "sum",
                SUBSET: "first",
                LABEL_DUR: lambda dur: ",".join(dur),
                LABELER_NAMES: lambda name: ",".join(name),
            }
        )
        df[COUNTRY] = self.country
        df[DATASET] = self.dataset
        df[FEATURE_FILENAME] = (
            "lat="
            + df[LAT].round(8).astype(str)
            + "_lon="
            + df[LON].round(8).astype(str)
            + "_date="
            + df[START].astype(str)
            + "_"
            + df[END].astype(str)
        )

        df = df.reset_index(drop=True)
        df.to_csv(self.labels_path, index=False)
        return df

    def load_labels(
        self,
        allow_processing: bool = False,
        fail_if_missing_features: bool = False,
    ) -> pd.DataFrame:
        if allow_processing:
            labels = self.create_processed_labels()
            self._cached_labels_csv = labels
        elif self._cached_labels_csv is not None:
            labels = self._cached_labels_csv
        elif self.labels_path.exists():
            labels = pd.read_csv(self.labels_path)
            self._cached_labels_csv = labels
        else:
            raise FileNotFoundError(f"{self.labels_path} does not exist")
        labels = labels[labels[CLASS_PROB] != 0.5]
        unexported_labels = labels[FEATURE_FILENAME].isin(unexported)
        missing_data_labels = labels[FEATURE_FILENAME].isin(missing_data)
        duplicate_labels = labels[FEATURE_FILENAME].isin(duplicates_data)
        labels = labels[
            ~unexported_labels & ~missing_data_labels & ~duplicate_labels
        ].copy()
        labels["feature_dir"] = str(dp.FEATURES)
        labels[FEATURE_PATH] = (
            labels["feature_dir"] + "/" + labels[FEATURE_FILENAME] + ".pkl"
        )
        labels[ALREADY_EXISTS] = np.vectorize(lambda p: Path(p).exists())(
            labels[FEATURE_PATH]
        )
        if fail_if_missing_features and not labels[ALREADY_EXISTS].all():
            raise FileNotFoundError(
                f"{self.dataset} has missing features: {labels[FEATURE_FILENAME].to_list()}"
            )
        return labels

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
        labels = self.load_labels(allow_processing=True)

        # -------------------------------------------------
        # STEP 2: Check if features already exist
        # -------------------------------------------------
        labels_with_no_features = labels[~labels[ALREADY_EXISTS]].copy()
        if len(labels_with_no_features) == 0:
            return self.summary(df=labels)

        # -------------------------------------------------
        # STEP 3: Match labels to tif files (X)
        # -------------------------------------------------
        labels_with_no_features[TIF_PATHS] = match_labels_to_tifs(
            labels_with_no_features
        )
        tifs_found = labels_with_no_features[TIF_PATHS].str.len() > 0

        labels_with_no_tifs = labels_with_no_features.loc[~tifs_found].copy()
        labels_with_tifs_but_no_features = labels_with_no_features.loc[tifs_found]

        # -------------------------------------------------
        # STEP 4: If no matching tif, download it
        # -------------------------------------------------
        if len(labels_with_no_tifs) > 0:
            print(f"{len(labels_with_no_tifs )} labels not matched")
            if not disable_gee_export:
                labels_with_no_tifs[START] = pd.to_datetime(
                    labels_with_no_tifs[START]
                ).dt.date
                labels_with_no_tifs[END] = pd.to_datetime(
                    labels_with_no_tifs[END]
                ).dt.date
                EarthEngineExporter(
                    check_ee=True,
                    check_gcp=True,
                    dest_bucket=BucketNames.LABELED_TIFS,
                ).export_for_labels(labels=labels_with_no_tifs)

        # -------------------------------------------------
        # STEP 5: Create the features (X, y)
        # -------------------------------------------------
        if len(labels_with_tifs_but_no_features) > 0:
            create_pickled_labeled_dataset(labels=labels_with_tifs_but_no_features)
            labels[ALREADY_EXISTS] = np.vectorize(lambda p: Path(p).exists())(
                labels[FEATURE_PATH]
            )
        return self.summary(df=labels)
