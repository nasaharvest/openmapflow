import yaml
from .utils import find_project_root

PROJECT_ROOT = find_project_root(["openmapflow.yaml", "data"])

with (PROJECT_ROOT / "openmapflow.yaml").open() as f:
    CONFIG_YML = yaml.safe_load(f)

RELATIVE_PATHS = {k: f"data/{v}" for k, v in CONFIG_YML["data_paths"].items()}
FULL_PATHS = {k: PROJECT_ROOT / v for k, v in RELATIVE_PATHS.items()}
TIF_BUCKET_NAME = CONFIG_YML["labeled_tifs_bucket"]

# df column names
SOURCE = "source"
CLASS_PROB = "class_probability"
START = "start_date"
END = "end_date"
LON = "lon"
LAT = "lat"
COUNTRY = "country"
NUM_LABELERS = "num_labelers"
SUBSET = "subset"
DATASET = "dataset"
ALREADY_EXISTS = "already_exists"
FEATURE_FILENAME = "filename"
FEATURE_PATH = "save_path"
TIF_PATHS = "tif_paths"
LABELER_NAMES = "email"
LABEL_DUR = "analysis_duration"

MONTHS = [
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December",
]
