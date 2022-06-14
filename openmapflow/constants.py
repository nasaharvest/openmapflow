from pathlib import Path

CONFIG_FILE = "openmapflow.yaml"
DATA_DIR = "data/"
LIBRARY_DIR = Path(__file__).parent
TEMPLATES_DIR = LIBRARY_DIR / "templates"
DEFAULT_CONFIG_PATH = TEMPLATES_DIR / "openmapflow-default.yaml"

TEMPLATE_DATASETS = TEMPLATES_DIR / "datasets.py"
TEMPLATE_TRAIN = TEMPLATES_DIR / "train.py"
TEMPLATE_EVALUATE = TEMPLATES_DIR / "evaluate.py"
TEMPLATE_DEPLOY_YML = TEMPLATES_DIR / "github-deploy.yaml"
TEMPLATE_TEST_YML = TEMPLATES_DIR / "github-test.yaml"
VERSION = "0.0.1"

# -------------- Dataframe column names ---------------------------------------
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

# -------------- Months -------------------------------------------------------
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
