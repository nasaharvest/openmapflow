from pathlib import Path

CONFIG_FILE = "openmapflow.yaml"
DATA_DIR = "data/"
LIBRARY_DIR = Path(__file__).parent
TEMPLATES_DIR = LIBRARY_DIR / "templates"
DEFAULT_CONFIG_PATH = TEMPLATES_DIR / "openmapflow-default.yaml"

TEMPLATE_DATASETS = TEMPLATES_DIR / "datasets.py"
TEMPLATE_TRAIN = TEMPLATES_DIR / "train.py"
TEMPLATE_EVALUATE = TEMPLATES_DIR / "evaluate.py"
TEMPLATE_REQUIREMENTS = TEMPLATES_DIR / "requirements.txt"
TEMPLATE_DEPLOY_YML = TEMPLATES_DIR / "github-deploy.yaml"
TEMPLATE_TEST_YML = TEMPLATES_DIR / "github-test.yaml"
VERSION = "0.1.0"

# -------------- Dataframe column names --------------------------------------
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
TIF_PATHS = "tif_paths"
LABELER_NAMES = "email"
LABEL_DUR = "analysis_duration"
EO_DATA = "eo_data"
EO_LAT = "eo_lat"
EO_LON = "eo_lon"
EO_FILE = "eo_file"
EO_STATUS = "eo_status"

# -------------- EO data statuses ---------------------------------------------
EO_STATUS_WAITING = "waiting_for_eo_data"
EO_STATUS_DUPLICATE = "duplicate_eo_data"
EO_STATUS_EXPORTING = "exporting_eo_data"
EO_STATUS_EXPORT_FAILED = "eo_data_export_failed"
EO_STATUS_MISSING_VALUES = "eo_data_missing_values"
EO_STATUS_COMPLETE = "eo_data_complete"

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
