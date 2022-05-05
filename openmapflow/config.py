import yaml
from pathlib import Path
from .utils import find_project_root

# -------------- Load configuration -------------------------------------------
PROJECT_ROOT = find_project_root(["openmapflow.yaml", "data"])

with (PROJECT_ROOT / "openmapflow.yaml").open() as f:
    CONFIG_YML = yaml.safe_load(f)

PROJECT = CONFIG_YML["project"]["id"]

# -------------- PATHS --------------------------------------------------------
RELATIVE_PATHS = {k: f"data/{v}" for k, v in CONFIG_YML["data"]["paths"].items()}
FULL_PATHS = {k: PROJECT_ROOT / v for k, v in RELATIVE_PATHS.items()}
OPENMAPFLOW_DIR = Path(__file__).parent


# -------------- GCLOUD -------------------------------------------------------
GCLOUD_PROJECT_ID = CONFIG_YML["gcloud"]["project_id"]
GCLOUD_LOCATION = CONFIG_YML["gcloud"]["location"]
TIF_BUCKET_NAME = CONFIG_YML["gcloud"]["buckets"]["labeled_tifs"]
DOCKER_TAG = f"{GCLOUD_LOCATION}-docker.pkg.dev/{GCLOUD_PROJECT_ID}/{PROJECT}/{PROJECT}"


# -------------- Helper functions ---------------------------------------------
def get_model_names_as_str():
    return " ".join([p.stem for p in Path(FULL_PATHS["models"]).glob("*.pt")])


def deploy_env_variables():
    env_variables = f"PROJECT={PROJECT} "
    env_variables += f"MODELS_DIR={RELATIVE_PATHS['models']} "
    env_variables += f"OPENMAPFLOW_DIR={OPENMAPFLOW_DIR} "
    env_variables += f"GCLOUD_PROJECT_ID={GCLOUD_PROJECT_ID} "
    env_variables += f"GCLOUD_LOCATION={GCLOUD_LOCATION} "
    env_variables += f"TAG={DOCKER_TAG} "
    return env_variables


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
