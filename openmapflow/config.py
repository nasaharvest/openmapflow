import yaml
from pathlib import Path
from typing import List


def find_project_root(files_to_check: List[str]) -> Path:
    """
    Find the project root directory by checking for existence of certain files
    """
    possible_roots = [Path.cwd(), Path.cwd().parent]

    for root in possible_roots:
        if all([(root / c).exists() for c in files_to_check]):
            return root

    raise FileExistsError(f"{files_to_check} not found in {possible_roots}.")


# -------------- Load configuration -------------------------------------------
PROJECT_ROOT = find_project_root(["openmapflow.yaml"])

with (PROJECT_ROOT / "openmapflow.yaml").open() as f:
    CONFIG_YML = yaml.safe_load(f)

PROJECT = CONFIG_YML["project"]

# -------------- PATHS --------------------------------------------------------
RELATIVE_PATHS = {k: f"data/{v}" for k, v in CONFIG_YML["data"]["paths"].items()}
FULL_PATHS = {k: PROJECT_ROOT / v for k, v in RELATIVE_PATHS.items()}
LIBRARY_DIR = Path(__file__).parent


# -------------- GCLOUD -------------------------------------------------------
GCLOUD_PROJECT_ID = CONFIG_YML["gcloud"]["project_id"]
GCLOUD_LOCATION = CONFIG_YML["gcloud"]["location"]

GCLOUD_BUCKET_LABELED_TIFS = CONFIG_YML["gcloud"]["bucket_labeled_tifs"]
GCLOUD_BUCKET_INFERENCE_TIFS = CONFIG_YML["gcloud"]["bucket_inference_tifs"]
GCLOUD_BUCKET_PREDS = CONFIG_YML["gcloud"]["bucket_preds"]
GCLOUD_BUCKET_PREDS_MERGED = CONFIG_YML["gcloud"]["bucket_preds_merged"]

DOCKER_TAG = f"{GCLOUD_LOCATION}-docker.pkg.dev/{GCLOUD_PROJECT_ID}/{PROJECT}/{PROJECT}"


# -------------- Helper functions ---------------------------------------------
def get_model_names_as_str() -> str:
    return " ".join([p.stem for p in Path(FULL_PATHS["models"]).glob("*.pt")])


def deploy_env_variables() -> str:
    prefix = "OPENMAPFLOW"
    deploy_env_dict = {
        "PROJECT": PROJECT,
        "MODELS_DIR": RELATIVE_PATHS["models"],
        "LIBRARY_DIR": LIBRARY_DIR,
        "GCLOUD_PROJECT_ID": GCLOUD_PROJECT_ID,
        "GCLOUD_LOCATION": GCLOUD_LOCATION,
        "GCLOUD_BUCKET_LABELED_TIFS": GCLOUD_BUCKET_LABELED_TIFS,
        "GCLOUD_BUCKET_INFERENCE_TIFS": GCLOUD_BUCKET_INFERENCE_TIFS,
        "GCLOUD_BUCKET_PREDS": GCLOUD_BUCKET_PREDS,
        "GCLOUD_BUCKET_PREDS_MERGED": GCLOUD_BUCKET_PREDS_MERGED,
        "DOCKER_TAG": DOCKER_TAG,
    }
    env_variables = " ".join([f"{prefix}_{k}={v}" for k, v in deploy_env_dict.items()])
    return env_variables
