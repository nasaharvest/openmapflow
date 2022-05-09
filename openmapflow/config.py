import yaml
from pathlib import Path

config_file = "openmapflow.yaml"

possible_roots = [Path.cwd(), Path.cwd().parent]
try:
    root = next(r for r in possible_roots if (r / config_file).exists())
except StopIteration:
    raise FileExistsError(
        f"Could not find {config_file} in {[str(p) for p in possible_roots]} "
        + f"please cd to a directory with {config_file} or create a new one."
    )

with (root / config_file).open() as f:
    CONFIG_YML = yaml.safe_load(f)

PROJECT = CONFIG_YML["project"]

default_data_names = {
    "raw": "raw_labels",
    "processed": "processed_labels",
    "features": "features",
    "compressed_features": "compressed_features.tar.gz",
    "models": "models",
    "metrics": "metrics.yaml",
    "datasets": "datasets.txt",
    "missing": "missing.txt",
    "duplicates": "duplicates.txt",
    "unexported": "unexported.txt",
}
names_from_config = CONFIG_YML.get("data_paths", {})
data_names = {k: names_from_config.get(k, v) for k, v in default_data_names.items()}
RELATIVE_PATHS = {k: f"data/{v}" for k, v in data_names.items()}
FULL_PATHS = {k: root / v for k, v in RELATIVE_PATHS.items()}
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
