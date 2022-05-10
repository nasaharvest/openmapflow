import yaml
from pathlib import Path

from .constants import CONFIG_FILE

possible_roots = [Path.cwd(), Path.cwd().parent]
try:
    PROJECT_ROOT = next(r for r in possible_roots if (r / CONFIG_FILE).exists())
except StopIteration:
    raise FileExistsError(
        f"Could not find {CONFIG_FILE} in {[str(p) for p in possible_roots]} "
        + f"please cd to a directory with {CONFIG_FILE} or create a new one."
    )

with (PROJECT_ROOT / CONFIG_FILE).open() as f:
    CONFIG_YML = yaml.safe_load(f)

_data_paths = CONFIG_YML.get("data_paths", {})
_gcloud = CONFIG_YML.get("gcloud", {})

PROJECT = CONFIG_YML["project"]
LIBRARY_DIR = Path(__file__).parent
GCLOUD_PROJECT_ID = _gcloud.get("project_id", "")
GCLOUD_LOCATION = _gcloud.get("location", "")
DOCKER_TAG = f"{GCLOUD_LOCATION}-docker.pkg.dev/{GCLOUD_PROJECT_ID}/{PROJECT}/{PROJECT}"


def _gen_path(k, default):
    return f"data/{_data_paths.get(k, default)}"


class DataPaths:
    RAW_LABELS = _gen_path("raw_labels", "raw_labels")
    PROCESSED_LABELS = _gen_path("processed_labels", "processed_labels")
    FEATURES = _gen_path("features", "features")
    COMPRESSED_FEATURES = _gen_path("compressed_features", "compressed_features.tar.gz")
    MODELS = _gen_path("models", "models")
    METRICS = _gen_path("metrics", "metrics.yaml")
    DATASETS = _gen_path("datasets", "datasets.txt")
    MISSING = _gen_path("missing", "missing.txt")
    DUPLICATES = _gen_path("duplicates", "duplicates.txt")
    UNEXPORTED = _gen_path("unexported", "unexported.txt")


class BucketNames:
    LABELED_TIFS = _gcloud.get("bucket_labeled_tifs", f"{PROJECT}-labeled-tifs")
    INFERENCE_TIFS = _gcloud.get("bucket_inference_tifs", f"{PROJECT}-inference-tifs")
    PREDS = _gcloud.get("bucket_preds", f"{PROJECT}-preds")
    PREDS_MERGED = _gcloud.get("bucket_preds_merged", f"{PROJECT}-preds-merged")


# -------------- Helper functions ---------------------------------------------
def get_model_names_as_str() -> str:
    return " ".join(
        [p.stem for p in Path(PROJECT_ROOT / DataPaths.MODELS).glob("*.pt")]
    )


def deploy_env_variables() -> str:
    prefix = "OPENMAPFLOW"
    deploy_env_dict = {
        "PROJECT": PROJECT,
        "MODELS_DIR": DataPaths.MODELS,
        "LIBRARY_DIR": LIBRARY_DIR,
        "GCLOUD_PROJECT_ID": GCLOUD_PROJECT_ID,
        "GCLOUD_LOCATION": GCLOUD_LOCATION,
        "GCLOUD_BUCKET_LABELED_TIFS": BucketNames.LABELED_TIFS,
        "GCLOUD_BUCKET_INFERENCE_TIFS": BucketNames.INFERENCE_TIFS,
        "GCLOUD_BUCKET_PREDS": BucketNames.PREDS,
        "GCLOUD_BUCKET_PREDS_MERGED": BucketNames.PREDS_MERGED,
        "DOCKER_TAG": DOCKER_TAG,
    }
    env_variables = " ".join([f"{prefix}_{k}={v}" for k, v in deploy_env_dict.items()])
    return env_variables
