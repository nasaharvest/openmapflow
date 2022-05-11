import collections.abc
import yaml
from pathlib import Path

from .constants import CONFIG_FILE, LIBRARY_DIR


def update(d, u):
    for k, v in u.items():
        if isinstance(v, collections.abc.Mapping):
            d[k] = update(d.get(k, {}), v)
        else:
            d[k] = v
    return d


def load_custom_config(path: Path) -> dict:
    if path.exists():
        with path.open() as f:
            return yaml.safe_load(f)
    print(f"{path.name} not found in: {path.parent}\n")
    print(f"Using folder as project name: {path.parent.name}")
    return {"project": path.parent.name}


def load_default_config(project_name: str) -> dict:
    with (LIBRARY_DIR / "templates/openmapflow-default.yaml").open() as f:
        content = f.read().replace("<PROJECT>", project_name)
        return yaml.safe_load(content)


cwd = Path.cwd()
PROJECT_ROOT = cwd.parent if (cwd.parent / CONFIG_FILE).exists() else cwd
CUSTOM_CONFIG = load_custom_config(PROJECT_ROOT / CONFIG_FILE)
PROJECT = CUSTOM_CONFIG["project"]
DEFAULT_CONFIG = load_default_config(PROJECT)
CONFIG_YML = update(DEFAULT_CONFIG, CUSTOM_CONFIG)
GCLOUD_PROJECT_ID = CONFIG_YML["gcloud"]["project_id"]
GCLOUD_LOCATION = CONFIG_YML["gcloud"]["location"]
DOCKER_TAG = f"{GCLOUD_LOCATION}-docker.pkg.dev/{GCLOUD_PROJECT_ID}/{PROJECT}/{PROJECT}"


class DataPaths:
    RAW_LABELS = "data/" + CONFIG_YML["data_paths"]["raw_labels"]
    PROCESSED_LABELS = "data/" + CONFIG_YML["data_paths"]["processed_labels"]
    FEATURES = "data/" + CONFIG_YML["data_paths"]["features"]
    COMPRESSED_FEATURES = "data/" + CONFIG_YML["data_paths"]["compressed_features"]
    MODELS = "data/" + CONFIG_YML["data_paths"]["models"]
    METRICS = "data/" + CONFIG_YML["data_paths"]["metrics"]
    DATASETS = "data/" + CONFIG_YML["data_paths"]["datasets"]
    MISSING = "data/" + CONFIG_YML["data_paths"]["missing"]
    DUPLICATES = "data/" + CONFIG_YML["data_paths"]["duplicates"]
    UNEXPORTED = "data/" + CONFIG_YML["data_paths"]["unexported"]

    @classmethod
    def get(cls, key: str = "") -> str:
        if key in cls.__dict__:
            return cls.__dict__[key]
        dp_list = [
            f"{k}: {v}"
            for k, v in vars(cls).items()
            if not k.startswith("__") and k == "get"
        ]
        return "\n".join(dp_list)


class BucketNames:
    LABELED_TIFS = CONFIG_YML["gcloud"]["bucket_labeled_tifs"]
    INFERENCE_TIFS = CONFIG_YML["gcloud"]["bucket_inference_tifs"]
    PREDS = CONFIG_YML["gcloud"]["bucket_preds"]
    PREDS_MERGED = CONFIG_YML["gcloud"]["bucket_preds_merged"]


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
