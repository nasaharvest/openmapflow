from collections.abc import Mapping
from pathlib import Path
from typing import Dict

import yaml

from openmapflow.constants import (
    CONFIG_FILE,
    DATA_DIR,
    DEFAULT_CONFIG_PATH,
    LIBRARY_DIR,
    VERSION,
)


def update_dict(d: Dict, u: Mapping) -> Dict:
    """
    Update a dictionary with another dictionary.
    Source: https://stackoverflow.com/a/3233356/8702341
    """
    for k, v in u.items():
        if isinstance(v, Mapping):
            d[k] = update_dict(d.get(k, {}), v)
        else:
            d[k] = v
    return d


def load_custom_config(path: Path) -> Dict:
    if path.exists():
        with path.open() as f:
            return yaml.safe_load(f)
    print(f"{path.name} not found in: {path.parent}\n")
    print(f"Using folder as project name: {path.parent.name}")
    return {"project": path.parent.name}


def load_default_config(project_name: str) -> Dict:
    with DEFAULT_CONFIG_PATH.open() as f:
        content = f.read().replace("<PROJECT>", project_name)
        return yaml.safe_load(content)


cwd = Path.cwd()
PROJECT_ROOT: Path = cwd.parent if (cwd.parent / CONFIG_FILE).exists() else cwd
CUSTOM_CONFIG = load_custom_config(PROJECT_ROOT / CONFIG_FILE)
PROJECT = CUSTOM_CONFIG["project"]
DEFAULT_CONFIG = load_default_config(PROJECT)
CONFIG_YML = update_dict(DEFAULT_CONFIG, CUSTOM_CONFIG)
GCLOUD_PROJECT_ID = CONFIG_YML["gcloud"]["project_id"]
GCLOUD_LOCATION = CONFIG_YML["gcloud"]["location"]
DOCKER_TAG = f"{GCLOUD_LOCATION}-docker.pkg.dev/{GCLOUD_PROJECT_ID}/{PROJECT}/{PROJECT}"


class DataPaths:
    RAW_LABELS = DATA_DIR + CONFIG_YML["data_paths"]["raw_labels"]
    DATASETS = DATA_DIR + CONFIG_YML["data_paths"]["datasets"]
    MODELS = DATA_DIR + CONFIG_YML["data_paths"]["models"]
    METRICS = DATA_DIR + CONFIG_YML["data_paths"]["metrics"]
    REPORT = DATA_DIR + CONFIG_YML["data_paths"]["report"]

    @classmethod
    def get(cls, key: str = "") -> str:
        if key in cls.__dict__:
            return cls.__dict__[key]
        dp_list = [
            f"{k}: {v}"
            for k, v in vars(cls).items()
            if not k.startswith("__") and k != "get"
        ]
        return "\n".join(dp_list)


class BucketNames:
    LABELED_EO = CONFIG_YML["gcloud"]["bucket_labeled_eo"]
    INFERENCE_EO = CONFIG_YML["gcloud"]["bucket_inference_eo"]
    PREDS = CONFIG_YML["gcloud"]["bucket_preds"]
    PREDS_MERGED = CONFIG_YML["gcloud"]["bucket_preds_merged"]


def get_model_names_as_str() -> str:
    """Get the names of all models as a string."""
    models = [Path(p).stem for p in Path(PROJECT_ROOT / DataPaths.MODELS).glob("*.pt")]
    return " ".join(models)


def deploy_env_variables(empty_check: bool = True) -> str:
    prefix = "OPENMAPFLOW"
    deploy_env_dict = {
        "PROJECT": PROJECT,
        "MODELS_DIR": DataPaths.MODELS,
        "LIBRARY_DIR": LIBRARY_DIR,
        "GCLOUD_PROJECT_ID": GCLOUD_PROJECT_ID,
        "GCLOUD_LOCATION": GCLOUD_LOCATION,
        "GCLOUD_BUCKET_LABELED_EO": BucketNames.LABELED_EO,
        "GCLOUD_BUCKET_INFERENCE_EO": BucketNames.INFERENCE_EO,
        "GCLOUD_BUCKET_PREDS": BucketNames.PREDS,
        "GCLOUD_BUCKET_PREDS_MERGED": BucketNames.PREDS_MERGED,
        "DOCKER_TAG": DOCKER_TAG,
        "VERSION": VERSION,
    }
    if empty_check:
        for k, v in deploy_env_dict.items():
            if v == "" or v is None:
                raise ValueError(f"{k} is not set")

    env_variables = " ".join([f"{prefix}_{k}={v}" for k, v in deploy_env_dict.items()])
    return env_variables
