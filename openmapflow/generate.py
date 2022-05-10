import argparse
import shutil
import tarfile

from pathlib import Path
from openmapflow.constants import CONFIG_FILE


def allow_write(p, force=False):
    if force or not Path(p).exists():
        return True
    overwrite = input(f"{str(p)} already exists. Overwrite? (y/n): ")
    return overwrite.lower() == "y"


def openmapflow_config_from_args(args):
    return f"""version: 0.0.1
project: {args.name}
description: {args.description}
gcloud:
    project_id: {args.gcloud_project_id}
    location: {args.gcloud_location}
    bucket_labeled_tifs: {args.gcloud_bucket_labeled_tifs}
"""


def openmapflow_config_write(openmapflow_str: str, force: bool):
    if allow_write(CONFIG_FILE, force):
        with open(CONFIG_FILE, "w") as f:
            f.write(openmapflow_str)


def copy_datasets_py_file(LIBRARY_DIR, PROJECT_ROOT, force: bool):
    if allow_write("datasets.py", force):
        shutil.copy(
            str(LIBRARY_DIR / "example_datasets.py"), str(PROJECT_ROOT / "datasets.py")
        )


def create_data_dirs(dp, force):
    for p in [dp.FEATURES, dp.RAW_LABELS, dp.PROCESSED_LABELS, dp.FEATURES, dp.MODELS]:
        if allow_write(p, force):
            Path(p).mkdir(parents=True, exist_ok=True)

    if allow_write(dp.COMPRESSED_FEATURES):
        with tarfile.open(dp.COMPRESSED_FEATURES, "w:gz") as tar:
            tar.add(dp.FEATURES, arcname=Path(dp.FEATURES).name)


def dvc_instructions(dp):
    dvc_files = " ".join(
        [dp.RAW_LABELS, dp.PROCESSED_LABELS, dp.COMPRESSED_FEATURES, dp.MODELS]
    )
    print(
        f"""
    #########################################################################################
    DVC Setup Instructions
    #########################################################################################
    dvc (https://dvc.org/) is used to manage data. To setup run:

    # Initializes dvc (use --subdir if in subdirectory)
    dvc init

    # Tells dvc to track directories
    dvc add {dvc_files}

    # Connect remmote (we suggest gdrive): https://dvc.org/doc/user-guide/setup-google-drive-remote
    dvc remote add -d gdrive <your google drive folder>

    # Push files to remote storage
    dvc push
    """
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate OpenMapFlow project.")
    parser.add_argument(
        "--name", type=str, help="Project name", default=Path.cwd().name
    )
    parser.add_argument(
        "--description", type=str, help="Project description", default=""
    )
    parser.add_argument(
        "--gcloud_project_id",
        type=str,
        help="Google Cloud Project ID",
        default="",
    )
    parser.add_argument(
        "--gcloud_location",
        type=str,
        help="Google Cloud Location (default: us-central-1)",
        default="us-central-1",
    )
    parser.add_argument(
        "--gcloud_bucket_labeled_tifs",
        type=str,
        help="Google Cloud Bucket for labeled tifs (default: crop-mask-tifs2)",
        default="crop-mask-tifs2",
    )
    parser.add_argument("--force", action="store_true", help="Force overwrite")
    args = parser.parse_args()

    print("1/5 Parsing arguments")
    openmapflow_str = openmapflow_config_from_args(args)

    print(f"2/5 Writing {CONFIG_FILE}")
    openmapflow_config_write(openmapflow_str, force=args.force)

    # Can only import when openmapflow.yaml is available
    from openmapflow.config import LIBRARY_DIR, PROJECT_ROOT, DataPaths  # noqa E402

    print("3/5 Copying datasets.py file")
    copy_datasets_py_file(LIBRARY_DIR, PROJECT_ROOT, args.force)

    print("4/5 Creating data directories")
    create_data_dirs(dp=DataPaths, force=args.force)

    print("5/5 Printing dvc instructions")
    dvc_instructions(dp=DataPaths)
