import argparse
import shutil
import tarfile

from pathlib import Path
from openmapflow.constants import CONFIG_FILE

parser = argparse.ArgumentParser(description="Generate OpenMapFlow project.")
parser.add_argument("--name", type=str, help="Project name", default=Path.cwd().name)
parser.add_argument("--description", type=str, help="Project description", default="")
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


def allow_write(p):
    if args.force or not Path(p).exists():
        return True
    overwrite = input(f"{str(p)} already exists. Overwrite? (y/n): ")
    return overwrite.lower() == "y"


print("1/5 Parsing arguments")
openmapflow_str = f"""version: 0.0.1
project: {args.name}
description: {args.description}
gcloud: 
    project_id: {args.gcloud_project_id}
    location: {args.gcloud_location}
    bucket_labeled_tifs: {args.gcloud_bucket_labeled_tifs}
"""
print(f"2/5 Writing {CONFIG_FILE}")
if allow_write(CONFIG_FILE):
    with open(CONFIG_FILE, "w") as f:
        f.write(openmapflow_str)

from openmapflow.config import LIBRARY_DIR, PROJECT_ROOT, DataPaths as dp

print("3/5 Copying over template files")
for template_file in ["datasets.py", "integration_test_data.py"]:
    if allow_write(template_file):
        shutil.copy(str(LIBRARY_DIR / f"template/{template_file}"), str(PROJECT_ROOT))

print("4/5 Creating data directories")
for p in [dp.FEATURES, dp.RAW_LABELS, dp.PROCESSED_LABELS, dp.FEATURES, dp.MODELS]:
    if allow_write(p):
        Path(p).mkdir(parents=True, exist_ok=True)

if allow_write(dp.COMPRESSED_FEATURES):
    with tarfile.open(dp.COMPRESSED_FEATURES, "w:gz") as tar:
        tar.add(dp.FEATURES, arcname=Path(dp.FEATURES).name)

print("5/5 Printing dvc instructions")
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

# Connect to remmote storage (we recommend gdrive): https://dvc.org/doc/user-guide/setup-google-drive-remote
dvc remote add -d gdrive <your google drive folder>

# Push files to remote storage
dvc push
"""
)
