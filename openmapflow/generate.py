import argparse

parser = argparse.ArgumentParser(description="Generate OpenMapFlow project.")
parser.add_argument("name", type=str, help="Project name")
parser.add_argument("description", type=str, help="Project description", default="")
parser.add_argument(
    "gcloud_project_id", type=str, help="Google Cloud Project ID", default=""
)
parser.add_argument(
    "gcloud_location", type=str, help="Google Cloud Location", default="us-central-1"
)
parser.add_argument(
    "gcloud_bucket_labeled_tifs",
    type=str,
    help="Google Cloud Bucket for labeled tifs",
    default="crop-mask-tifs2",
)


args = parser.parse_args()

openmapflow_dict = {
    "version": "0.0.1",
    "project": args.name,
    "description": args.description,
    "gcloud": {
        "project_id": args.gcloud_project_id,
        "location": args.gcloud_location,
        "bucket_labeled_tifs": args.gcloud_bucket_labeled_tifs,
    },
}


# - take unique project name - set it inside the notebooks
# - bucket creation: remote storage, tifs, earth engine, press, preds-merged
# - container registry creation
# env creation

# If project is directly in git repo
# dvc init

# If project is in subdirectory
# cd <subdir> dvc init --subdir

# Make data directories: [models, features, processed_labels, raw_labels]

# Add dvc data: [models, features, processed_labels, raw_labels]
# dvc add data/raw_labels ...
# dvc commit

# Set dvc remote, Google Drive by default for simplicity
# https://dvc.org/doc/user-guide/setup-google-drive-remote

# dvc remote add -d gdrive \
#   gdrive://1EMHILcNFwdusMHHs4eC4OVIJ0Ncp9fiY/crop-mask-example-dvc-store

# https://dvc.org/doc/user-guide/setup-google-drive-remote#authorization

# from google.cloud import storage  # type: ignore
