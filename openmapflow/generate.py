"""
Generates an openmapflow.json and and sets up the project.
"""

from google.cloud import storage  # type: ignore


gcloud_project_id = input("Gcloud Project ID:")
project_id = input("Project ID:")

labeled_tifs_bucket_name = f"{project_id}-labeled-tifs"
tifs_bucket_name = f"{project_id}-inference-tifs"
preds_bucket_name = f"{project_id}-preds"
preds_merged_bucket_name = f"{project_id}-preds-merged"

# Check if any of them exist, if yes, choose new project id

# Create all buckets

# Create container registry

# Save to cloud run

# - download example labels, features, models
# - dvc init things
