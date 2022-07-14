from unittest import TestCase

from openmapflow.config import DataPaths, deploy_env_variables, load_default_config
from openmapflow.constants import LIBRARY_DIR, VERSION


class TestConfig(TestCase):
    def test_load_default_config(self):
        actual_config = load_default_config(project_name="fake-project")
        expected_config = {
            "version": VERSION,
            "project": "fake-project",
            "description": None,
            "data_paths": {
                "raw_labels": "raw_labels",
                "processed_labels": "processed_labels",
                "features": "features",
                "compressed_features": "compressed_features.tar.gz",
                "models": "models",
                "metrics": "metrics.yaml",
                "datasets": "datasets.txt",
                "missing": "missing.txt",
                "duplicates": "duplicates.txt",
                "unexported": "unexported.txt",
            },
            "gcloud": {
                "project_id": None,
                "location": "us-central1",
                "bucket_labeled_tifs": "fake-project-labeled-tifs",
                "bucket_inference_tifs": "fake-project-inference-tifs",
                "bucket_preds": "fake-project-preds",
                "bucket_preds_merged": "fake-project-preds-merged",
            },
        }
        self.assertEqual(actual_config, expected_config)

    def test_get_datapaths(self):
        actual_data_path_str = DataPaths.get()
        expected_data_path_str = (
            "RAW_LABELS: data/raw_labels"
            + "\nPROCESSED_LABELS: data/processed_labels"
            + "\nFEATURES: data/features"
            + "\nCOMPRESSED_FEATURES: data/compressed_features.tar.gz"
            + "\nMODELS: data/models"
            + "\nMETRICS: data/metrics.yaml"
            + "\nDATASETS: data/datasets.txt"
            + "\nMISSING: data/missing.txt"
            + "\nDUPLICATES: data/duplicates.txt"
            + "\nUNEXPORTED: data/unexported.txt"
        )
        self.assertEqual(actual_data_path_str, expected_data_path_str)

    def test_deploy_env_variables(self):
        actual_env_variables = deploy_env_variables()
        expected_env_variables = (
            "OPENMAPFLOW_PROJECT=openmapflow "
            + "OPENMAPFLOW_MODELS_DIR=data/models "
            + f"OPENMAPFLOW_LIBRARY_DIR={LIBRARY_DIR} "
            + "OPENMAPFLOW_GCLOUD_PROJECT_ID=None "
            + "OPENMAPFLOW_GCLOUD_LOCATION=us-central1 "
            + "OPENMAPFLOW_GCLOUD_BUCKET_LABELED_TIFS=openmapflow-labeled-tifs "
            + "OPENMAPFLOW_GCLOUD_BUCKET_INFERENCE_TIFS=openmapflow-inference-tifs "
            + "OPENMAPFLOW_GCLOUD_BUCKET_PREDS=openmapflow-preds "
            + "OPENMAPFLOW_GCLOUD_BUCKET_PREDS_MERGED=openmapflow-preds-merged "
            + "OPENMAPFLOW_DOCKER_TAG=us-central1-docker.pkg.dev/None/openmapflow/openmapflow"
        )
        self.assertEqual(actual_env_variables, expected_env_variables)
