from unittest import TestCase

from openmapflow.config import DataPaths, deploy_env_variables, load_default_config
from openmapflow.constants import LIBRARY_DIR


class TestConfig(TestCase):
    def test_load_default_config(self):
        self.maxDiff = None
        actual_config = load_default_config(project_name="fake-project")
        expected_config = {
            "version": "<VERSION>",
            "project": "fake-project",
            "description": "",
            "data_paths": {
                "datasets": "datasets",
                "raw_labels": "raw_labels",
                "models": "models",
                "metrics": "metrics.yaml",
                "report": "report.txt",
            },
            "gcloud": {
                "project_id": "",
                "location": "us-central1",
                "bucket_labeled_eo": "fake-project-labeled-eo",
                "bucket_inference_eo": "fake-project-inference-eo",
                "bucket_preds": "fake-project-preds",
                "bucket_preds_merged": "fake-project-preds-merged",
            },
        }
        self.assertEqual(actual_config, expected_config)

    def test_get_datapaths(self):
        actual_data_path_str = DataPaths.get()
        expected_data_path_str = (
            "RAW_LABELS: data/raw_labels"
            + "\nDATASETS: data/datasets"
            + "\nMODELS: data/models"
            + "\nMETRICS: data/metrics.yaml"
            + "\nREPORT: data/report.txt"
        )
        self.assertEqual(actual_data_path_str, expected_data_path_str)

    def test_deploy_env_variables(self):
        self.maxDiff = None

        self.assertRaises(ValueError, deploy_env_variables)

        actual_env_variables = deploy_env_variables(empty_check=False)
        expected_env_variables = (
            "OPENMAPFLOW_PROJECT=openmapflow "
            + "OPENMAPFLOW_MODELS_DIR=data/models "
            + f"OPENMAPFLOW_LIBRARY_DIR={LIBRARY_DIR} "
            + "OPENMAPFLOW_GCLOUD_PROJECT_ID= "
            + "OPENMAPFLOW_GCLOUD_LOCATION=us-central1 "
            + "OPENMAPFLOW_GCLOUD_BUCKET_LABELED_EO=openmapflow-labeled-eo "
            + "OPENMAPFLOW_GCLOUD_BUCKET_INFERENCE_EO=openmapflow-inference-eo "
            + "OPENMAPFLOW_GCLOUD_BUCKET_PREDS=openmapflow-preds "
            + "OPENMAPFLOW_GCLOUD_BUCKET_PREDS_MERGED=openmapflow-preds-merged "
            + "OPENMAPFLOW_DOCKER_TAG=us-central1-docker.pkg.dev//openmapflow/openmapflow"
        )
        self.assertEqual(actual_env_variables, expected_env_variables)
