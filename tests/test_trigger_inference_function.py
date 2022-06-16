import os
from unittest import TestCase
from unittest.mock import patch

from openmapflow.trigger_inference_function.main import trigger


class TestTriggerInferenceFunction(TestCase):
    def setUp(self):
        self.model_name = "fake_model"
        self.bucket = "test-bucket"
        self.file_path = f"{self.model_name}/path"
        self.event = {"bucket": self.bucket, "name": self.file_path}
        os.environ["INFERENCE_HOST"] = "http://fake-host"
        os.environ["MODELS"] = f"{self.model_name} fake_model2"

    @patch("openmapflow.trigger_inference_function.main.requests")
    def test_trigger_inference_function(self, mock_requests):
        trigger(self.event, None)
        mock_requests.post.assert_called_once()
        expected_url = f"http://fake-host/predictions/{self.model_name}"
        expected_data = {"uri": f"gs://{self.bucket}/{self.file_path}"}
        mock_requests.post.assert_called_with(expected_url, data=expected_data)

    def test_trigger_inference_function_model_unavailable(self):
        os.environ["MODELS"] = "fake_model2"
        self.assertRaises(ValueError, trigger, self.event, None)

    def test_trigger_inference_function_incomplete_event(self):
        self.assertRaises(ValueError, trigger, {}, None)
