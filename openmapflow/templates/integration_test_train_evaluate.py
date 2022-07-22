import os
import subprocess
import unittest
from random import randrange
from unittest import TestCase

from datasets import datasets

from openmapflow.config import PROJECT_ROOT, DataPaths
from openmapflow.constants import TEMPLATE_EVALUATE, TEMPLATE_TRAIN


class TestExampleProjectsGenerated(TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        if len(datasets) == 0:
            print("WARNING: No datasets found. Skipping all tests.")
            raise unittest.SkipTest("No datasets found. Skipping all tests.")

    def test_train_and_evaluate(self):
        """Runs the train and evaluate scripts for the given project."""

        if not (PROJECT_ROOT / TEMPLATE_TRAIN).exists():
            raise unittest.SkipTest("train.py script not found.")
        if not (PROJECT_ROOT / TEMPLATE_EVALUATE).exists():
            raise unittest.SkipTest("evaluate.py script not found.")

        models_before = set((PROJECT_ROOT / DataPaths.MODELS).glob("*.pt"))

        test_model_name = f"test_model_{randrange(0, 100)}"
        subprocess.check_output(
            ["python", "train.py", "--epochs", "2", "--model_name", test_model_name]
        )
        print("\u2714 train.py ran successfully")

        models_after = set((PROJECT_ROOT / DataPaths.MODELS).glob("*.pt"))
        model_difference = list(models_after - models_before)
        self.assertEqual(len(model_difference), 1)
        new_model_path = model_difference[0]
        self.assertEqual(new_model_path.stem, test_model_name)
        print("\u2714 train.py generated a new model")

        subprocess.check_output(
            ["python", "evaluate.py", "--skip_yaml", "--model_name", test_model_name]
        )
        print("\u2714 evaluate.py ran successfully on new model")

        new_model_path.unlink()

    def test_evaluate_existing_models(self):
        """Checks that existing models can be evaluated."""
        if not (PROJECT_ROOT / TEMPLATE_EVALUATE).exists():
            raise unittest.SkipTest("evaluate.py script not found.")

        model_paths = list((PROJECT_ROOT / DataPaths.MODELS).glob("*.pt"))
        if len(model_paths) == 0:
            raise unittest.SkipTest(f"No models found in {DataPaths.MODELS}")

        for model_path in model_paths:
            evaluate_output = subprocess.check_output(
                [
                    "python",
                    "evaluate.py",
                    "--skip_yaml",
                    "--model_name",
                    model_path.stem,
                ]
            ).decode("utf-8")
            print(f"\u2714 evaluate.py ran successfully on {model_path.stem}")

            for metric in ["accuracy", "precision", "recall", "f1", "roc_au"]:
                self.assertIn(metric, evaluate_output)


if __name__ == "__main__":
    runner = unittest.TextTestRunner(stream=open(os.devnull, "w"), verbosity=2)
    unittest.main(testRunner=runner)
