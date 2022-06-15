import os
import tempfile
from pathlib import Path
from unittest import TestCase

from openmapflow.constants import VERSION


class TestCLI(TestCase):
    """
    openmapflow must be installed for these tests to run
    """

    def test_cp(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            os.chdir(tmpdir)
            self.assertFalse(Path("Dockerfile").exists())
            os.system("openmapflow cp Dockerfile .")
            self.assertTrue(Path("Dockerfile").exists())

    def test_dir(self):
        output = os.popen("openmapflow dir").read().rstrip()
        self.assertTrue(output.endswith("openmapflow"))

    def test_ls(self):
        output = os.popen("openmapflow ls").read().rstrip()
        self.assertIn("Dockerfile", output)
        self.assertIn("__init__.py", output)
        self.assertIn("config.py", output)
        self.assertIn("constants.py", output)
        self.assertIn("data_instance.py", output)
        self.assertIn("features.py", output)
        self.assertIn("generate.py", output)
        self.assertIn("inference_utils.py", output)
        self.assertIn("inference_widgets.py", output)
        self.assertIn("labeled_dataset.py", output)
        self.assertIn("notebooks", output)
        self.assertIn("pytorch_dataset.py", output)
        self.assertIn("raw_labels.py", output)
        self.assertIn("scripts", output)
        self.assertIn("templates", output)
        self.assertIn("train_utils.py", output)
        self.assertIn("trigger_inference_function", output)
        self.assertIn("utils.py", output)

    def test_version(self):
        self.assertEqual(os.popen("openmapflow version").read().rstrip(), VERSION)
        self.assertEqual(os.popen("openmapflow --version").read().rstrip(), VERSION)

    def test_help(self):
        actual_output = os.popen("openmapflow help").read().rstrip()
        expected_output = """---------------------------------------------------------------------------------
                              OpenMapFlow CLI
---------------------------------------------------------------------------------
openmapflow cp <source> <destination> - copy a file or directory from the library
openmapflow create-features - creates features for all datasets in datasets.py
openmapflow datapath <DATAPATH> - outputs a relative path to the data directory
openmapflow datasets - outputs a list of all datasets
openmapflow deploy - deploys Google Cloud Architecture for project
openmapflow dir - outputs openmapflow library directory
openmapflow generate - generates an openmapflow project
openmapflow help - outputs this message
openmapflow ls - lists files in openmapflow library directory
openmapflow version - package version"""
        self.assertEqual(actual_output, expected_output)
