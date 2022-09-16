import os
import shutil
import tempfile
from pathlib import Path
from subprocess import check_output
from unittest import TestCase, skipIf

from openmapflow.constants import VERSION


class TestCLI(TestCase):
    """
    openmapflow must be installed for these tests to run
    """

    @skipIf(os.name == "nt", "Not yet available on Windows")
    def test_cp(self):
        tmpdir = tempfile.mkdtemp()
        p = Path(tmpdir) / "Dockerfile"
        self.assertFalse(p.exists())
        check_output(
            ["openmapflow", "cp", "Dockerfile", f"{tmpdir}"], cwd=tmpdir
        ).decode()
        self.assertTrue(p.exists())
        shutil.rmtree(tmpdir)

    @skipIf(os.name == "nt", "Not yet available on Windows")
    def test_dir(self):
        output = check_output(["openmapflow", "dir"]).decode().rstrip()
        self.assertTrue(output.endswith("openmapflow"))

    @skipIf(os.name == "nt", "Not yet available on Windows")
    def test_ls(self):
        output = check_output(["openmapflow", "ls"]).decode().rstrip()
        self.assertIn("Dockerfile", output)
        self.assertIn("__init__.py", output)
        self.assertIn("config.py", output)
        self.assertIn("constants.py", output)
        self.assertIn("generate.py", output)
        self.assertIn("inference_utils.py", output)
        self.assertIn("inference_widgets.py", output)
        self.assertIn("labeled_dataset.py", output)
        self.assertIn("notebooks", output)
        self.assertIn("pytorch_dataset.py", output)
        self.assertIn("scripts", output)
        self.assertIn("templates", output)
        self.assertIn("train_utils.py", output)
        self.assertIn("trigger_inference_function", output)
        self.assertIn("utils.py", output)

    @skipIf(os.name == "nt", "Not yet available on Windows")
    def test_version(self):
        self.assertEqual(
            check_output(["openmapflow", "version"]).decode().rstrip(), VERSION
        )
        self.assertEqual(
            check_output(["openmapflow", "--version"]).decode().rstrip(), VERSION
        )

    @skipIf(os.name == "nt", "Not yet available on Windows")
    def test_help(self):
        actual_output = check_output(["openmapflow", "help"]).decode().rstrip()
        long_line = "-" * 81
        expected_output = f"""{long_line}
                              OpenMapFlow CLI\n{long_line}
openmapflow cp <source> <destination> - copy a file or directory from the library
openmapflow create-datasets - creates datasets for all datasets in datasets.py
openmapflow datapath <DATAPATH> - outputs a relative path to the data directory
openmapflow datasets - outputs a list of all datasets
openmapflow deploy - deploys Google Cloud Architecture for project
openmapflow dir - outputs openmapflow library directory
openmapflow generate - generates an openmapflow project
openmapflow help - outputs this message
openmapflow ls - lists files in openmapflow library directory
openmapflow version - package version"""
        self.assertEqual(actual_output, expected_output)
