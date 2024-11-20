import os
import tempfile
from pathlib import Path
from unittest import TestCase, skipIf
from unittest.mock import patch

import yaml

from openmapflow.constants import (
    DATA_DIR,
    TEMPLATE_DATASETS,
    TEMPLATE_DEPLOY_YML,
    TEMPLATE_EVALUATE,
    TEMPLATE_README,
    TEMPLATE_TEST_YML,
    TEMPLATE_TRAIN,
)
from openmapflow.generate import (
    allow_write,
    copy_template_files,
    create_data_dirs,
    create_github_actions,
    fill_in_and_write_action,
    get_git_root,
    setup_dvc,
)


class TestGenerate(TestCase):
    def test_allow_write(self):
        self.assertTrue(allow_write(p="non-existent/file/path", overwrite=False))

        with tempfile.NamedTemporaryFile() as tmp:
            self.assertTrue(allow_write(p=tmp.name, overwrite=True))
            __builtins__["input"] = lambda _: "y"
            self.assertTrue(allow_write(p=tmp.name, overwrite=False))
            __builtins__["input"] = lambda _: "n"
            self.assertFalse(allow_write(p=tmp.name, overwrite=False))

    def test_copy_template_files(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            copy_template_files(Path(tmpdir), overwrite=False)
            for p in [
                TEMPLATE_DATASETS,
                TEMPLATE_TRAIN,
                TEMPLATE_EVALUATE,
                TEMPLATE_README,
            ]:
                self.assertTrue(
                    (Path(tmpdir) / p.name).exists(), f"{p.name} not copied"
                )

    @skipIf(os.name == "nt", "Tempdir doesn't work on windows")
    def test_create_data_dirs(self):

        with tempfile.TemporaryDirectory() as tmpdir:
            os.chdir(tmpdir)

            # Can only be imported once in directory
            from openmapflow.config import DataPaths as dp

            create_data_dirs(dp, overwrite=False)

            for p in [dp.RAW_LABELS, dp.DATASETS, dp.MODELS]:
                self.assertTrue(Path(p).exists())

    @skipIf(os.name == "nt", "Tempdir doesn't work on windows")
    def test_fill_in_and_write_action(self):

        srcs = [TEMPLATE_DEPLOY_YML, TEMPLATE_TEST_YML]
        dests = [Path("test.yaml"), Path("deploy.yaml")]

        for src, dest in zip(srcs, dests):

            with src.open("r") as f:
                template_action = f.read()

            yaml.safe_load(template_action)  # Verify it's valid YAML

            with tempfile.TemporaryDirectory() as tmpdir:
                os.chdir(tmpdir)

                fill_in_and_write_action(
                    src_yml_path=src,
                    dest_yml_path=dest,
                    sub_prefix="project-prefix",
                    sub_paths="path/project/data",
                    sub_cd="path/project",
                )

                with dest.open("r") as f:
                    project_action = f.read()

            yaml.safe_load(project_action)  # Verify it's valid YAML

            self.assertIn("<PREFIX>", template_action)
            self.assertIn("<CD>", template_action)
            self.assertNotIn("<PREFIX>", project_action)
            self.assertNotIn("<PATHS>", project_action)
            self.assertNotIn("<CD>", project_action)
            self.assertIn("project-prefix", project_action)
            self.assertIn("pip install -r requirements.txt", project_action)
            self.assertIn("path/project", project_action)

    @skipIf(os.name == "nt", "Tempdir doesn't work on windows")
    def test_create_github_actions_deploy(self):

        with tempfile.TemporaryDirectory() as tmpdir:
            os.chdir(tmpdir)

            # Can only be imported once in directory
            from openmapflow.config import DataPaths as dp

            create_github_actions(
                git_root=Path(tmpdir),
                is_subdir=False,
                PROJECT="fake-project",
                dp=dp,
                overwrite=False,
            )

            deploy_path = Path(f"{tmpdir}/.github/workflows/fake-project-deploy.yaml")

            with deploy_path.open("r") as f:
                actual_deploy_action = yaml.safe_load(f)

        expected_deploy_action = {
            "name": "fake-deploy",
            True: {"push": {"branches": ["main"], "paths": "data/models.dvc"}},
            "jobs": {
                "deploy": {
                    "runs-on": "ubuntu-latest",
                    "defaults": {"run": {"working-directory": "."}},
                    "steps": [
                        {"uses": "actions/checkout@v2"},
                        {
                            "name": "Set up python",
                            "uses": "actions/setup-python@v2",
                            "with": {"python-version": 3.9},
                        },
                        {
                            "name": "Install dependencies",
                            "run": "pip install -r requirements.txt",
                        },
                        {
                            "uses": "google-github-actions/auth@v0",
                            "with": {"credentials_json": "${{ secrets.GCP_SA_KEY }}"},
                        },
                        {
                            "name": "Set up Cloud SDK",
                            "uses": "google-github-actions/setup-gcloud@v0",
                        },
                        {"uses": "iterative/setup-dvc@v1"},
                        {
                            "name": "Deploy Google Cloud Architecture",
                            "run": "openmapflow deploy",
                        },
                    ],
                }
            },
        }
        self.assertEqual(expected_deploy_action, actual_deploy_action)

    @skipIf(os.name == "nt", "Tempdir doesn't work on windows")
    def test_create_github_actions_test(self):

        self.maxDiff = None

        with tempfile.TemporaryDirectory() as tmpdir:
            os.chdir(tmpdir)

            # Can only be imported once in directory
            from openmapflow.config import DataPaths as dp

            create_github_actions(
                git_root=Path(tmpdir),
                is_subdir=False,
                PROJECT="fake-project",
                dp=dp,
                overwrite=False,
            )

            test_path = Path(f"{tmpdir}/.github/workflows/fake-project-test.yaml")

            with test_path.open("r") as f:
                actual_test_action = yaml.safe_load(f)

        expected_test_action = {
            "name": "fake-test",
            True: {
                "push": {"branches": ["main"]},
                "pull_request": {"branches": ["main"]},
            },
            "jobs": {
                "test": {
                    "runs-on": "ubuntu-latest",
                    "defaults": {"run": {"working-directory": "."}},
                    "steps": [
                        {"name": "Clone repo", "uses": "actions/checkout@v2"},
                        {
                            "name": "Set up python",
                            "uses": "actions/setup-python@v2",
                            "with": {"python-version": 3.9},
                        },
                        {
                            "name": "Install dependencies",
                            "run": "pip install -r requirements.txt",
                        },
                        {
                            "name": "dvc pull data",
                            "run": "dvc pull -f",
                        },
                        {
                            "name": "Integration test - Project",
                            "run": "openmapflow cp templates/integration_test_project.py ."
                            + "\npython -m unittest integration_test_project.py\n",
                        },
                        {
                            "name": "Integration test - Data integrity",
                            "run": "openmapflow cp templates/integration_test_datasets.py ."
                            + "\npython -m unittest integration_test_datasets.py\n",
                        },
                        {
                            "name": "Integration test - Train and evaluate",
                            "run": "openmapflow cp templates/integration_test_train_evaluate.py ."
                            + "\npython -m unittest integration_test_train_evaluate.py\n",
                        },
                    ],
                }
            },
        }
        self.assertEqual(expected_test_action, actual_test_action)

    def test_get_git_root(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            self.assertRaises(FileExistsError, get_git_root, tmpdir_path)
            (tmpdir_path / ".git").mkdir()
            self.assertEqual(get_git_root(tmpdir_path), tmpdir_path)
            (tmpdir_path / "subdir").mkdir()
            self.assertEqual(get_git_root(tmpdir_path / "subdir"), tmpdir_path)

    @skipIf(os.name == "nt", "Tempdir doesn't work on windows")
    @patch("openmapflow.generate.os.system")
    def test_setup_dvc(self, mock_system):
        def input_response(prompt):
            if "a)" in prompt:
                return "a"
            return "gs://fake-bucket"

        with tempfile.TemporaryDirectory() as tmpdir:
            os.chdir(tmpdir)
            from openmapflow.config import DataPaths as dp

            Path(DATA_DIR).mkdir(parents=True, exist_ok=True)
            Path(DATA_DIR + ".gitignore").touch()

            __builtins__["input"] = input_response

            setup_dvc(Path(tmpdir), is_subdir=False, dp=dp)

        system_calls = [call[0][0] for call in mock_system.call_args_list]
        dvc_files = [
            dp.RAW_LABELS,
            dp.DATASETS,
            dp.MODELS,
        ]
        self.assertIn("dvc init", system_calls)
        self.assertIn("dvc add " + " ".join(dvc_files), system_calls)
        self.assertIn("dvc remote add -d gcs gs://fake-bucket", system_calls)
        self.assertIn("dvc push", system_calls)
