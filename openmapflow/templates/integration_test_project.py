import os
import unittest
from pathlib import Path

from openmapflow.config import CONFIG_YML, PROJECT_ROOT
from openmapflow.config import DataPaths as dp
from openmapflow.constants import CONFIG_FILE, TEMPLATE_DATASETS, VERSION


def path_exists(path: Path) -> bool:
    """Utility function to check if a path exists"""
    if path.exists():
        print(f"\u2714 {path.name} exists")
    else:
        print(f"\u2716 {path.name} not found")
    return path.exists()


class TestProjectConfig(unittest.TestCase):
    def test_config(self):
        """Checks that the config file is valid for a given project."""

        has_issues = False
        if not path_exists(PROJECT_ROOT / CONFIG_FILE):
            has_issues = True

        if CONFIG_YML["version"] == VERSION:
            print(f"\u2714 openmapflow.yaml version matches package version: {VERSION}")
        else:
            has_issues = True
            print(
                f"\u2716 openmapflow.yaml version: {CONFIG_YML['version']} "
                + f"does not match package version: {VERSION}"
            )

        if not path_exists(Path(dp.RAW_LABELS + ".dvc")):
            has_issues = True

        if not path_exists(Path(dp.DATASETS + ".dvc")):
            has_issues = True
        else:
            is_not_empty = any(Path(dp.DATASETS).iterdir())
            if is_not_empty and not path_exists(Path(dp.REPORT)):
                has_issues = True

        if not path_exists(Path(dp.MODELS)):
            has_issues = True
        else:
            is_not_empty = any(Path(dp.MODELS).iterdir())
            if is_not_empty and not path_exists(Path(dp.METRICS)):
                has_issues = True

        if not path_exists(PROJECT_ROOT / TEMPLATE_DATASETS.name):
            has_issues = True

        self.assertTrue(
            not has_issues,
            "There were issues with the project config, please check the logs.",
        )


if __name__ == "__main__":
    runner = unittest.TextTestRunner(stream=open(os.devnull, "w"), verbosity=2)
    unittest.main(testRunner=runner)
