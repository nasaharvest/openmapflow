import os
import unittest

from pathlib import Path

from openmapflow.constants import (
    CONFIG_FILE,
    TEMPLATE_DATASETS,
    TEMPLATE_EVALUATE,
    TEMPLATE_TRAIN,
    VERSION,
)

from openmapflow.config import CONFIG_YML, PROJECT_ROOT
from openmapflow.config import DataPaths as dp


class TestProjectConfig(unittest.TestCase):
    def test_config(self):
        """Checks that the config file is valid for a given project."""

        has_issues = False
        if (PROJECT_ROOT / CONFIG_FILE).exists():
            print(f"\u2714 openmapflow.yaml exists")
        else:
            has_issues = True
            print(f"\u2716 openmapflow.yaml not found")

        if CONFIG_YML["version"] == VERSION:
            print(f"\u2714 openmapflow.yaml version matches package version: {VERSION}")
        else:
            has_issues = True
            print(
                f"\u2716 openmapflow.yaml version: {CONFIG_YML['version']} does not match package version: {VERSION}"
            )

        for p in [
            dp.RAW_LABELS,
            dp.PROCESSED_LABELS,
            dp.COMPRESSED_FEATURES,
            dp.MODELS,
        ]:
            if Path(f"{p}.dvc").exists():
                print(f"\u2714 data path {p}.dvc found")
            else:
                has_issues = True
                print(f"\u2716 data path {p}.dvc not found")

        for p in [dp.METRICS, dp.DATASETS]:
            if Path(p).exists():
                print(f"\u2714 data path {p} found")
            else:
                has_issues = True
                print(f"\u2716 data path {p} not found")

        if (PROJECT_ROOT / TEMPLATE_DATASETS.name).exists():
            print(f"\u2714 file {TEMPLATE_DATASETS.name} found")
        else:
            has_issues = True
            print(f"\u2716 file {TEMPLATE_DATASETS.name} not found")

        self.assertTrue(
            not has_issues,
            "There were issues with the project config, please check the logs.",
        )


if __name__ == "__main__":
    runner = unittest.TextTestRunner(stream=open(os.devnull, "w"), verbosity=2)
    unittest.main(testRunner=runner)
