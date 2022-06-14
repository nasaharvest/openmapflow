import os
from importlib import reload
from pathlib import Path
from unittest import TestCase

import openmapflow.config
from openmapflow.constants import (
    CONFIG_FILE,
    TEMPLATE_DATASETS,
    TEMPLATE_EVALUATE,
    TEMPLATE_TRAIN,
    VERSION,
)


class TestExampleProjectsGenerated(TestCase):
    def test_example_projects_generated(self):
        for project in ["buildings-example", "crop-mask-example", "maize-example"]:
            project_dir = Path(__file__).parent.parent / project
            os.chdir(project_dir)

            reload(openmapflow.config)
            from openmapflow.config import PROJECT_ROOT, CONFIG_YML, DataPaths as dp

            self.assertTrue(
                (PROJECT_ROOT / CONFIG_FILE).exists(),
                f"{project}: config file not found",
            )

            self.assertEqual(CONFIG_YML["version"], VERSION)

            for p in [
                dp.RAW_LABELS,
                dp.PROCESSED_LABELS,
                dp.COMPRESSED_FEATURES,
                dp.MODELS,
            ]:
                self.assertTrue(
                    Path(f"{p}.dvc").exists(), f"{project}: data path {p}.dvc not found"
                )
            for p in [dp.METRICS, dp.DATASETS]:
                self.assertTrue(Path(p).exists(), f"{project}: data path {p} not found")

            for p in [TEMPLATE_DATASETS, TEMPLATE_TRAIN, TEMPLATE_EVALUATE]:
                self.assertTrue(
                    (PROJECT_ROOT / p.name).exists(),
                    f"{project}: file {p.name} not found",
                )
