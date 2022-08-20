from unittest import TestCase

import pandas as pd

from openmapflow.constants import END, START
from openmapflow.labeled_dataset import get_label_timesteps


class TestLabeledDataset(TestCase):
    def test_get_label_timesteps(self):
        df = pd.DataFrame(
            {
                START: ["2019-01-01", "2019-01-01", "2019-01-01"],
                END: ["2020-10-31", "2020-12-31", "2020-12-31"],
            }
        )
        actual_output = list(get_label_timesteps(df).unique())
        self.assertEqual(actual_output, [22, 24])
