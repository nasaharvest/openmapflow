from unittest import TestCase
import os
import sys
import pandas as pd
import numpy as np

from datetime import date
from openmapflow.constants import SUBSET
from openmapflow.raw_labels import RawLabels, _to_date, _train_val_test_split

os.chdir(os.path.dirname(os.path.realpath(__file__)))
sys.path.append("..")


class TestRawLabels(TestCase):
    def test_init(self):
        kwargs = dict(
            filename="test.csv", class_prob=0.5, train_val_test=(0.5, 0.5, 0.5)
        )
        self.assertRaises(ValueError, lambda: RawLabels(**kwargs))

    def test_to_date(self):
        np_date = np.datetime64("2020-01-01")
        str_date = "2020-01-01"
        df_date = pd.to_datetime("2020-01-01")
        obj_date = date(2020, 1, 1)
        self.assertEqual(_to_date(np_date), obj_date)
        self.assertEqual(_to_date(str_date), obj_date)
        self.assertEqual(_to_date(df_date), obj_date)

    def test_train_val_test_split(self):
        df = pd.DataFrame({"col": list(range(100))})

        df = _train_val_test_split(df, (1.0, 0.0, 0.0))
        expected_subsets = {"training": 100}
        self.assertEqual(df[SUBSET].value_counts().to_dict(), expected_subsets)

        df = _train_val_test_split(df, (0.8, 0.1, 0.1))
        actual_subsets = df[SUBSET].value_counts().to_dict()
        threshold = 10
        self.assertTrue(
            abs(actual_subsets["training"] - 80) < threshold, actual_subsets["training"]
        )
        self.assertTrue(
            abs(actual_subsets["validation"] - 10) < threshold,
            actual_subsets["validation"],
        )
        self.assertTrue(
            abs(actual_subsets["testing"] - 10) < threshold, actual_subsets["testing"]
        )

    def test_get_points(self):
        pass
