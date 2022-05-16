from unittest import TestCase

import pandas as pd

from openmapflow.constants import CLASS_PROB, SUBSET
from openmapflow.raw_labels import RawLabels, _set_class_prob, _train_val_test_split


class TestRawLabels(TestCase):
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
        # TODO: implement
        pass

    def test_read_in_file(self):
        # TODO: implement
        pass

    def test_set_class_prob_float(self):
        df = pd.DataFrame({"expected": [1.0, 1.0, 1.0]})
        df = _set_class_prob(df, 1.0)
        self.assertTrue(df[CLASS_PROB].equals(df["expected"]))

    def test_set_class_prob_int(self):
        df = pd.DataFrame({"expected": [1.0, 1.0, 1.0]})
        df = _set_class_prob(df, 1)
        self.assertTrue(df[CLASS_PROB].equals(df["expected"]))

    def test_init(self):
        kwargs = dict(
            filename="test.csv", class_prob=0.5, train_val_test=(0.5, 0.5, 0.5)
        )
        self.assertRaises(ValueError, lambda: RawLabels(**kwargs))
