from unittest import TestCase

from openmapflow.label_utils import train_val_test_split


class TestLabelUtils(TestCase):
    def test_train_val_test_split(self):
        series = train_val_test_split(index=range(100), val=0.0, test=0.0)
        self.assertEqual(series.value_counts().to_dict(), {"training": 100})

        series = train_val_test_split(index=range(100), val=0.1, test=0.1)
        actual_subsets = series.value_counts().to_dict()
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
