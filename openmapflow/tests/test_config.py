from unittest import TestCase
from unittest.mock import patch


class TestAllFeatures(TestCase):
    def test_all_features(self):
        all_features = AllFeatures()
        self.assertEqual(all_features.df, None)
