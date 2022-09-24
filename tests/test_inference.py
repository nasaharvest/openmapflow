from unittest import TestCase

import numpy as np

from openmapflow.inference import Inference


class TestInference(TestCase):
    def test_combine_predictions(self):
        flat_lat = np.array(
            [14.95313164, 14.95313164, 14.95313164, 14.95313164, 14.95313164]
        )
        flat_lon = np.array(
            [-86.25070894, -86.25061911, -86.25052928, -86.25043945, -86.25034962]
        )
        batch_predictions = np.array(
            [[0.43200156], [0.55286014], [0.5265], [0.5236109], [0.4110847]]
        )
        df_predictions = Inference._combine_predictions(
            flat_lat=flat_lat, flat_lon=flat_lon, batch_predictions=batch_predictions
        )

        # Check size
        self.assertEqual(df_predictions.index.levels[0].name, "lat")
        self.assertEqual(df_predictions.index.levels[1].name, "lon")
        self.assertEqual(len(df_predictions.index.levels[0]), 1)
        self.assertEqual(len(df_predictions.index.levels[1]), 5)

        # Check coords
        self.assertTrue((df_predictions.index.levels[0].values == flat_lat[0:1]).all())
        self.assertTrue((df_predictions.index.levels[1].values == flat_lon).all())

        # Check all predictions between 0 and 1
        self.assertTrue(df_predictions["prediction_0"].min() >= 0)
        self.assertTrue(df_predictions["prediction_0"].max() <= 1)
