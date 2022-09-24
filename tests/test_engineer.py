import unittest
from pathlib import Path

import numpy as np

from openmapflow.bands import DYNAMIC_BANDS, STATIC_BANDS
from openmapflow.engineer import load_tif, process_test_file

TIF_FILE = Path(__file__).parent / "98-togo_2019-02-06_2020-02-01.tif"

NUM_TIMESTEPS = 12


class TestEngineer(unittest.TestCase):
    def test_load_tif_file(self):

        loaded_file = load_tif(TIF_FILE)
        self.assertEqual(loaded_file.shape[0], NUM_TIMESTEPS)
        self.assertEqual(loaded_file.shape[1], len(DYNAMIC_BANDS) + len(STATIC_BANDS))

        # also, check the static bands are actually constant across time
        static_bands = loaded_file.values[:, len(DYNAMIC_BANDS)]
        for i in range(1, NUM_TIMESTEPS):
            self.assertTrue(
                np.array_equal(static_bands[0], static_bands[i], equal_nan=True)
            )

        # finally, check expected for temperature
        temperature_band = DYNAMIC_BANDS.index("temperature_2m")
        temperature_values = loaded_file.values[:, temperature_band, :, :]
        self.assertTrue(((temperature_values) > 0).all())  # in Kelvin
        # https://en.wikipedia.org/wiki/Highest_temperature_recorded_on_Earth
        self.assertTrue(((temperature_values) < 329.85).all())

    def test_fillna_real(self):
        loaded_file1 = load_tif(TIF_FILE, fillnans=False)
        self.assertTrue(np.isnan(loaded_file1).any())
        loaded_file2 = load_tif(TIF_FILE, fillnans=True)
        self.assertFalse(np.isnan(loaded_file2).any())

    def test_process_test_file(self):
        x_np, flat_lat, flat_lon = process_test_file(TIF_FILE)
        self.assertEqual(x_np.shape, (289, 12, 18))
        self.assertEqual(flat_lat.shape, (289,))
        self.assertEqual(flat_lon.shape, (289,))
