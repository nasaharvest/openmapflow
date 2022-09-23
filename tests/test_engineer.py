import unittest
from datetime import datetime
from pathlib import Path

import numpy as np

from openmapflow.bands import BANDS, DYNAMIC_BANDS, STATIC_BANDS
from openmapflow.engineer import fillna, load_tif, process_test_file

TIF_FILE = Path(__file__).parent / "98-togo_2019-02-06_2020-02-01.tif"

NUM_TIMESTEPS = 12


class TestEngineer(unittest.TestCase):
    def test_load_tif_file(self):

        loaded_file, _ = load_tif(TIF_FILE)
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

        loaded_file, average_slope = load_tif(TIF_FILE)

        # slope is calculated from neighbouring points, so the
        # edges are NaN
        for lat_idx in range(loaded_file.shape[2]):
            for lon_idx in range(loaded_file.shape[3]):
                # we remove the first index to simulate removing B1 and B10
                array = loaded_file.values[:, 2:, lat_idx, lon_idx]
                num_timesteps = array.shape[0]
                # and we add an array to simulate adding NDVI
                array = np.concatenate([array, np.ones([num_timesteps, 1])], axis=1)
                new_array = fillna(array, average_slope)
                if np.isnan(array[-2]).all():
                    self.assertTrue((new_array[-2] == average_slope).all())
                self.assertFalse(np.isnan(new_array).any())

    def test_fillna_simulated(self):
        array = np.array([[1, float("NaN"), 3]] * len(BANDS)).T
        expected_array = np.array([[1, 2, 3]] * len(BANDS)).T
        new_array = fillna(array, average_slope=1)
        self.assertTrue(np.array_equal(new_array, expected_array))

    def test_process_test_file(self):
        x_np, flat_lat, flat_lon = process_test_file(
            TIF_FILE, start_date=datetime(2019, 2, 6, 0, 0)
        )
        self.assertEqual(x_np.shape, (289, 12, 18))
        self.assertEqual(flat_lat.shape, (289,))
        self.assertEqual(flat_lon.shape, (289,))
