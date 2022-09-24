import unittest
from pathlib import Path

import numpy as np
import xarray as xr

from openmapflow.bands import DYNAMIC_BANDS, STATIC_BANDS
from openmapflow.engineer import load_tif, process_test_file, _fillna

TIF_FILE = Path(__file__).parent / "98-togo_2019-02-06_2020-02-01.tif"

NUM_TIMESTEPS = 12
BANDS = DYNAMIC_BANDS + STATIC_BANDS


class TestEngineer(unittest.TestCase):
    def setUp(self):
        data = np.ones((NUM_TIMESTEPS, len(BANDS), 17, 17))

        # Make each band have a unique value
        for i in range(len(BANDS)):
            data[:, i] = data[:, i] * i
        self.xr_data = xr.DataArray(data=data, dims=("time", "band", "y", "x"))

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

    def test_fillna_identity(self):
        self.assertTrue((self.xr_data == _fillna(self.xr_data)).all())

    def test_fillna_wrong_shape(self):
        with self.assertRaises(ValueError):
            _fillna(self.xr_data[:, 0])
        with self.assertRaises(ValueError):
            _fillna(self.xr_data[:, 1:5])

    def test_fillna_missing_value(self):
        self.xr_data[0, 0, 0, 0] = float("nan")
        self.xr_data[5, 2, 5, 5] = float("nan")
        self.xr_data[11, 10, 16, 16] = float("nan")
        new_xr_data = _fillna(self.xr_data)
        self.assertEqual(new_xr_data[0, 0, 0, 0], 0)
        self.assertEqual(new_xr_data[5, 2, 5, 5], 2)
        self.assertEqual(new_xr_data[11, 10, 16, 16], 10)

    def test_fillna_missing_band(self):
        self.xr_data[:, 10, 16, 16] = float("nan")
        new_xr_data = _fillna(self.xr_data)
        self.assertEqual(new_xr_data[11, 10, 16, 16], 10)

    def test_fillna_missing_band_everywhere(self):
        self.xr_data[:, 10, :, :] = float("nan")
        new_xr_data = _fillna(self.xr_data)
        self.assertEqual(new_xr_data[11, 10, 16, 16], 0)

    def test_fillna_missing_whole_timestep(self):
        self.xr_data[3, :, :, :] = float("nan")
        new_xr_data = _fillna(self.xr_data)
        self.assertTrue((new_xr_data[3] == self.xr_data[4]).all())

    def test_fillna_missing_everything(self):
        self.xr_data[:, ::, :] = float("nan")
        new_xr_data = _fillna(self.xr_data)
        self.assertTrue((new_xr_data == 0).all())

    def test_fillna_real(self):
        loaded_file1 = load_tif(TIF_FILE, fillna=False)
        self.assertTrue(np.isnan(loaded_file1).any())
        loaded_file2 = load_tif(TIF_FILE, fillna=True)
        self.assertFalse(np.isnan(loaded_file2).any())

    def test_process_test_file(self):
        x_np, flat_lat, flat_lon = process_test_file(TIF_FILE)
        self.assertEqual(x_np.shape, (289, 12, 18))
        self.assertEqual(flat_lat.shape, (289,))
        self.assertEqual(flat_lon.shape, (289,))
