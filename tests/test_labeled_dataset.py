from pathlib import Path
from unittest import TestCase
from unittest.mock import patch

import numpy as np
import xarray as xr

from openmapflow.labeled_dataset import (
    distance,
    distance_point_from_center,
    find_matching_point,
    find_nearest,
)


class TestDataset(TestCase):
    """Tests for dataset"""

    @patch("openmapflow.labeled_dataset.storage")
    @patch("openmapflow.labeled_dataset.Engineer.load_tif")
    def test_find_matching_point_from_one(self, mock_load_tif, mock_storage):
        mock_data = xr.DataArray(
            data=np.ones((24, 19, 17, 17)), dims=("time", "band", "y", "x")
        )
        mock_load_tif.return_value = mock_data, 0.0
        labelled_np, closest_lon, closest_lat, source_file = find_matching_point(
            start="2020-10-10",
            eo_paths=[Path("mock")],
            label_lon=5,
            label_lat=5,
            tif_bucket=mock_storage.Client().bucket,
        )
        self.assertEqual(closest_lon, 5)
        self.assertEqual(closest_lat, 5)
        self.assertEqual(source_file, "mock")
        self.assertEqual(labelled_np.shape, (24, 18))
        expected = np.ones((24, 18))
        expected[:, -1] = 0  # NDVI is 0
        self.assertTrue((labelled_np == expected).all())

    @patch("openmapflow.labeled_dataset.storage")
    @patch("openmapflow.labeled_dataset.Engineer.load_tif")
    def test_find_matching_point_from_multiple(self, mock_load_tif, mock_storage):
        tif_paths = [Path("mock1"), Path("mock2"), Path("mock3")]

        def side_effect(path, start_date, num_timesteps):
            idx = [i for i, p in enumerate(tif_paths) if p.stem == Path(path).stem][0]
            return (
                xr.DataArray(
                    dims=("time", "band", "y", "x"),
                    data=np.ones((24, 19, 17, 17)) * idx,
                ),
                0.0,
            )

        mock_load_tif.side_effect = side_effect
        labelled_np, closest_lon, closest_lat, source_file = find_matching_point(
            start="2020-10-10",
            tif_paths=tif_paths,
            label_lon=8,
            label_lat=8,
            tif_bucket=mock_storage.Client().bucket,
        )
        self.assertEqual(closest_lon, 8)
        self.assertEqual(closest_lat, 8)
        self.assertEqual(source_file, "mock1")
        expected = np.ones((24, 18)) * 0.0
        self.assertTrue((labelled_np == expected).all())

    def test_find_nearest(self):
        val, idx = find_nearest([1.0, 2.0, 3.0, 4.0, 5.0], 4.0)
        self.assertEqual(val, 4.0)
        self.assertEqual(idx, 3)

        val, idx = find_nearest(xr.DataArray([1.0, 2.0, 3.0, -4.0, -5.0]), -1.0)
        self.assertEqual(val, 1.0)
        self.assertEqual(idx, 0)

    def test_distance(self):
        self.assertAlmostEqual(distance(0, 0, 0.01, 0.01), 1.5725337265584898)
        self.assertAlmostEqual(distance(0, 0, 0.01, 0), 1.1119492645167193)
        self.assertAlmostEqual(distance(0, 0, 0, 0.01), 1.1119492645167193)

    def test_distance_point_from_center(self):
        tif = xr.DataArray(
            attrs={"x": np.array([25, 35, 45]), "y": np.array([35, 45, 55])}
        )
        self.assertEqual(distance_point_from_center(0, 0, tif), 2.0)
        self.assertEqual(distance_point_from_center(0, 1, tif), 1.0)
        self.assertEqual(distance_point_from_center(1, 1, tif), 0.0)
        self.assertEqual(distance_point_from_center(2, 1, tif), 1.0)
