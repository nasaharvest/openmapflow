import warnings
from pathlib import Path
from unittest import TestCase, skipIf
from unittest.mock import Mock, patch

import numpy as np

from openmapflow.bbox import BBox
from openmapflow.labeled_dataset_custom import (
    _distance,
    _distance_point_from_center,
    _find_matching_point,
    _find_nearest,
    _generate_bbox_from_paths,
    _get_tif_paths,
)

try:
    import xarray as xr

    XARRAY_NOT_INSTALLED = False
except ImportError:
    warnings.warn("xarray not installed, some tests will be skipped")
    XARRAY_NOT_INSTALLED = True


paths = [
    "tifs/min_lat=-0.0001_min_lon=30.4263_max_lat=0.0014_max_lon=30.4277_dates=2020-01-01_2021-10-31_all.tif",  # noqa: E501
    "tifs/min_lat=-0.0001_min_lon=30.4263_max_lat=0.0014_max_lon=30.4277_dates=2020-01-01_2021-12-31_all.tif",  # noqa: E501
    "tifs/min_lat=-0.0007_min_lon=28.5108_max_lat=0.0007_max_lon=28.5123_dates=2019-01-01_2020-12-31_all.tif",  # noqa: E501
    "tifs/min_lat=-0.0007_min_lon=28.5109_max_lat=0.0007_max_lon=28.5123_dates=2019-01-01_2020-12-31_all.tif",  # noqa: E501
    "tifs/min_lat=-0.0008_min_lon=28.5109_max_lat=0.0007_max_lon=28.5123_dates=2019-01-01_2020-12-31_all.tif",  # noqa: E501
]
path_to_bbox = {
    Path(paths[0]): BBox(
        min_lat=-0.0001, max_lat=0.0014, min_lon=30.4263, max_lon=30.4277, name=paths[0]
    ),
    Path(paths[1]): BBox(
        min_lat=-0.0001, max_lat=0.0014, min_lon=30.4263, max_lon=30.4277, name=paths[1]
    ),
    Path(paths[2]): BBox(
        min_lat=-0.0007, max_lat=0.0007, min_lon=28.5108, max_lon=28.5123, name=paths[2]
    ),
    Path(paths[3]): BBox(
        min_lat=-0.0007, max_lat=0.0007, min_lon=28.5109, max_lon=28.5123, name=paths[3]
    ),
    Path(paths[4]): BBox(
        min_lat=-0.0008, max_lat=0.0007, min_lon=28.5109, max_lon=28.5123, name=paths[4]
    ),
}


class TestLabeledDatasetCustom(TestCase):
    """Tests for dataset"""

    @skipIf(XARRAY_NOT_INSTALLED, "xarray not installed")
    @patch("openmapflow.labeled_dataset_custom.load_tif")
    def test_find_matching_point_from_one(self, mock_load_tif):
        mock_data = xr.DataArray(
            data=np.ones((24, 19, 17, 17)), dims=("time", "band", "y", "x")
        )
        mock_load_tif.return_value = mock_data, 0.0
        labelled_np, closest_lon, closest_lat, source_file = _find_matching_point(
            start="2020-10-10",
            eo_paths=[Path("mock")],
            label_lon=5,
            label_lat=5,
            tif_bucket=Mock(),
        )
        self.assertEqual(closest_lon, 5)
        self.assertEqual(closest_lat, 5)
        self.assertEqual(source_file, "mock")
        self.assertEqual(labelled_np.shape, (24, 18))
        expected = np.ones((24, 18))
        expected[:, -1] = 0  # NDVI is 0
        self.assertTrue((labelled_np == expected).all())

    @skipIf(XARRAY_NOT_INSTALLED, "xarray not installed")
    @patch("openmapflow.labeled_dataset_custom.load_tif")
    def test_find_matching_point_from_multiple(self, mock_load_tif):
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
        labelled_np, closest_lon, closest_lat, source_file = _find_matching_point(
            start="2020-10-10",
            eo_paths=tif_paths,
            label_lon=8,
            label_lat=8,
            tif_bucket=Mock(),
        )
        self.assertEqual(closest_lon, 8)
        self.assertEqual(closest_lat, 8)
        self.assertEqual(source_file, "mock1")
        expected = np.ones((24, 18)) * 0.0
        self.assertTrue((labelled_np == expected).all())

    def test_find_nearest(self):
        val, idx = _find_nearest([1.0, 2.0, 3.0, 4.0, 5.0], 4.0)
        self.assertEqual(val, 4.0)
        self.assertEqual(idx, 3)

        val, idx = _find_nearest(np.array([1.0, 2.0, 3.0, -4.0, -5.0]), -1.0)
        self.assertEqual(val, 1.0)
        self.assertEqual(idx, 0)

    def test_distance(self):
        self.assertAlmostEqual(_distance(0, 0, 0.01, 0.01), 1.5725337265584898)
        self.assertAlmostEqual(_distance(0, 0, 0.01, 0), 1.1119492645167193)
        self.assertAlmostEqual(_distance(0, 0, 0, 0.01), 1.1119492645167193)

    @skipIf(XARRAY_NOT_INSTALLED, "xarray not installed")
    def test_distance_point_from_center(self):
        tif = xr.DataArray(
            attrs={"x": np.array([25, 35, 45]), "y": np.array([35, 45, 55])}
        )
        self.assertEqual(_distance_point_from_center(0, 0, tif), 2.0)
        self.assertEqual(_distance_point_from_center(0, 1, tif), 1.0)
        self.assertEqual(_distance_point_from_center(1, 1, tif), 0.0)
        self.assertEqual(_distance_point_from_center(2, 1, tif), 1.0)

    @skipIf(XARRAY_NOT_INSTALLED, "xarray not installed")
    @patch("openmapflow.labeled_dataset_custom.get_cloud_tif_list")
    def test_generate_bbox_from_paths(self, mock_get_cloud_tif_list):
        mock_get_cloud_tif_list.return_value = paths
        actual_output = _generate_bbox_from_paths()
        self.assertEqual(actual_output, path_to_bbox)

    def test_get_tif_paths(self):
        candidate_paths = _get_tif_paths(
            path_to_bbox,
            lat=0,
            lon=28.511,
            start_date="2019-01-01",
            end_date="2020-12-31",
        )
        expected_paths = [Path(p) for p in paths[2:]]
        self.assertEqual(candidate_paths, expected_paths)
