from datetime import date
from unittest import TestCase

from openmapflow.bbox import BBox


class TestBBox(TestCase):
    def test_get_identifier(self):
        bbox = BBox(min_lat=46.2195, min_lon=-74.5539, max_lat=46.221, max_lon=-74.5518)
        self.assertEqual(
            bbox.get_identifier(start_date=date(2019, 4, 1), end_date=date(2020, 4, 1)),
            "min_lat=46.2195_min_lon=-74.5539_max_lat=46.221_"
            + "max_lon=-74.5518_dates=2019-04-01_2020-04-01_all",
        )

    def test_bbox_from_str_regular(self):
        uri = (
            "tifs/min_lat=-0.0001_min_lon=30.4263_max_lat=0.0014_"
            + "max_lon=30.4277_dates=2020-01-01_2021-10-31_all.tif"
        )
        actual_bbox = BBox.from_str(uri)
        expected_bbox = BBox(
            min_lat=-0.0001, max_lat=0.0014, min_lon=30.4263, max_lon=30.4277, name=uri
        )
        self.assertEqual(actual_bbox, expected_bbox)

    def test_bbox_from_str_single_decimal(self):
        uri = (
            "tifs/min_lat=0.0_min_lon=30.3_max_lat=0.1_max_lon=30.4"
            + "_dates=2020-01-01_2021-10-31_all.tif"
        )
        actual_bbox = BBox.from_str(uri)
        expected_bbox = BBox(
            min_lat=0.0, max_lat=0.1, min_lon=30.3, max_lon=30.4, name=uri
        )
        self.assertEqual(actual_bbox, expected_bbox)

    def test_bbox_from_str_whole_number(self):
        uri = "tifs/min_lat=0_min_lon=30_max_lat=1_max_lon=31_dates=2020-01-01_2021-10-31_all.tif"
        actual_bbox = BBox.from_str(uri)
        expected_bbox = BBox(min_lat=0, max_lat=1, min_lon=30, max_lon=31, name=uri)
        self.assertEqual(actual_bbox, expected_bbox)

    def test_bbox_from_str_another_number(self):
        uri = "tifs=2/min_lat=0_min_lon=30_max_lat=1_max_lon=31_dates=2020-01-01_2021-10-31_all.tif"
        actual_bbox = BBox.from_str(uri)
        expected_bbox = BBox(min_lat=0, max_lat=1, min_lon=30, max_lon=31, name=uri)
        self.assertEqual(actual_bbox, expected_bbox)

    def test_bbox_from_str_wrong_order(self):
        uri = "tifs/min_lon=0_min_lat=30_max_lat=1_max_lon=31_dates=2020-01-01_2021-10-31_all.tif"
        self.assertRaises(ValueError, BBox.from_str, uri)

    def test_bbox_from_str_missing_coord(self):
        uri = "tifs/min_lat=30_max_lat=1_max_lon=31_dates=2020-01-01_2021-10-31_all.tif"
        self.assertRaises(ValueError, BBox.from_str, uri)
