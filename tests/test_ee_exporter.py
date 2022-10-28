from datetime import date
from unittest import TestCase
from unittest.mock import call, patch

import pandas as pd

from openmapflow.bbox import BBox
from openmapflow.admin_bounds import AdminBoundary
from openmapflow.constants import END, LAT, LON, START
from openmapflow.ee_exporter import (
    EarthEngineExporter,
    get_cloud_tif_list,
    get_ee_task_amount,
)

try:
    import ee  # noqa: F401

    SKIP_TEST = False
except ImportError:
    SKIP_TEST = True


class TestEarthEngineExporter(TestCase):
    def setUp(self) -> None:
        if SKIP_TEST:
            self.skipTest("earthengine-api not installed")

    @patch("openmapflow.ee_exporter.import_optional_dependency")
    def test_get_cloud_tif_list(self, mock_dependency):
        mock_dependency.storage.Client().list_blobs("mock_bucket").return_value = []
        tif_list = get_cloud_tif_list("mock_bucket")
        assert tif_list == []

    @patch("openmapflow.ee_exporter.ee.Initialize")
    @patch("openmapflow.ee_exporter.ee.Geometry.Polygon")
    @patch("openmapflow.ee_exporter.EarthEngineExporter._export_for_polygon")
    def test_export_for_labels(
        self, mock_export_for_polygon, mock_polygon, mock_ee_initialize
    ):
        mock_polygon.return_value = None
        start_date = date(2019, 4, 22)
        end_date = date(2020, 4, 16)

        labels = pd.DataFrame(
            {
                LON: [-74.55285616553732, -75.55285616553732],
                LAT: [46.22024965230018, 47.22024965230018],
                END: [str(end_date), str(end_date)],
                START: [str(start_date), str(start_date)],
            }
        )
        EarthEngineExporter(
            dest_bucket="mock", check_gcp=False, check_ee=False
        ).export_for_labels(labels=labels)

        assert mock_export_for_polygon.call_count == 2

        ending = f"dates={start_date}_{end_date}_all"

        identifier_1 = (
            f"min_lat=46.2195_min_lon=-74.5539_max_lat=46.221_max_lon=-74.5518_{ending}"
        )
        identifier_2 = (
            f"min_lat=47.2195_min_lon=-75.5539_max_lat=47.221_max_lon=-75.5518_{ending}"
        )
        mock_export_for_polygon.assert_has_calls(
            [
                call(
                    end_date=end_date,
                    polygon=None,
                    polygon_identifier=identifier_1,
                    start_date=start_date,
                    test=False,
                ),
                call(
                    end_date=end_date,
                    polygon=None,
                    polygon_identifier=identifier_2,
                    start_date=start_date,
                    test=False,
                ),
            ],
            any_order=True,
        )

    @patch("openmapflow.ee_exporter.ee.Initialize")
    @patch("openmapflow.ee_exporter.ee.Geometry.Polygon")
    @patch("openmapflow.ee_exporter.EarthEngineExporter._export_for_polygon")
    def test_export_for_bbox_metres_per_polygon_None(
        self, mock_export_for_polygon, mock_polygon, mock_ee_initialize
    ):
        mock_polygon.return_value = None

        start_date, end_date = date(2019, 4, 1), date(2020, 4, 1)
        bbox = BBox(
            min_lon=-0.1501,
            max_lon=1.7779296875,
            min_lat=6.08940429687,
            max_lat=11.115625,
        )

        EarthEngineExporter(
            dest_bucket="mock", check_gcp=False, check_ee=False
        ).export_for_bbox(
            bbox=bbox,
            bbox_name="Togo",
            start_date=start_date,
            end_date=end_date,
            metres_per_polygon=None,
        )
        self.assertEqual(mock_export_for_polygon.call_count, 1)
        mock_export_for_polygon.assert_called_with(
            end_date=end_date,
            polygon=None,
            polygon_identifier="Togo/batch/0",
            start_date=start_date,
            file_dimensions=None,
            test=True,
        )

    @patch("openmapflow.ee_exporter.ee.Initialize")
    @patch("openmapflow.ee_exporter.ee.Geometry.Polygon")
    @patch("openmapflow.ee_exporter.EarthEngineExporter._export_for_polygon")
    def test_export_for_bbox_metres_per_polygon_10000(
        self, mock_export_for_polygon, mock_polygon, mock_ee_initialize
    ):
        mock_polygon.return_value = None

        start_date, end_date = date(2019, 4, 1), date(2020, 4, 1)
        bbox = BBox(
            min_lon=-0.1501,
            max_lon=1.7779296875,
            min_lat=6.08940429687,
            max_lat=11.115625,
        )
        EarthEngineExporter(
            dest_bucket="mock", check_gcp=False, check_ee=False
        ).export_for_bbox(
            bbox=bbox,
            bbox_name="Togo",
            start_date=start_date,
            end_date=end_date,
            metres_per_polygon=10000,
        )
        self.assertEqual(mock_export_for_polygon.call_count, 1155)
        mock_export_for_polygon.assert_has_calls(
            [
                call(
                    end_date=end_date,
                    polygon=None,
                    polygon_identifier=f"Togo/batch_{i}/{i}",
                    start_date=start_date,
                    file_dimensions=None,
                    test=True,
                )
                for i in range(1155)
            ],
            any_order=True,
        )

    @patch("openmapflow.ee_exporter.ee.Initialize")
    @patch("openmapflow.ee_exporter.ee.Geometry.Polygon")
    @patch("openmapflow.ee_exporter.EarthEngineExporter._export_for_polygon")
    def test_export_for_adminbounds_size_per_polygon_None(
        self, mock_export_for_polygon, mock_polygon, mock_ee_initialize
    ):
        mock_polygon.return_value = None

        start_date, end_date = date(2019, 4, 1), date(2020, 4, 1)
        admin_bounds = AdminBoundary(
            country_iso3="NGA",
            regions_of_interest=[],
        )
        EarthEngineExporter(
            dest_bucket="mock", check_gcp=False, check_ee=False
        ).export_for_adminbondary(
            adminboundary=admin_bounds,
            start_date=start_date,
            end_date=end_date,
            metres_per_polygon=None,
        )
        self.assertEqual(mock_export_for_polygon.call_count, 1)
        mock_export_for_polygon.assert_called_with(
            end_date=end_date,
            polygon=None,
            polygon_identifier="NGA/batch/0",
            start_date=start_date,
            file_dimensions=None,
            test=True,
        )

    @patch("openmapflow.ee_exporter.ee.Initialize")
    @patch("openmapflow.ee_exporter.ee.Geometry.Polygon")
    @patch("openmapflow.ee_exporter.EarthEngineExporter._export_for_polygon")
    def test_export_for_adminbounds_size_per_polygon_10000(
        self, mock_export_for_polygon, mock_polygon, mock_ee_initialize
    ):
        mock_polygon.return_value = None

        start_date, end_date = date(2019, 4, 1), date(2020, 4, 1)
        admin_bounds = AdminBoundary(
            country_iso3="TGO",
            regions_of_interest=[],
        )
        EarthEngineExporter(
            dest_bucket="mock", check_gcp=False, check_ee=False
        ).export_for_adminbondary(
            adminboundary=admin_bounds,
            start_date=start_date,
            end_date=end_date,
            metres_per_polygon=100000,
        )
        self.assertEqual(mock_export_for_polygon.call_count, 6)
        mock_export_for_polygon.assert_has_calls(
            [
                call(
                    end_date=end_date,
                    polygon=None,
                    polygon_identifier=f"NGA/batch_{i}/{i}",
                    start_date=start_date,
                    file_dimensions=None,
                    test=True,
                )
                for i in range(6)
            ],
            any_order=True,
        )

    @patch("openmapflow.ee_exporter.ee")
    def test_get_ee_task_amount_one(self, mock_ee):
        mock_ee.data.getTaskList.return_value = [{"state": "READY"}]
        self.assertEqual(get_ee_task_amount(), 1)

    @patch("openmapflow.ee_exporter.ee")
    def test_get_ee_task_amount_many(self, mock_ee):
        mock_ee.data.getTaskList.return_value = [
            {"state": "READY"},
            {"state": "RUNNING"},
        ] * 10
        self.assertEqual(get_ee_task_amount(), 20)

    @patch("openmapflow.ee_exporter.ee")
    def test_get_ee_task_amount_completed(self, mock_ee):
        mock_ee.data.getTaskList.return_value = [{"state": "COMPLETED"}] * 10
        self.assertEqual(get_ee_task_amount(), 0)

    @patch("openmapflow.ee_exporter.ee")
    def test_get_ee_task_amount_prefix(self, mock_ee):
        mock_ee.data.getTaskList.return_value = [
            {"state": "READY", "description": "special_prefix_suffix_ending"}
        ] * 10
        self.assertEqual(get_ee_task_amount(), 10)
        self.assertEqual(get_ee_task_amount("special_prefix"), 10)
        self.assertEqual(get_ee_task_amount("other_prefix"), 0)
