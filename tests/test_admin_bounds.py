from unittest import TestCase

from openmapflow.admin_bounds import AdminBoundary


class TestAdminBound(TestCase):
    def test_adminbound_instance(self):
        admin_bound = AdminBoundary(country_iso3="RWA", regions_of_interest=[])
        self.assertEqual(admin_bound.boundary.geom_type[0], "Polygon")
