from dataclasses import dataclass
from typing import List

import ee
import numpy as np
from geopandas import gpd
from shapely.geometry import Polygon

from openmapflow.admin_bounds import AdminBoundary


@dataclass
class EEAdminBoundary(AdminBoundary):
    r""" """

    def to_ee_polygon(self) -> ee.Geometry.Polygon:
        if self.boundary.geom_type[0] == "Polygon":
            for i in self.boundary.geometry:
                return ee.Geometry.Polygon(np.dstack(i.exterior.coords.xy).tolist())
        elif self.boundary.geom_type[0] == "MultiPolygon":
            for i in self.boundary.geometry:
                return [
                    ee.Geometry.Polygon(np.dstack(j.exterior.coords.xy).tolist())
                    for j in i
                ]

    def to_polygons(self, metres_per_polygon: int = 10000) -> List[ee.Geometry.Polygon]:

        self.boundary = self.boundary.to_crs(
            epsg=3857
        )  # convert to web mercator for area calculation
        xmin, ymin, xmax, ymax = self.boundary.total_bounds

        cols = np.arange(xmin, xmax + metres_per_polygon, metres_per_polygon)
        rows = np.arange(ymin, ymax + metres_per_polygon, metres_per_polygon)

        print(f"Splitting into {len(cols)-1} columns and {len(rows)-1} rows")

        polygons = []
        for x in cols[:-1]:
            for y in rows[:-1]:
                polygons.append(
                    Polygon(
                        [
                            (x, y),
                            (x + metres_per_polygon, y),
                            (x + metres_per_polygon, y + metres_per_polygon),
                            (x, y + metres_per_polygon),
                        ]
                    )
                )
        fish_net = gpd.GeoDataFrame({"geometry": polygons}, crs=self.boundary.crs)
        boundary_clip = gpd.clip(fish_net, self.boundary).to_crs(epsg=4326)
        output_polygons: List[ee.Geometry.Polygon] = []

        for i in boundary_clip.geometry:
            if i.geom_type == "Polygon":
                output_polygons.append(
                    ee.Geometry.Polygon(np.dstack(i.exterior.coords.xy).tolist())
                )
            elif i.geom_type == "MultiPolygon":
                for j in i:
                    output_polygons.append(
                        ee.Geometry.Polygon(np.dstack(j.exterior.coords.xy).tolist())
                    )
        return output_polygons

    def from_adminboundary(admin_bounds: AdminBoundary) -> "EEAdminBoundary":
        return EEAdminBoundary(
            country_iso3=admin_bounds.country_iso3,
            regions_of_interest=admin_bounds.regions_of_interest,
        )
