from dataclasses import dataclass
from typing import List

import ee
from geopandas import gpd
import numpy as np
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

    def to_polygons(self, size_per_patch: int = 3300) -> List[ee.Geometry.Polygon]:

        self.boundary = self.boundary.to_crs(
            epsg=3857
        ) # convert to web mercator for area calculation
        xmin, ymin, xmax, ymax = self.boundary.total_bounds

        cols = np.arange(xmin, xmax + size_per_patch, size_per_patch)
        rows = np.arange(ymin, ymax + size_per_patch, size_per_patch)

        print(f"Splitting into {len(cols)-1} columns and {len(rows)-1} rows")

        polygons = []
        for x in cols[:-1]:
            for y in rows[:-1]:
                polygons.append(
                    Polygon(
                        [
                            (x, y),
                            (x + size_per_patch, y),
                            (x + size_per_patch, y + size_per_patch),
                            (x, y + size_per_patch),
                        ]
                    )
                )
        fish_net = gpd.GeoDataFrame({"geometry": polygons}, crs=self.boundary.crs)
        boundary_clip = gpd.clip(fish_net, self.boundary).to_crs(epsg=4326)
        for i in boundary_clip.geometry:
            return [ee.Geometry.Polygon(np.dstack(i.exterior.coords.xy).tolist())]
