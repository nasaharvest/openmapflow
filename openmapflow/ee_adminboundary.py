from dataclasses import dataclass
from typing import List

import ee
import numpy as np

from openmapflow.admin_bounds import AdminBoundary


@dataclass
class EEAdminBoundary(AdminBoundary):
    r""" """

    def to_ee_polygon(self) -> ee.Geometry.Polygon:
        features = []
        for i in range(len(self.boundary)):
            if len(self.boundary) == 0:
                return ee.Geometry
            geoms_ = [i for i in self.boundary.geometry]
            x, y = geoms_[i].exterior.coords.xy
            cords = np.dstack((x, y)).tolist()
            if i > 0:
                return ee.FeatureCollection(features.append(ee.Geometry.Polygon(cords)))
            else:
                return ee.Feature(ee.Geometry.Polygon(cords))

    def to_polygon(self, size_per_patch: int = 3300) -> List[ee.Geometry.Polygon]:

        self.boundary = self.boundary.to_crs(
            epsg=3857
        )  # TODO: might need a second on this
        xmin, ymin, xmax, ymax = self.boundary.total_bounds

        cols = np.arange(xmin, xmax + size_per_patch, size_per_patch)
        rows = np.arange(ymin, ymax + size_per_patch, size_per_patch)

        print(f"Splitting into {len(cols)-1} columns and {len(rows)-1} rows")

        out_poygons: List[ee.Geometry.Polygon] = []
        for x in cols[:-1]:
            for y in rows[:-1]:
                out_poygons.append(
                    ee.Geometry.Polygon(
                        [
                            [x, y],
                            [x, y + size_per_patch],
                            [x + size_per_patch, y + size_per_patch],
                            [x + size_per_patch, y],
                        ],
                    )
                )

        return out_poygons
