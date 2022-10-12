from dataclasses import dataclass

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

    # TODO: divide the polygon into smaller polygons
    # def to_polygons(self) -> List[ee.Geometry.Polygon]:
    #     pass

    # def from_center
