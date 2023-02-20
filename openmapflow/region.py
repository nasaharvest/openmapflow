from abc import ABC, abstractmethod

import geopandas as gpd


class Region(ABC):
    """ Abstract base class for define region of 
        interest """

    @abstractmethod
    def from_str(cls, s: str) -> "Region":
        pass

    @abstractmethod
    def get_identifier(self, start_date, end_date):
        pass

    @abstractmethod
    def get_area_km2(self):
        pass

    @abstractmethod
    def get_leaflet_geometry(self):
        pass