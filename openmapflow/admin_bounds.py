from dataclasses import dataclass
from typing import List

import geopandas as gpd


@dataclass
class AdminBoundary:
    country_iso3: str
    regions_of_interest: List[str]

    def __post_init__(self):

        assert len(self.country_iso3) == 3, "country_iso3 should be 3 letters"
        self.country_iso3 = self.country_iso3.upper()
        self.regions_of_interest = [
            region.title() for region in self.regions_of_interest
        ]

        natural_earth_data = gpd.read_file(
            "openmapflow/dataset/natural_earth/ne_10m_admin_1_states_provinces.shp"
        )

        if len(self.regions_of_interest) == 0:
            boundary = natural_earth_data[
                natural_earth_data["adm1_code"].str.startswith(self.country_iso3)
            ].copy()
            self.boundary = boundary.dissolve(by="adm0_a3")

        else:
            available_regions = natural_earth_data[
                natural_earth_data["adm1_code"].str.startswith(self.country_iso3)
            ]["name"].tolist()

            region_not_found = [
                region
                for region in self.regions_of_interest
                if region not in available_regions
            ]

            assert (
                len(region_not_found) == 0
            ), f"Region(s) not found: {region_not_found} in {self.country_iso3}"

            country_boundary = natural_earth_data[
                natural_earth_data["adm1_code"].str.startswith(self.country_iso3)
            ]
            self.boundary = country_boundary[
                country_boundary["name"].isin(self.regions_of_interest)
            ].copy()
            self.boundary = self.boundary.dissolve(by="adm0_a3")
