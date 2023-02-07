from dataclasses import dataclass
import re
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
            "../dataset/natural_earth/ne_10m_admin_1_states_provinces.shp"
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
    
    @classmethod
    def from_str(cls, s: str) -> "AdminBoundary":
        "Get country_iso3 and regions_of_interest from a string"

        country_iso3 = re.search(r"country_code=(\w{3})", s).group(1)
        regions_of_interest = re.search(r"region=([\w_]+)", s)
        if regions_of_interest:
            regions_of_interest = regions_of_interest.group(1).split("_")[:-1]
        else:
            regions_of_interest = []
        return cls(country_iso3=country_iso3, regions_of_interest=regions_of_interest)
    
    def get_identifier(self, start_date, end_date) -> str:
        "Get a unique identifier for the admin boundary"

        if len(self.regions_of_interest) == 0:
            return f"{self.country_iso3}_{start_date}_{end_date}_all"
        else:
            return f"{self.country_iso3}_{'_'.join(self.regions_of_interest)}_{start_date}_{end_date}_all"
        