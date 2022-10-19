import re
import geopandas as gpd

import cartopy.io.shapereader as shpreader
from dataclasses import dataclass
from typing import List, Tuple


@dataclass
class AdminBoundary:
    country_iso3: str
    regions_of_interest: List[str]

    def __post_init__(self):

        assert len(self.country_iso3) == 3, "country_iso3 should be 3 letters"
        self.country_iso3 = self.country_iso3.upper()

        natural_earth_data = gpd.read_file(
            shpreader.natural_earth(
                resolution="10m", category="cultural", name="admin_1_states_provinces"
            )
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

    def get_admin_identifier(self, start_date, end_start) -> str:

        country_code = self.country_iso3.lower()
        if len(self.regions_of_interest) == 0:
            return f"country_code={country_code}_dates={start_date}_{end_start}"
        else:
            return (
                f"country_code={country_code}_region(s)={'_'.join(self.regions_of_interest)}_"
                f"dates={start_date}_{end_start}"
            )

    @classmethod
    def from_str(cls, s: str) -> "AdminBoundary":
        "Get country_iso3 and regions_of_interest from a string"

        country_iso3 = re.search(r"country_code=(\w{3})", s).group(1)
        regions_of_interest = re.search(r"region\(s\)=([\w_]+)", s)
        if regions_of_interest:
            regions_of_interest = regions_of_interest.group(1).split("_")
        else:
            regions_of_interest = []
        return cls(country_iso3=country_iso3, regions_of_interest=regions_of_interest)
