import json
import os
import warnings
from datetime import date, timedelta
from typing import Dict, List, Optional, Union

import pandas as pd
from pandas.compat._optional import import_optional_dependency

from openmapflow.bands import DAYS_PER_TIMESTEP, DYNAMIC_BANDS
from openmapflow.bbox import BBox
from openmapflow.constants import END, LAT, LON, START
from openmapflow.utils import tqdm

try:
    import ee

    from openmapflow.ee_boundingbox import EEBoundingBox
    from openmapflow.eo.era5 import get_single_image as get_single_era5_image
    from openmapflow.eo.sentinel1 import get_image_collection as get_s1_image_collection
    from openmapflow.eo.sentinel1 import get_single_image as get_single_s1_image
    from openmapflow.eo.sentinel2 import get_single_image as get_single_s2_image
    from openmapflow.eo.srtm import get_single_image as get_single_srtm_image

    DYNAMIC_IMAGE_FUNCTIONS = [get_single_s2_image, get_single_era5_image]
    STATIC_IMAGE_FUNCTIONS = [get_single_srtm_image]

except ImportError:
    warnings.warn("ee_exporter requires earthengine-api, `pip install earthengine-api`")


def get_ee_task_list(key: str = "description") -> List[str]:
    """Gets a list of all active tasks in the EE task list."""
    task_list = ee.data.getTaskList()
    return [
        task[key]
        for task in tqdm(task_list, desc="Loading Earth Engine tasks")
        if task["state"] in ["READY", "RUNNING", "FAILED"]
    ]


def get_ee_task_amount(prefix: Optional[str] = None):
    """
    Gets amount of active tasks in Earth Engine.
    Args:
        prefix: Prefix to filter tasks.
    Returns:
        Amount of active tasks.
    """
    ee_prefix = None if prefix is None else ee_safe_str(prefix)
    amount = 0
    task_list = ee.data.getTaskList()
    for t in tqdm(task_list):
        valid_state = t["state"] in ["READY", "RUNNING"]
        if valid_state and (
            ee_prefix is None or t["description"].startswith(ee_prefix)
        ):
            amount += 1
    return amount


def get_cloud_tif_list(
    dest_bucket: str, prefix: str = "tifs", region: str = "us-central1"
) -> List[str]:
    """Gets a list of all cloud-free TIFs in a bucket."""
    storage = import_optional_dependency("google.cloud.storage")
    cloud_tif_list_iterator = storage.Client().list_blobs(dest_bucket, prefix=prefix)
    try:
        tif_list = [
            blob.name
            for blob in tqdm(
                cloud_tif_list_iterator, desc="Loading tifs already on Google Cloud"
            )
        ]
    except Exception as e:
        raise Exception(
            f"{e}\nPlease create the Google Cloud bucket: {dest_bucket}"
            + f"\nCommand: gsutil mb -l {region} gs://{dest_bucket}"
        )

    return tif_list


def make_combine_bands_function(bands):
    def combine_bands(current, previous):
        # Transforms an Image Collection with 1 band per Image into a single
        # Image with items as bands
        # Author: Jamie Vleeshouwer

        # Rename the band
        previous = ee.Image(previous)
        current = current.select(bands)
        # Append it to the result (Note: only return current item on first
        # element/iteration)
        return ee.Algorithms.If(
            ee.Algorithms.IsEqual(previous, None),
            current,
            previous.addBands(ee.Image(current)),
        )

    return combine_bands


def ee_safe_str(s: str):
    """Earth Engine descriptions only allow certain characters"""
    return s.replace(".", "-").replace("=", "-").replace("/", "-")[:100]


def create_ee_image(
    polygon: "ee.Geometry.Polygon",
    start_date: date,
    end_date: date,
    days_per_timestep: int = DAYS_PER_TIMESTEP,
):
    image_collection_list: List[ee.Image] = []
    cur_date = start_date
    cur_end_date = cur_date + timedelta(days=days_per_timestep)

    # first, we get all the S1 images in an exaggerated date range
    vv_imcol, vh_imcol = get_s1_image_collection(
        polygon, start_date - timedelta(days=31), end_date + timedelta(days=31)
    )

    while cur_end_date <= end_date:
        image_list: List[ee.Image] = []

        # first, the S1 image which gets the entire s1 collection
        image_list.append(
            get_single_s1_image(
                region=polygon,
                start_date=cur_date,
                end_date=cur_end_date,
                vv_imcol=vv_imcol,
                vh_imcol=vh_imcol,
            )
        )
        for image_function in DYNAMIC_IMAGE_FUNCTIONS:
            image_list.append(
                image_function(
                    region=polygon, start_date=cur_date, end_date=cur_end_date
                )
            )
        image_collection_list.append(ee.Image.cat(image_list))

        cur_date += timedelta(days=days_per_timestep)
        cur_end_date += timedelta(days=days_per_timestep)

    # now, we want to take our image collection and append the bands into a single image
    imcoll = ee.ImageCollection(image_collection_list)
    combine_bands_function = make_combine_bands_function(DYNAMIC_BANDS)
    img = ee.Image(imcoll.iterate(combine_bands_function))

    # finally, we add the SRTM image seperately since its static in time
    total_image_list: List[ee.Image] = [img]
    for static_image_function in STATIC_IMAGE_FUNCTIONS:
        total_image_list.append(static_image_function(region=polygon))

    return ee.Image.cat(total_image_list)


def get_ee_credentials():
    gcp_sa_key = os.environ.get("GCP_SA_KEY")
    if gcp_sa_key is not None:
        gcp_sa_email = json.loads(gcp_sa_key)["client_email"]
        print(f"Logging into EarthEngine with {gcp_sa_email}")
        return ee.ServiceAccountCredentials(gcp_sa_email, key_data=gcp_sa_key)
    else:
        print("Logging into EarthEngine with default credentials")
        return "persistent"


class EarthEngineExporter:
    """
    Export satellite data from Earth engine. It's called using the following
    script:
    ```
    from openmapflow.eo import EarthEngineExporter
    EarthEngineExporter(dest_bucket="bucket_name").export_for_labels(df)
    ```
    :param check_ee: Whether to check Earth Engine before exporting
    :param check_gcp: Whether to check Google Cloud Storage before exporting,
        google-cloud-storage must be installed.
    :param credentials: The credentials to use for the export. If not specified,
        the default credentials will be used
    :param dest_bucket: The bucket to export to, google-cloud-storage must be installed.
    """

    def __init__(
        self,
        dest_bucket: str,
        check_ee: bool = False,
        check_gcp: bool = False,
        credentials=None,
    ) -> None:
        self.dest_bucket = dest_bucket
        ee.Initialize(credentials=credentials if credentials else get_ee_credentials())
        self.check_ee = check_ee
        self.ee_task_list = get_ee_task_list() if self.check_ee else []
        self.check_gcp = check_gcp
        self.cloud_tif_list = get_cloud_tif_list(dest_bucket) if self.check_gcp else []

    def _export_for_polygon(
        self,
        polygon: "ee.Geometry.Polygon",
        polygon_identifier: Union[int, str],
        start_date: date,
        end_date: date,
        test: bool = False,
        file_dimensions: Optional[int] = None,
    ) -> bool:

        filename = str(polygon_identifier)

        # Description of the export cannot contain certrain characters
        description = ee_safe_str(filename)

        if (
            f"{filename}.tif" in self.cloud_tif_list
            and f"tifs/{filename}.tif" in self.cloud_tif_list
        ):
            return True

        # Check if task is already started in EarthEngine
        if not test and description in self.ee_task_list:
            return True

        if len(self.ee_task_list) >= 3000:
            return False

        img = create_ee_image(polygon, start_date, end_date)

        # and finally, export the image
        if not test:
            # If training data make sure it goes in the tifs folder
            filename = f"tifs/{filename}"

        try:
            ee.batch.Export.image.toCloudStorage(
                bucket=self.dest_bucket,
                fileNamePrefix=filename,
                image=img.clip(polygon),
                description=description,
                scale=10,
                region=polygon,
                maxPixels=1e13,
                fileDimensions=file_dimensions,
            ).start()
            self.ee_task_list.append(description)
        except ee.ee_exception.EEException as e:
            print(f"Task not started! Got exception {e}")

        return True

    def export_for_bbox(
        self,
        bbox: BBox,
        bbox_name: str,
        start_date: date,
        end_date: date,
        metres_per_polygon: Optional[int] = 10000,
        file_dimensions: Optional[int] = None,
    ) -> Dict[str, bool]:
        if start_date > end_date:
            raise ValueError(f"Start date {start_date} is after end date {end_date}")

        ee_bbox = EEBoundingBox.from_bounding_box(bounding_box=bbox, padding_metres=0)
        if metres_per_polygon is not None:
            regions = ee_bbox.to_polygons(metres_per_patch=metres_per_polygon)
            ids = [f"batch_{i}/{i}" for i in range(len(regions))]
        else:
            regions = [ee_bbox.to_ee_polygon()]
            ids = ["batch/0"]

        return_obj = {}
        for identifier, region in zip(ids, regions):
            return_obj[identifier] = self._export_for_polygon(
                polygon=region,
                polygon_identifier=f"{bbox_name}/{identifier}",
                start_date=start_date,
                end_date=end_date,
                file_dimensions=file_dimensions,
                test=True,
            )
        return return_obj

    def export_for_labels(
        self,
        labels: pd.DataFrame,
        num_labelled_points: int = 3000,
        surrounding_metres: int = 80,
    ) -> None:

        for expected_column in [START, END, LAT, LON]:
            assert expected_column in labels

        labels[START] = pd.to_datetime(labels[START]).dt.date
        labels[END] = pd.to_datetime(labels[END]).dt.date

        exports_started = 0
        print(f"Exporting {len(labels)} labels: ")

        for _, row in tqdm(labels.iterrows(), desc="Exporting", total=len(labels)):
            ee_bbox = EEBoundingBox.from_centre(
                mid_lat=row[LAT],
                mid_lon=row[LON],
                surrounding_metres=surrounding_metres,
            )

            export_started = self._export_for_polygon(
                polygon=ee_bbox.to_ee_polygon(),
                polygon_identifier=ee_bbox.get_identifier(row[START], row[END]),
                start_date=row[START],
                end_date=row[END],
                test=False,
            )
            if export_started:
                exports_started += 1
                if (
                    num_labelled_points is not None
                    and exports_started >= num_labelled_points
                ):
                    print(f"Started {exports_started} exports. Ending export")
                    return None


class EarthEngineAPI:
    """
    Fetch satellite data from Earth engine by URL.
    :param credentials: The credentials to use for the export. If not specified,
        the default credentials will be used
    """

    def __init__(self, credentials=None) -> None:
        ee.Initialize(
            credentials if credentials else get_ee_credentials(),
            opt_url="https://earthengine-highvolume.googleapis.com",
        )

    def get_ee_url(self, lat, lon, start_date, end_date):
        ee_bbox = EEBoundingBox.from_centre(
            mid_lat=lat,
            mid_lon=lon,
            surrounding_metres=80,
        ).to_ee_polygon()
        img = create_ee_image(ee_bbox, start_date, end_date)
        return img.getDownloadURL(
            {
                "region": ee_bbox,
                "scale": 10,
                "filePerBand": False,
                "format": "GEO_TIFF",
            }
        )
