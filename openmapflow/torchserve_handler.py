import os
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from google.cloud import storage
from ts.torch_handler.base_handler import BaseHandler

from openmapflow.inference import Inference


def get_bucket_name(uri: str) -> str:
    """Gets bucket name from Google Cloud Storage URI"""
    if not uri.startswith("gs://"):
        raise ValueError(f"{uri} is not a Google Cloud Storage URI")
    parts = Path(uri).parts
    if len(parts) < 2:
        raise ValueError(f"{uri} is not complete, expecting gs://bucket/path")
    return parts[1]


def get_path(uri: str, replace_filename: Optional[str] = None) -> str:
    """Gets str path from Google Cloud Storage URI"""
    if not uri.startswith("gs://"):
        raise ValueError(f"{uri} is not a Google Cloud Storage URI")
    parts = Path(uri).parts
    if len(parts) < 3:
        raise ValueError(f"{uri} is not complete, expecting gs://bucket/path")
    if replace_filename is not None:
        if len(parts) == 3:  # uri is just gs://bucket/path
            return replace_filename
        return "/".join(parts[2:-1]) + "/" + replace_filename
    return "/".join(parts[2:])


def download_file(uri: str) -> str:
    """
    Downloads file from Google Cloud Storage bucket and returns local file path
    Args:
        uri (str):  Path to file on Google Cloud Storage bucket
    """
    blob = storage.Client().bucket(get_bucket_name(uri)).blob(get_path(uri))
    if blob.exists():
        print(f"HANDLER: Verified {uri} exists.")
    else:
        raise ValueError(f"HANDLER ERROR: {uri} does not exist.")

    local_path = str(Path(tempfile.gettempdir()) / Path(uri).name)
    blob.download_to_filename(local_path)
    if not Path(local_path).exists():
        raise FileExistsError(f"HANDLER: {uri} from storage was not downloaded")
    print(f"HANDLER: Verified file downloaded to {local_path}")
    return local_path


def upload_file(bucket_name: str, local_path: Path, src_uri: str) -> str:
    """
    Uploads file to Google Cloud Storage bucket using match directory
    structure of src_uri
    Args:
        bucket_name (str): Name of Google Cloud Storage bucket
        local_path (Path):  Path to local file
        src_uri (Path):  Gcloud path of source file
    """
    if not local_path.exists():
        raise FileNotFoundError(f"HANDLER: {local_path} does not exist")
    cloud_dest_path = get_path(uri=src_uri, replace_filename=local_path.name)
    blob = storage.Client().bucket(bucket_name).blob(cloud_dest_path)
    blob.upload_from_filename(str(local_path))
    return f"gs://{bucket_name}/{cloud_dest_path}"


class ModelHandler(BaseHandler):
    """
    A custom model handler implementation.
    The basehandler calls initialize() on server start up and
    preprocess(), inference(), and postprocess() on each request.
    """

    def __init__(self):
        print("HANDLER: Starting up handler")
        super().__init__()

    def initialize(self, context):
        """Sets up torchserve handler with destination bucket name and inference module"""
        super().initialize(context)
        properties = context.system_properties
        model_dir = properties.get("model_dir")
        sys.path.append(model_dir)

        normalizing_dict = None
        if hasattr(self.model, "normalizing_dict_jit"):
            normalizing_dict = {
                k: np.array(v) for k, v in self.model.normalizing_dict_jit.items()
            }

        batch_size = 64
        if hasattr(self.model, "batch_size"):
            batch_size = self.model.batch_size

        self.inference_module = Inference(
            model=self.model, normalizing_dict=normalizing_dict, batch_size=batch_size
        )
        self.dest_bucket_name: str = os.environ["DEST_BUCKET"]
        print(f"HANDLER: Dest bucket: {self.dest_bucket_name}")

    def preprocess(self, data) -> str:
        """Extracts the URI from the request"""
        print(data)
        print("HANDLER: Starting preprocessing")
        try:
            uri = next(q["uri"].decode() for q in data if "uri" in q)
        except Exception:
            raise ValueError("'uri' not found.")
        return uri

    def inference(self, data: str, *args, **kwargs) -> Tuple[str, str]:
        """
        1. Downloads file from source Google Cloud Storage bucket
        2. Runs model inference on source tif file
        3. Uploads prediction file to destination Google Cloud Storage bucket
        """
        uri = data
        local_src_path = download_file(uri)
        local_dest_path = Path(tempfile.gettempdir() + f"/pred_{Path(uri).stem}.nc")

        print("HANDLER: Starting inference")
        self.inference_module.run(local_path=local_src_path, dest_path=local_dest_path)
        print("HANDLER: Completed inference")
        dest_uri = upload_file(
            bucket_name=self.dest_bucket_name, local_path=local_dest_path, src_uri=uri
        )
        print(f"HANDLER: Uploaded to {dest_uri}")
        return uri, dest_uri

    def postprocess(self, data: Tuple[str, str]) -> List[Dict[str, str]]:
        """Returns URIs in torchserve format"""
        return [{"src_uri": data[0], "dest_uri": data[1]}]
