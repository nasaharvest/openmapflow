import os
import re
from collections import defaultdict
from glob import glob
from pathlib import Path

import ee
from google.cloud import storage
from tqdm.notebook import tqdm

from openmapflow.config import GCLOUD_PROJECT_ID
from openmapflow.config import BucketNames
from openmapflow.config import BucketNames as bn
from openmapflow.labeled_dataset import bbox_from_str

#######################################################
# Status functions
#######################################################
bbox_regex = (
    r".*min_lat=-?\d*\.?\d*_min_lon=-?\d*\.?\d*_max_lat=-?\d*\.?\d*_max_lon=-?\d*\.?\d*_"
    + r"dates=\d{4}-\d{2}-\d{2}_\d{4}-\d{2}-\d{2}.*?\/"
)


def get_available_bboxes(buckets_to_check=[BucketNames.INFERENCE_TIFS]):
    client = storage.Client()
    previous_matches = []
    available_bboxes = []
    for bucket_name in buckets_to_check:
        blobs = client.list_blobs(bucket_or_name=bucket_name)
        for blob in blobs:
            match = re.search(bbox_regex, blob.name)
            if not match:
                continue
            p = match.group()
            if p not in previous_matches:
                previous_matches.append(p)
                available_bboxes.append(bbox_from_str(f"gs://{bucket_name}/{p}"))
    return available_bboxes


def get_ee_task_amount(prefix=None):
    amount = 0
    task_list = ee.data.getTaskList()
    for t in tqdm(task_list):
        if t["state"] in ["READY", "RUNNING"]:
            if prefix and prefix in t["description"]:
                amount += 1
            else:
                amount += 1
    return amount


def get_gcs_file_dict_and_amount(bucket_name, prefix):
    blobs = storage.Client(project=GCLOUD_PROJECT_ID).list_blobs(
        bucket_name, prefix=prefix
    )
    files_dict = defaultdict(lambda: [])
    amount = 0
    for blob in tqdm(blobs, desc=f"From {bucket_name}"):
        p = Path(blob.name)
        files_dict[str(p.parent)].append(p.stem.replace("pred_", ""))
        amount += 1
    return files_dict, amount


def get_gcs_file_amount(bucket_name, prefix):
    gcs_files = storage.Client(project=GCLOUD_PROJECT_ID).list_blobs(
        bucket_name, prefix=prefix
    )

    return len(list(gcs_files))


def get_status(model_name_version):
    print("-----------------------------------------------------------------------")
    print(model_name_version)
    print("-----------------------------------------------------------------------")
    ee_task_amount = get_ee_task_amount(prefix=model_name_version.replace("/", "-"))
    tifs_amount = get_gcs_file_amount(bn.INFERENCE_TIFS, prefix=model_name_version)
    predictions_amount = get_gcs_file_amount(bn.PREDS, prefix=model_name_version)
    print(f"Earth Engine tasks: {ee_task_amount}")
    print(f"Data available: {tifs_amount}")
    print(f"Predictions: {predictions_amount}")
    return ee_task_amount, tifs_amount, predictions_amount


#######################################################
# Inference functions
#######################################################
def find_missing_predictions(model_name_version, verbose=False):
    print("Addressing missing files")
    tif_files, tif_amount = get_gcs_file_dict_and_amount(
        bn.INFERENCE_TIFS, prefix=model_name_version
    )
    pred_files, pred_amount = get_gcs_file_dict_and_amount(
        bn.PREDS, prefix=model_name_version
    )
    missing = {}
    for full_k in tqdm(tif_files.keys(), desc="Missing files"):
        if full_k not in pred_files:
            diffs = tif_files[full_k]
        else:
            diffs = list(set(tif_files[full_k]) - set(pred_files[full_k]))
        if len(diffs) > 0:
            missing[full_k] = diffs

    batches_with_issues = len(missing.keys())
    if verbose:
        print("-----------------------------------------------------------------------")
        print(model_name_version)
        print("-----------------------------------------------------------------------")
    if batches_with_issues > 0:
        print(
            f"\u2716 {batches_with_issues}/{len(tif_files.keys())} "
            + f"batches have a total {tif_amount - pred_amount} missing predictions"
        )
        if verbose:
            for batch, files in missing.items():
                print("\t--------------------------------------------------")
                print(f"\t{Path(batch).stem}: {len(files)}")
                print("\t--------------------------------------------------")
                [print(f"\t{f}") for f in files]
    else:
        print("\u2714 all files in each batch match")
    return missing


def make_new_predictions(missing):
    bucket = storage.Client(project=GCLOUD_PROJECT_ID).bucket(bn.INFERENCE_TIFS)
    for batch, files in tqdm(missing.items(), desc="Going through batches"):
        for file in tqdm(files, desc="Renaming files", leave=False):
            blob_name = f"{batch}/{file}.tif"
            blob = bucket.blob(blob_name)
            if blob.exists():
                new_blob_name = f"{batch}/{file}-retry1.tif"
                bucket.rename_blob(blob, new_blob_name)
            else:
                print(f"Could not find: {blob_name}")


#######################################################
# Map making functions
#######################################################
def gdal_cmd(cmd_type: str, in_file: str, out_file: str, msg=None, print_cmd=False):
    if cmd_type == "gdalbuildvrt":
        cmd = f"gdalbuildvrt {out_file} {in_file}"
    elif cmd_type == "gdal_translate":
        cmd = f"gdal_translate -a_srs EPSG:4326 -of GTiff {in_file} {out_file}"
    else:
        raise NotImplementedError(f"{cmd_type} not implemented.")
    if msg:
        print(msg)
    if print_cmd:
        print(cmd)
    os.system(cmd)


def build_vrt(prefix):
    # Build vrts for each batch of predictions
    print("Building vrt for each batch")
    for d in tqdm(glob(f"{prefix}_preds/*/*/")):
        if "batch" not in d:
            continue

        match = re.search("batch_(.*?)/", d)
        if match:
            i = int(match.group(1))
        else:
            raise ValueError(f"Cannot parse i from {d}")
        vrt_file = Path(f"{prefix}_vrts/{i}.vrt")
        if not vrt_file.exists():
            gdal_cmd(cmd_type="gdalbuildvrt", in_file=f"{d}*", out_file=str(vrt_file))

    gdal_cmd(
        cmd_type="gdalbuildvrt",
        in_file=f"{prefix}_vrts/*.vrt",
        out_file=f"{prefix}_final.vrt",
        msg="Building full vrt",
    )
