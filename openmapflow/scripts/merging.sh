####################################################################################################################################
# This script downloads OpenMapFlow predictions, merges them, and uploads the merged tif file to Google Cloud Storage.
# It is intended to be run in a Google Cloud VM when large maps cannot be merged in Colab.
####################################################################################################################################
sudo apt-get install python3-gdal -y
sudo apt-get install gdal-bin -y

# Config
export MAP_KEY="Kenya_maize/min_lat=-4.8_min_lon=33.9_max_lat=5.5_max_lon=42.0_dates=2021-02-01_2022-02-01_all"
export BUCKET_PREDS="gs://crop-type-preds"
export BUCKET_PREDS_MERGED="gs://crop-type-preds-merged"

# Setup
export PREFIX=$(python3 -c "import os; print(os.environ['MAP_KEY'].replace('/', '_'))")
export EE_PREFIX=$(python3 -c "import os; print(os.environ['PREFIX'].replace('.', '-').replace('=', '-').replace('/', '-')[:100])")
export VRT_FILE=$PREFIX-final.vrt
export TIF_FILE=$PREFIX-final.tif
export GCLOUD_DEST=$BUCKET_PREDS_MERGED/$MAP_KEY-final.tif

mkdir $PREFIX-preds
mkdir $PREFIX-vrts

# Download data
gsutil -m cp -n -r $BUCKET_PREDS/$MAP_KEY* $PREFIX-preds

# Build VRT
cat > buildvrts.py << EOL
import os
import re
from glob import glob
from pathlib import Path

PREFIX = os.environ["PREFIX"]
print("Building vrt for each batch")
for d in glob(f"{PREFIX}-preds/*/*/"):
    if "batch" not in d:
        continue
    match = re.search("batch_(.*?)/", d)
    if match:
        i = int(match.group(1))
    else:
        raise ValueError(f"Cannot parse i from {d}")
    os.system(f"gdalbuildvrt {PREFIX}-vrts/{i}.vrt {d}*")
EOL

python3 buildvrts.py
gdalbuildvrt $VRT_FILE $PREFIX-vrts/*.vrt

# Translate VRT to GeoTIFF
gdal_translate -a_srs EPSG:4326 -of GTiff $VRT_FILE $TIF_FILE #> /dev/null 2>&1 &

# Check that job is running
jobs -l

# Upload GEOTIF
gsutil cp $TIF_FILE $GCLOUD_DEST

# EarthEngine 
echo "On ee authenticated machine run: "
echo "earthengine upload image --asset_id users/<ee-user>/$EE_PREFIX -ts <start-date> -te <end-date> $GCLOUD_DEST"