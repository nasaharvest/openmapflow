####################################################################################################################################
# This script downloads OpenMapFlow predictions, merges them, and uploads the merged tif file to Google Cloud Storage.
# It is intended to be run in a Google Cloud VM (with all Cloud APIs enabled) when large maps cannot be merged in Colab.
#
# Running script in the backgroud:
# sh merging.sh > out.txt 2>&1 &
# Checking logs
# tail -100 out.txt 
####################################################################################################################################
sudo apt-get install python3-gdal -y
sudo apt-get install gdal-bin -y

# Config
export MAP_KEY="Togo_crop-mask_2019_February/min_lat=7.23_min_lon=31.65_max_lat=8.23_max_lon=32.65_dates=2019-02-01_2020-02-01_all"
export BUCKET_PREDS="gs://crop-mask-example-preds"
export BUCKET_PREDS_MERGED="gs://crop-mask-example-preds-merged"

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