# Config
export MAP_KEY="Kenya_maize/min_lat=-4.8_min_lon=33.9_max_lat=5.5_max_lon=42.0_dates=2021-02-01_2022-02-01_all"
export BUCKET_PREDS="gs://crop-type-preds"
export BUCKET_PREDS_MERGED="gs://crop-type-preds-merged"

# Setup
export PREFIX=$(python3 -c "import os; print(os.environ['MAP_KEY'].replace('/', '_'))")
mkdir $PREFIX-preds
mkdir $PREFIX-vrts
sudo apt-get install wget
sudo apt-get install python3-gdal
sudo apt-get install gdal-bin

# Download data
gsutil -m cp -n -r $BUCKET_PREDS/$MAP_KEY* $PREFIX-preds

# Build VRT
cat > buildvrt.py << EOL
import os
import re
from glob import glob
from pathlib import Path

PREFIX = os.environ["PREFIX"]
SEP = "-"

print("Building vrt for each batch")
for d in glob(f"{PREFIX}{SEP}preds/*/*/"):
    if "batch" not in d:
        continue
    match = re.search("batch_(.*?)/", d)
    if match:
        i = int(match.group(1))
    else:
        raise ValueError(f"Cannot parse i from {d}")
    vrt_file = Path(f"{PREFIX}{SEP}vrts/{i}.vrt")
    if not vrt_file.exists():
        os.system(f"gdalbuildvrt {vrt_file} {d}*")
print("Building full vrt")
os.system(f"gdalbuildvrt {PREFIX}{SEP}final.vrt {PREFIX}{SEP}vrts/*.vrt")
EOL

python3 buildvrt.py

# Translate VRT to GeoTIFF
gdal_translate -a_srs EPSG:4326 -of GTiff $PREFIX-final.vrt $PREFIX-final.tif #> /dev/null 2>&1 &

# Check that job is running
jobs -l

# Upload GEOTIF
gsutil cp $PREFIX-final.tif $BUCKET_PREDS_MERGED/$MAP_KEY-final.tif