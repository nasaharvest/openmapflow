from pathlib import Path
import yaml


cwd = Path.cwd()

config_path = cwd / "openmapflow.yaml"
if not config_path.exists():
    raise FileExistsError(
        f"openmapflow.yml not found in {cwd}. Please change the current working directory."
    )

if not (cwd / "data").exists():
    raise FileExistsError(f"{cwd}/data was not found.")

with config_path.open() as f:
    config_yml = yaml.safe_load(f)

relative_paths = {k: f"data/{v}" for k, v in config_yml["data_paths"].items()}
full_paths = {k: cwd / v for k, v in relative_paths.items()}
tif_bucket_name = config_yml["labeled_tifs_bucket"]
