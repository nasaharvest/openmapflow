from pathlib import Path
import yaml


if (Path.cwd() / "openmapflow.yaml").exists():
    project_root = Path.cwd()
elif (Path.cwd().parent / "openmapflow.yaml").exists():
    project_root = Path.cwd().parent
else:
    raise FileExistsError(
        f"openmapflow.yml not found in {Path.cwd()} or {Path.cwd().parent}"
    )

if not (project_root / "data").exists():
    raise FileExistsError(f"{project_root}/data was not found.")


with (project_root / "openmapflow.yaml").open() as f:
    config_yml = yaml.safe_load(f)

relative_paths = {k: f"data/{v}" for k, v in config_yml["data_paths"].items()}
full_paths = {k: project_root / v for k, v in relative_paths.items()}
tif_bucket_name = config_yml["labeled_tifs_bucket"]
