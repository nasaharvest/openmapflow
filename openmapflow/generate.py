import argparse
import shutil
import tarfile

from pathlib import Path
from openmapflow.constants import CONFIG_FILE, LIBRARY_DIR


def allow_write(p, force=False):
    if force or not Path(p).exists():
        return True
    p = Path(*Path(p).parts[-4:])
    overwrite = input(f"{str(p)} already exists. Overwrite? (y/[n]): ")
    return overwrite.lower() == "y"


def openmapflow_config_from_args(args):
    return f"""version: 0.0.1
project: {args.name}
description: {args.description}
gcloud:
    project_id: {args.gcloud_project_id}
    location: {args.gcloud_location}
    bucket_labeled_tifs: {args.gcloud_bucket_labeled_tifs}
"""


def openmapflow_config_write(openmapflow_str: str, force: bool):
    if allow_write(CONFIG_FILE, force):
        with open(CONFIG_FILE, "w") as f:
            f.write(openmapflow_str)


def copy_datasets_py_file(LIBRARY_DIR, PROJECT_ROOT, force: bool):
    if allow_write("datasets.py", force):
        shutil.copy(
            str(LIBRARY_DIR / "example_datasets.py"), str(PROJECT_ROOT / "datasets.py")
        )


def create_data_dirs(dp, force):
    for p in [dp.FEATURES, dp.RAW_LABELS, dp.PROCESSED_LABELS, dp.MODELS]:
        if allow_write(p, force):
            Path(p).mkdir(parents=True, exist_ok=True)

    if allow_write(dp.COMPRESSED_FEATURES):
        with tarfile.open(dp.COMPRESSED_FEATURES, "w:gz") as tar:
            tar.add(dp.FEATURES, arcname=Path(dp.FEATURES).name)


def fill_in_action(src_yml_path, dest_yml_path, sub_paths, sub_cd, sub_project):
    with src_yml_path.open("r") as f:
        content = f.read()
    content = content.replace("<PATHS>", sub_paths)
    content = content.replace("<CD>", sub_cd)
    content = content.replace("<PROJECT>", sub_project)
    with dest_yml_path.open("w") as f:
        f.write(content)


def create_github_actions(LIBRARY_DIR, PROJECT_ROOT, PROJECT, dp, force):
    possible_git_roots = [PROJECT_ROOT, PROJECT_ROOT.parent]
    try:
        git_root = next(r for r in possible_git_roots if (r / ".git").exists())
    except StopIteration:
        raise FileExistsError(
            f"Could not find .git in {str(PROJECT_ROOT)} or its parent"
        )

    src_deploy_yml_path = LIBRARY_DIR / "templates/openmapflow-github-deploy.yml"
    src_test_yml_path = LIBRARY_DIR / "templates/openmapflow-github-test.yml"
    dest_deploy_yml_path = git_root / f".github/workflows/{PROJECT}-deploy.yml"
    dest_test_yml_path = git_root / f".github/workflows/{PROJECT}-test.yml"
    is_subdir = git_root != PROJECT_ROOT

    if allow_write(dest_deploy_yml_path, force):
        fill_in_action(
            src_yml_path=src_deploy_yml_path,
            dest_yml_path=dest_deploy_yml_path,
            sub_paths=f"{f'{PROJECT}/' if is_subdir else ''}{dp.MODELS}.dvc",
            sub_cd=f"cd {PROJECT}" if is_subdir else "",
            sub_project=PROJECT,
        )

    if allow_write(dest_test_yml_path, force):
        fill_in_action(
            src_yml_path=src_test_yml_path,
            dest_yml_path=dest_test_yml_path,
            sub_paths=f"{f'{PROJECT}/' if is_subdir else ''}data/**",
            sub_cd=f"cd {PROJECT}" if is_subdir else "",
            sub_project=PROJECT,
        )


long_line = "########################################################################"

dvc_instructions = f"""{long_line}\nDVC Setup Instructions\n{long_line}
dvc (https://dvc.org/) is used to manage data. To setup run:
    # Initializes dvc (use --subdir if in subdirectory)
    dvc init 
    
    dvc add <DVC_FILES>
    
    # https://dvc.org/doc/user-guide/setup-google-drive-remote
    dvc remote add -d gdrive gdrive://<last part of gdrive folder url>
    
    # Push files to remote storage
    dvc push 
"""

colab_url = "https://colab.research.google.com"
nb_home = f"{colab_url}/github/nasaharvest/openmapflow/blob/main/openmapflow/notebooks"
using_openampflow = f"""{long_line}\nUsing OpenMapFlow\n{long_line}
After dvc is setup, push your changes to Github and you'll be able to run Colab notebooks:
1) Adding new data\n{nb_home}/new_data.ipynb
2) Training a model\n{nb_home}/train.ipynb
3) Creating a map\n{nb_home}/create_map.ipynb

Notebooks can also be run locally:
cp -r $(openmapflow-dir)/notebooks .
jupyter notebook
"""

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate OpenMapFlow project.")
    parser.add_argument(
        "--name", type=str, help="Project name", default=Path.cwd().name
    )
    parser.add_argument(
        "--description", type=str, help="Project description", default=""
    )
    parser.add_argument(
        "--gcloud_project_id",
        type=str,
        help="Google Cloud Project ID",
        default="~",
    )
    parser.add_argument(
        "--gcloud_location",
        type=str,
        help="Google Cloud Location (default: us-central-1)",
        default="us-central-1",
    )
    parser.add_argument(
        "--gcloud_bucket_labeled_tifs",
        type=str,
        help="Google Cloud Bucket for labeled tifs (default: crop-mask-tifs2)",
        default="crop-mask-tifs2",
    )
    parser.add_argument("--force", action="store_true", help="Force overwrite")
    args = parser.parse_args()

    print("1/7 Parsing arguments")
    openmapflow_str = openmapflow_config_from_args(args)

    print(f"2/7 Writing {CONFIG_FILE}")
    openmapflow_config_write(openmapflow_str, force=args.force)

    # Can only import when openmapflow.yaml is available
    from openmapflow.config import PROJECT_ROOT, PROJECT, DataPaths as dp  # noqa E402

    print("3/7 Copying datasets.py file")
    copy_datasets_py_file(LIBRARY_DIR, PROJECT_ROOT, args.force)

    print("4/7 Creating data directories")
    create_data_dirs(dp=dp, force=args.force)

    print("5/7 Creating Github Actions")
    create_github_actions(LIBRARY_DIR, PROJECT_ROOT, PROJECT, dp, args.force)

    print("6/7 Printing dvc instructions")
    dvc_files = [
        dp.RAW_LABELS,
        dp.PROCESSED_LABELS,
        dp.FEATURES,
        dp.COMPRESSED_FEATURES,
        dp.MODELS,
    ]
    print(dvc_instructions.replace("<DVC_FILES>", " ".join(dvc_files)))

    print("7/7 Ready to go!")
    print(using_openampflow)
