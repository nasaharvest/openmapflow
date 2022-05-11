import argparse
import shutil
import tarfile

from pathlib import Path
from openmapflow.constants import CONFIG_FILE, LIBRARY_DIR


def allow_write(p, overwrite=False):
    if overwrite or not Path(p).exists():
        return True
    p = Path(*Path(p).parts[-4:])
    overwrite = input(f"  {str(p)} already exists. Overwrite? (y/[n]): ")
    return overwrite.lower() == "y"


def create_openmapflow_config(overwrite: bool):
    if not allow_write(CONFIG_FILE, overwrite):
        return
    cwd = Path.cwd()
    project_name = input(f"  Project name [{cwd.stem}]: ") or cwd.stem
    description = input("  Description: ")
    gcloud_project_id = input("  GCloud project ID: ")
    gcloud_location = input("  GCloud location [us-central1]: ") or "us-central1"
    gcloud_bucket_labeled_tifs = (
        input("  GCloud bucket labeled tifs [crop-mask-tifs2]: ") or "crop-mask-tifs2"
    )
    openmapflow_str = (
        "version: 0.0.1"
        + f"\nproject: {project_name}"
        + f"\ndescription: {description}"
        + "\ngcloud:"
        + f"\n    project_id: {gcloud_project_id}"
        + f"\n    location: {gcloud_location}"
        + f"\n    bucket_labeled_tifs: {gcloud_bucket_labeled_tifs}"
    )

    with open(CONFIG_FILE, "w") as f:
        f.write(openmapflow_str)


def copy_datasets_py_file(LIBRARY_DIR, PROJECT_ROOT, overwrite: bool):
    if allow_write("datasets.py", overwrite):
        shutil.copy(
            str(LIBRARY_DIR / "example_datasets.py"), str(PROJECT_ROOT / "datasets.py")
        )


def create_data_dirs(dp, overwrite):
    for p in [dp.FEATURES, dp.RAW_LABELS, dp.PROCESSED_LABELS, dp.MODELS]:
        if allow_write(p, overwrite):
            Path(p).mkdir(parents=True, exist_ok=True)

    if allow_write(dp.COMPRESSED_FEATURES):
        with tarfile.open(dp.COMPRESSED_FEATURES, "w:gz") as tar:
            tar.add(dp.FEATURES, arcname=Path(dp.FEATURES).name)


def fill_in_action(src_yml_path, dest_yml_path, sub_paths, sub_cd):
    with src_yml_path.open("r") as f:
        content = f.read()
    content = content.replace("<PATHS>", sub_paths)
    content = content.replace("<CD>", sub_cd)
    with dest_yml_path.open("w") as f:
        f.write(content)


def create_github_actions(LIBRARY_DIR, PROJECT_ROOT, PROJECT, dp, overwrite):
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

    if allow_write(dest_deploy_yml_path, overwrite):
        fill_in_action(
            src_yml_path=src_deploy_yml_path,
            dest_yml_path=dest_deploy_yml_path,
            sub_paths=f"{f'{PROJECT}/' if is_subdir else ''}{dp.MODELS}.dvc",
            sub_cd=f"cd {PROJECT}" if is_subdir else "",
        )

    if allow_write(dest_test_yml_path, overwrite):
        fill_in_action(
            src_yml_path=src_test_yml_path,
            dest_yml_path=dest_test_yml_path,
            sub_paths=f"{f'{PROJECT}/' if is_subdir else ''}data/**",
            sub_cd=f"cd {PROJECT}" if is_subdir else "",
        )


long_line = "########################################################################"

dvc_instructions = f"""{long_line}\nDVC Setup Instructions\n{long_line}
dvc (https://dvc.org/) is used to manage data. To setup run:
    # Initializes dvc (use --subdir if in subdirectory)
    dvc init\n
    dvc add <DVC_FILES>\n
    # https://dvc.org/doc/user-guide/setup-google-drive-remote
    dvc remote add -d gdrive gdrive://<last part of gdrive folder url>\n
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
openmapflow copy notebooks .
jupyter notebook
"""

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate OpenMapFlow project.")
    parser.add_argument("--overwrite", action="store_true", help="overwrite overwrite")
    args = parser.parse_args()

    n = 6

    print(f"1/{n} Writing openmapflow")
    create_openmapflow_config(args.overwrite)

    # Can only import when openmapflow.yaml is available
    from openmapflow.config import PROJECT_ROOT, PROJECT, DataPaths as dp  # noqa E402

    print(f"2/{n} Copying datasets.py file")
    copy_datasets_py_file(LIBRARY_DIR, PROJECT_ROOT, args.overwrite)

    print(f"3/{n} Creating data directories")
    create_data_dirs(dp=dp, overwrite=args.overwrite)

    print(f"4/{n} Creating Github Actions")
    create_github_actions(LIBRARY_DIR, PROJECT_ROOT, PROJECT, dp, args.overwrite)

    print(f"5/{n} Printing dvc instructions")
    dvc_files = [
        dp.RAW_LABELS,
        dp.PROCESSED_LABELS,
        dp.FEATURES,
        dp.COMPRESSED_FEATURES,
        dp.MODELS,
    ]
    print(dvc_instructions.replace("<DVC_FILES>", " ".join(dvc_files)))

    print(f"6/{n} Ready to go!")
    print(using_openampflow)
