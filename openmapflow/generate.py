import argparse
import os
import shutil
import tarfile
from pathlib import Path

from openmapflow.constants import (
    CONFIG_FILE,
    TEMPLATE_DATASETS,
    TEMPLATE_DEPLOY_YML,
    TEMPLATE_TEST_YML,
)


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


def copy_datasets_py_file(PROJECT_ROOT, overwrite: bool):
    if allow_write("datasets.py", overwrite):
        shutil.copy(str(TEMPLATE_DATASETS), str(PROJECT_ROOT / "datasets.py"))


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


def get_git_root(PROJECT_ROOT):
    possible_git_roots = [PROJECT_ROOT, PROJECT_ROOT.parent]
    try:
        return next(r for r in possible_git_roots if (r / ".git").exists())
    except StopIteration:
        raise FileExistsError(
            f"Could not find .git in {str(PROJECT_ROOT)} or its parent"
        )


def create_github_actions(git_root, is_subdir, PROJECT, dp, overwrite):
    dest_deploy_yml_path = git_root / f".github/workflows/{PROJECT}-deploy.yaml"
    dest_test_yml_path = git_root / f".github/workflows/{PROJECT}-test.yaml"
    if allow_write(dest_deploy_yml_path, overwrite):
        fill_in_action(
            src_yml_path=TEMPLATE_DEPLOY_YML,
            dest_yml_path=dest_deploy_yml_path,
            sub_paths=f"{f'{PROJECT}/' if is_subdir else ''}{dp.MODELS}.dvc",
            sub_cd=f"cd {PROJECT}" if is_subdir else "",
        )

    if allow_write(dest_test_yml_path, overwrite):
        fill_in_action(
            src_yml_path=TEMPLATE_TEST_YML,
            dest_yml_path=dest_test_yml_path,
            sub_paths=f"{f'{PROJECT}/' if is_subdir else ''}data/**",
            sub_cd=f"cd {PROJECT}" if is_subdir else "",
        )


def print_and_run(cmd):
    print(f"{cmd}")
    os.system(cmd)


def setup_dvc(PROJECT_ROOT, is_subdir, dp):
    if (PROJECT_ROOT / ".dvc").exists():
        print(f"  {PROJECT_ROOT}/.dvc already exists. Skipping.")
        return

    if is_subdir:
        print_and_run("dvc init --subdir")
    else:
        print_and_run("dvc init")

    dvc_files = [dp.RAW_LABELS, dp.PROCESSED_LABELS, dp.COMPRESSED_FEATURES, dp.MODELS]
    print_and_run("dvc add " + " ".join([dvc_files]))

    with open("data/.gitignore", "a") as f:
        f.write("/features")

    print("dvc stores data in remote storage (s3, gcs, gdrive, etc)")
    print("https://dvc.org/doc/command-reference/remote/add#supported-storage-types")
    option = input("a) Setup gdrive / b) Exit and setup own remote [a]/b: ")
    if option.lower() == "b":
        return

    print("We'll follow: https://dvc.org/doc/user-guide/setup-google-drive-remote")
    gdrive_url = input("Last part of gdrive folder url: ")
    if gdrive_url:
        print_and_run(f"dvc remote add -d gdrive gdrive://{gdrive_url}")
        print_and_run("dvc push")
    else:
        print("You must enter a valid gdrive url")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate OpenMapFlow project.")
    parser.add_argument("--overwrite", action="store_true", help="overwrite overwrite")
    args = parser.parse_args()

    n = 6

    print(f"1/{n} Writing openmapflow")
    create_openmapflow_config(args.overwrite)

    # Can only import when openmapflow.yaml is available
    from openmapflow.config import PROJECT, PROJECT_ROOT
    from openmapflow.config import DataPaths as dp  # noqa E402

    print(f"2/{n} Copying datasets.py file")
    copy_datasets_py_file(PROJECT_ROOT, args.overwrite)

    print(f"3/{n} Creating data directories")
    create_data_dirs(dp=dp, overwrite=args.overwrite)

    print(f"4/{n} Creating Github Actions")
    git_root = get_git_root(PROJECT_ROOT)
    is_subdir = git_root != PROJECT_ROOT
    create_github_actions(git_root, is_subdir, PROJECT, dp, args.overwrite)

    print(f"5/{n} Setting up dvc (data version control)")
    setup_dvc(PROJECT_ROOT, is_subdir, dp)

    print(f"6/{n} Ready to go! 🎉")
    colab_url = "https://colab.research.google.com"
    nb_home = (
        f"{colab_url}/github/nasaharvest/openmapflow/blob/main/openmapflow/notebooks"
    )
    print(
        f"""
Push your changes to Github and you'll be able to run Colab notebooks:
1) Adding new data\n{nb_home}/new_data.ipynb
2) Training a model\n{nb_home}/train.ipynb
3) Creating a map\n{nb_home}/create_map.ipynb

Notebooks can also be run locally:
openmapflow copy notebooks .
jupyter notebook
"""
    )
