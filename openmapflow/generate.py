import argparse
import os
import shutil
from pathlib import Path
from typing import Union

from openmapflow.constants import (
    CONFIG_FILE,
    TEMPLATE_DATASETS,
    TEMPLATE_DEPLOY_YML,
    TEMPLATE_EVALUATE,
    TEMPLATE_README,
    TEMPLATE_REQUIREMENTS,
    TEMPLATE_TEST_YML,
    TEMPLATE_TRAIN,
    VERSION,
)
from openmapflow.utils import confirmation


def allow_write(p: Union[Path, str], overwrite: bool = False) -> bool:
    """Prompts user if file already exists"""
    if overwrite or not Path(p).exists():
        return True
    p = Path(*Path(p).parts[-4:])  # Shorten for legibility
    overwrite_input = input(f"  {str(p)} already exists. Overwrite? (y/[n]): ")
    return overwrite_input.lower() == "y"


def create_openmapflow_config(overwrite: bool):
    """Creates openmapflow.yaml config file"""
    if not allow_write(CONFIG_FILE, overwrite):
        return
    cwd = Path.cwd()
    project_name = input(f"  Project name [{cwd.stem}]: ") or cwd.stem
    auto_description = f"OpenMapFlow {project_name.replace('-', ' ').replace('_', ' ')}"
    description = input(f"  Description [{auto_description}]: ") or auto_description
    gcloud_project_id = input("  GCloud project ID: ")
    gcloud_location = input("  GCloud location [us-central1]: ") or "us-central1"

    buckets = {
        "bucket_labeled_eo": f"{project_name}-labeled-eo",
        "bucket_inference_eo": f"{project_name}-inference-eo",
        "bucket_preds": f"{project_name}-preds",
        "bucket_preds_merged": f"{project_name}-preds-merged",
    }

    for k, v in buckets.items():
        buckets[k] = input(f"  GCloud {k.replace('_', ' ')} [{v}]: ") or v

    openmapflow_str = (
        f"version: {VERSION}"
        + f"\nproject: {project_name}"
        + f"\ndescription: {description}"
        + "\ngcloud:"
        + f"\n    project_id: {gcloud_project_id}"
        + f"\n    location: {gcloud_location}"
        + f"\n    bucket_labeled_eo: {buckets['bucket_labeled_eo']}"
        + f"\n    bucket_inference_eo: {buckets['bucket_inference_eo']}"
        + f"\n    bucket_preds: {buckets['bucket_preds']}"
        + f"\n    bucket_preds_merged: {buckets['bucket_preds_merged']}"
    )

    with open(CONFIG_FILE, "w") as f:
        f.write(openmapflow_str)


def copy_template_files(PROJECT_ROOT: Path, overwrite: bool):
    """Copies template files to project directory"""
    for p in [
        TEMPLATE_DATASETS,
        TEMPLATE_TRAIN,
        TEMPLATE_EVALUATE,
        TEMPLATE_REQUIREMENTS,
        TEMPLATE_README,
    ]:
        if allow_write(PROJECT_ROOT / p.name, overwrite):
            shutil.copy(str(p), str(PROJECT_ROOT / p.name))


def create_data_dirs(dp, overwrite: bool):
    """Creates data directories"""
    for p in [dp.RAW_LABELS, dp.DATASETS, dp.MODELS]:
        if allow_write(p, overwrite):
            Path(p).mkdir(parents=True, exist_ok=True)


def fill_in_and_write_action(
    src_yml_path: Path,
    dest_yml_path: Path,
    sub_prefix: str,
    sub_paths: str,
    sub_cd: str,
):
    """
    Fills in template action and writes to file
    Args:
        src_yml_path: Path to template yaml file
        dest_yml_path: Path to write filled in yaml file
        sub_prefix: Prefix to add to action name
        sub_paths: Paths to trigger action by
        sub_cd: Command to cd into project root
    """
    with src_yml_path.open("r") as f:
        content = f.read()
    content = content.replace("<PREFIX>", sub_prefix)
    content = content.replace("<PATHS>", sub_paths)
    content = content.replace("<CD>", sub_cd)

    dest_yml_path.parent.mkdir(parents=True, exist_ok=True)
    with dest_yml_path.open("w") as f:
        f.write(content)


def get_git_root(PROJECT_ROOT: Path):
    """Returns git root"""
    possible_git_roots = [PROJECT_ROOT, PROJECT_ROOT.parent]
    try:
        return next(r for r in possible_git_roots if (r / ".git").exists())
    except StopIteration:
        raise FileExistsError(
            f"Could not find .git in {str(PROJECT_ROOT)} or its parent"
        )


def create_github_actions(
    git_root: Path, is_subdir: bool, PROJECT: str, dp, overwrite: bool
):
    """
    Creates github actions files based on templates
    Args:
        git_root: Path to git root
        is_subdir: Whether project is a subdirectory
        PROJECT: Project name
        dp: Data paths
        overwrite: Whether to overwrite existing files
    """
    dest_deploy_yml_path = git_root / f".github/workflows/{PROJECT}-deploy.yaml"
    dest_test_yml_path = git_root / f".github/workflows/{PROJECT}-test.yaml"
    if allow_write(dest_deploy_yml_path, overwrite):
        fill_in_and_write_action(
            src_yml_path=TEMPLATE_DEPLOY_YML,
            dest_yml_path=dest_deploy_yml_path,
            sub_prefix=PROJECT.split("-")[0],
            sub_paths=f"{f'{PROJECT}/' if is_subdir else ''}{dp.MODELS}.dvc",
            sub_cd=PROJECT if is_subdir else ".",
        )

    if allow_write(dest_test_yml_path, overwrite):
        fill_in_and_write_action(
            src_yml_path=TEMPLATE_TEST_YML,
            dest_yml_path=dest_test_yml_path,
            sub_prefix=PROJECT.split("-")[0],
            sub_paths="",
            sub_cd=PROJECT if is_subdir else ".",
        )


def _print_and_run(cmd: str):
    """Prints command and runs it"""
    print(f"{cmd}")
    os.system(cmd)


def setup_dvc(PROJECT_ROOT: Path, is_subdir: bool, dp):
    """
    Sets up dvc (data version control)
    Args:
        PROJECT_ROOT: Path to project root
        is_subdir: Whether project is a subdirectory
        dp: Data paths
    """
    if (PROJECT_ROOT / ".dvc").exists():
        print(f"  {PROJECT_ROOT}/.dvc already exists. Skipping.")
        return

    if not confirmation("Install dvc for data version control?"):
        return

    _print_and_run("pip install dvc[gs]")
    if is_subdir:
        _print_and_run("dvc init --subdir")
    else:
        _print_and_run("dvc init")

    dvc_files = [dp.RAW_LABELS, dp.DATASETS, dp.MODELS]
    _print_and_run("dvc add " + " ".join(dvc_files))

    print("dvc stores data in remote storage (s3, gcs, gdrive, etc)")
    print("https://dvc.org/doc/command-reference/remote/add#supported-storage-types")
    option = input(
        "a) Setup google cloud remote storage / b) Exit and setup own remote [a]/b: "
    )
    if option.lower() == "b":
        return

    print(
        "We'll follow: https://dvc.org/doc/command-reference/remote/add#google-cloud-storage"
    )
    gcs_uri = input("Google Cloud Storage URI (E.g. gs://mybucket/path): ")
    if gcs_uri.startswith("gs://"):
        _print_and_run(f"dvc remote add -d gcs {gcs_uri}")
        _print_and_run("dvc push")
    else:
        print(
            "You must enter a valid Google Cloud Storage URI (E.g. gs://mybucket/path)"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate OpenMapFlow project.")
    parser.add_argument("--overwrite", action="store_true", help="overwrite")
    args = parser.parse_args()

    n = 6

    print(f"1/{n} Writing openmapflow")
    create_openmapflow_config(args.overwrite)

    # Can only import when openmapflow.yaml is available
    from openmapflow.config import PROJECT, PROJECT_ROOT
    from openmapflow.config import DataPaths as dp  # noqa E402

    print(f"2/{n} Copying datasets.py, train.py, evaluate.py requirements.txt")
    copy_template_files(PROJECT_ROOT, args.overwrite)

    print(f"3/{n} Creating data directories")
    create_data_dirs(dp=dp, overwrite=args.overwrite)

    print(f"4/{n} Creating Github Actions")
    git_root = get_git_root(PROJECT_ROOT)
    is_subdir = git_root != PROJECT_ROOT
    create_github_actions(git_root, is_subdir, PROJECT, dp, args.overwrite)

    print(f"5/{n} Setting up dvc (data version control)")
    setup_dvc(PROJECT_ROOT, is_subdir, dp)

    print(
        f"6/{n} Ready to go! 🎉\n"
        f"See: https://github.com/nasaharvest/openmapflow/blob/main/README.md "
        + "for guides on adding data, training models, and creating maps"
    )
