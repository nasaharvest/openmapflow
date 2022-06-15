from setuptools import find_packages, setup

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

long_description = "\n".join(
    [line for line in long_description.split("\n") if not line.startswith("<img")]
)

setup(
    name="openmapflow",
    description="Creating maps with machine learning models and earth observation data.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Ivan Zvonkov",
    author_email="izvonkov@umd.edu",
    url="https://github.com/nasaharvest/openmapflow",
    version="0.0.1rc1",
    license="Apache License 2.0",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
    packages=["openmapflow"]
    + [f"openmapflow.{f}" for f in find_packages("openmapflow")],
    package_data={
        "": [
            "Dockerfile",
            "notebooks/*.ipynb",
            "scripts/*",
            "trigger_inference_function/*",
            "templates/*",
        ]
    },
    scripts=["openmapflow/scripts/openmapflow", "openmapflow/scripts/deploy.sh"],
    install_requires=[
        "cmocean",
        "cropharvest>=0.3.0",
        "dvc[gdrive]>=2.10.1",
        "earthengine-api",
        "h5py>=3.1.0,!=3.7.0",
        "ipyleaflet>=0.16.0",
        "pandas==1.3.5",
        "protobuf==3.20.1",
        "pyyaml>=6.0",
    ],
    python_requires=">=3.6",
)
