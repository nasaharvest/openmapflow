from setuptools import find_packages, setup

from openmapflow.constants import VERSION

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
    version=VERSION,
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
        "numpy",
        "pandas==1.3.5",
        "pyyaml>=6.0",
        "requests",
        "tqdm>=4.9.0",
    ],
    extras_require={
        "data": [
            "cmocean",
            "dask",
            "earthengine-api",
            "geopandas",
            "google-cloud-storage",
            "netCDF4",
            "rasterio",
            "xarray",
        ],
        "all": [
            "cmocean",
            "dask",
            "earthengine-api",
            "geopandas",
            "google-cloud-storage",
            "ipython",
            "netCDF4",
            "rasterio",
            "xarray",
            "fastcore<1.5.18",
            "tsai",
        ],
    },
    python_requires=">=3.6",
)
