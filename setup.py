from setuptools import setup, find_packages

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
    version="0.0.1",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: Other/Proprietary License",
        "Operating System :: OS Independent",
    ],
    packages=["openmapflow"]
    + [f"openmapflow.{f}" for f in find_packages("openmapflow")],
    package_data={
        "": [
            "Dockerfile",
            "notebooks/*.ipynb",
            "trigger_inference_function/*",
            "templates/*",
        ]
    },
    scripts=[
        "openmapflow/scripts/openmapflow-create-features",
        "openmapflow/scripts/openmapflow-deploy",
        "openmapflow/scripts/openmapflow-dir",
        "openmapflow/scripts/openmapflow-datapath",
        "openmapflow/scripts/openmapflow-generate",
    ],
    install_requires=[
        "cropharvest>=0.3.0",
        "dvc[gdrive]>=2.10.1",
        "earthengine-api",
        "pandas==1.3.5",
        "pyyaml>=6.0",
    ],
    python_requires=">=3.6",
    include_package_data=True,
)
