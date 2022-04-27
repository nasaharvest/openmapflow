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
    install_requires=[
        "cropharvest>=0.3.0",
        "dvc[gdrive]>=2.10.1",
        "earthengine-api",
        "pyyaml==5.4.1",
        "torch",
    ],
    python_requires=">=3.6",
    include_package_data=True,
)
