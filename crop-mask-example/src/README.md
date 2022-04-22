# OpenMapFlow

[readme_url = https://github.com/nasaharvest/crop-mask]

[![Status](https://github.com/nasaharvest/crop-mask/actions/workflows/test.yml/badge.svg)](https://github.com/nasaharvest/crop-mask/actions/workflows/test.yml) [![Status](https://github.com/nasaharvest/crop-mask/actions/workflows/deploy.yml/badge.svg)](https://github.com/nasaharvest/crop-mask/actions/workflows/deploy.yml)

End-to-end workflow for generating high resolution maps.

## Contents
-   [Creating a map](#creating-a-map)
-   [Training a new model](#training-a-new-model)
-   [Adding new labeled data](#adding-new-labeled-data)
-   [Setting up a development environment](#setting-up-a-development-environment)
-   [Tests](#tests)
-   [Reference](#reference)

## Creating a map
To create a map run the following colab notebook (or use it as a guide): 
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/nasaharvest/crop-mask/blob/master/notebooks/inference.ipynb)

## Training a new model
To train a new model run the following colab notebook (or use it as a guide):
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/nasaharvest/crop-mask/blob/master/notebooks/train.ipynb)

## Adding new labeled data

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/nasaharvest/crop-mask/blob/master/notebooks/data.ipynb)

Ensure the local environment is setup.
1. Add the csv or shape file for new labels into [data/raw](data/raw)
2. In [dataset_labeled.py](src/datasets_labeled.py) add a new `LabeledDataset` object into the `labeled_datasets` list and specify the required parameters.
```bash
# Activate environment setup in: Setting up a local environment
conda activate landcover-mapping 

dvc pull                                    # Get latest data from dvc
earthengine authenticate                    # Authenticates Earth Engine 
python -c "import ee; ee.Initialize()"      # Will raise error if not setup

# Pull in latest EarthEngine tifs (you may need to rerun this command)
gsutil -m cp -n -r gs://crop-mask-tifs2/tifs data/

# Create features (you may need to rerun this command)
python scripts/create_features.py

dvc commit                                  # Save new features to repository
dvc push                                    # Push features to remote storage

# Push changes to github
git checkout -b'new-Ethiopia-Tigray-data'
git add .
git commit -m 'Added mew Ethiopia Tigray data for 2020'
git push
```

## Setting up a development environment
Ensure you have [anaconda](https://www.anaconda.com/download/#macos) and [gcloud](https://cloud.google.com/sdk/docs/install) installed.  
```bash
conda install mamba -n base -c conda-forge  # Install mamba
mamba env create -f environment.yml     # Create environment with mamba (faster)
conda activate landcover-mapping            # Activate environment
gcloud auth application-default login       # Authenticates with Google Cloud
```

## Tests

The following tests can be run against the pipeline:

```bash
flake8 . # code formatting
mypy .  # type checking
python -m unittest # unit tests

# Integration tests
python -m unittest test/integration_test_labeled.py
python -m unittest test/integration_test_model_bbox.py
python -m unittest test/integration_test_model_evaluation.py
```

## Reference

If you find this code useful, please cite the following paper:


