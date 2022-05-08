# OpenMapFlow 🌍

[![CI Status](https://github.com/nasaharvest/openmapflow/actions/workflows/ci.yml/badge.svg)](https://github.com/nasaharvest/openmapflow/actions/workflows/ci.yml)
[![Docker Status](https://github.com/nasaharvest/openmapflow/actions/workflows/docker.yml/badge.svg)](https://github.com/nasaharvest/openmapflow/actions/workflows/docker.yml)

Rapid map creation with machine learning and earth observation data.

### How it works
```bash
pip install openmapflow
openmapflow generate
```
This will generate all the following workflow: 
Adding data ➞ Training a model ➞ Creating a map

Each step of the workflow is accessible through a notebook:s
| Adding data  | Training a model | Creating a map |
| ------------ | ---------------- | -------------- |
| [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/nasaharvest/openmapflow/blob/main/openmapflow/notebooks//new_data.ipynb)    |   [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/nasaharvest/openmapflow/blob/main/openmapflow/notebooks/train.ipynb)  |   [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/nasaharvest/openmapflow/blob/main/openmapflow/notebooks/create_map.ipynb) |

## Examples

| Crop mask  | [WIP] Crop type | [WIP] Buildings |
| ------------ | ---------------- | -------------- |
| [![Test Status](https://github.com/nasaharvest/openmapflow/actions/workflows/crop-mask-example-test.yml/badge.svg)](https://github.com/nasaharvest/openmapflow/actions/workflows/crop-mask-example-test.yml) [![Deploy Status](https://github.com/nasaharvest/openmapflow/actions/workflows/crop-mask-example-deploy.yml/badge.svg)](https://github.com/nasaharvest/openmapflow/actions/workflows/crop-mask-example-deploy.yml) | ---------------- | -------------- |
| <img src="https://storage.googleapis.com/harvest-public-assets/openmapflow/crop-mask-example-map.png" />              | <img src="https://storage.googleapis.com/harvest-public-assets/openmapflow/crop-mask-example-map.png" />            |    <img src="https://storage.googleapis.com/harvest-public-assets/openmapflow/crop-mask-example-map.png" />         |


