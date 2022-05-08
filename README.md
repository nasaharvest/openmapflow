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

[cb]: https://colab.research.google.com/assets/colab-badge.svg
[1]: https://colab.research.google.com/github/nasaharvest/openmapflow/blob/main/openmapflow/notebooks/new_data.ipynb
[2]: https://colab.research.google.com/github/nasaharvest/openmapflow/blob/main/openmapflow/notebooks/train.ipynb
[3]: https://colab.research.google.com/github/nasaharvest/openmapflow/blob/main/openmapflow/notebooks/create_map.ipynb


Each step of the workflow is accessible through a notebook:
| Adding data  | Training a model | Creating a map |
| ------------ | ---------------- | -------------- |
| [![cb]][1]   | [![cb]][2]       | [![cb]][3]     |

## Examples

[img1]: https://storage.googleapis.com/harvest-public-assets/openmapflow/crop-mask-example-map.png

[ta1]: https://github.com/nasaharvest/openmapflow/actions/workflows/crop-mask-example-test.yml
[tb1]: https://github.com/nasaharvest/openmapflow/actions/workflows/crop-mask-example-test.yml/badge.svg
[da1]: https://github.com/nasaharvest/openmapflow/actions/workflows/crop-mask-example-deploy.yml
[db1]: https://github.com/nasaharvest/openmapflow/actions/workflows/crop-mask-example-deploy.yml/badge.svg


| Crop mask                     | [WIP] Crop type           | [WIP] Buildings       |
| ------------                  | ----------------          | --------------        |
| [![tb1]][ta1] [![db1]][da1]   | ----------------          | --------------        |
| ![Crop mask][img1]            | ![Crop type][img1]        | ![Buildings][img1]    |


