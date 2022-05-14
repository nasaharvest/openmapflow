# OpenMapFlow 🌍

[![CI Status](https://github.com/nasaharvest/openmapflow/actions/workflows/ci.yaml/badge.svg)](https://github.com/nasaharvest/openmapflow/actions/workflows/ci.yaml)
[![Docker Status](https://github.com/nasaharvest/openmapflow/actions/workflows/docker.yaml/badge.svg)](https://github.com/nasaharvest/openmapflow/actions/workflows/docker.yaml)

Rapid map creation with machine learning and earth observation data.

## Examples

[img1]: https://storage.googleapis.com/harvest-public-assets/openmapflow/crop-mask-example-map.png

[ta1]: https://github.com/nasaharvest/openmapflow/actions/workflows/crop-mask-example-test.yaml
[tb1]: https://github.com/nasaharvest/openmapflow/actions/workflows/crop-mask-example-test.yaml/badge.svg
[da1]: https://github.com/nasaharvest/openmapflow/actions/workflows/crop-mask-example-deploy.yaml
[db1]: https://github.com/nasaharvest/openmapflow/actions/workflows/crop-mask-example-deploy.yaml/badge.svg

[ta2]: https://github.com/nasaharvest/openmapflow/actions/workflows/buildings-example-test.yaml
[tb2]: https://github.com/nasaharvest/openmapflow/actions/workflows/buildings-example-test.yaml/badge.svg
[da2]: https://github.com/nasaharvest/openmapflow/actions/workflows/buildings-example-deploy.yaml
[db2]: https://github.com/nasaharvest/openmapflow/actions/workflows/buildings-example-deploy.yaml/badge.svg

[ta3]: https://github.com/nasaharvest/openmapflow/actions/workflows/maize-example-test.yaml
[tb3]: https://github.com/nasaharvest/openmapflow/actions/workflows/maize-example-test.yaml/badge.svg
[da3]: https://github.com/nasaharvest/openmapflow/actions/workflows/maize-example-deploy.yaml
[db3]: https://github.com/nasaharvest/openmapflow/actions/workflows/maize-example-deploy.yaml/badge.svg


| Cropland                      | [WIP] Buildings               | [WIP] Maize                   |
| ------------                  | ----------------              | --------------                |
| [![tb1]][ta1] [![db1]][da1]   | [![tb2]][ta2] [![db2]][da2]   | [![tb3]][ta3] [![db3]][da3]   |
| ![Crop mask][img1]            | ![Buildings][img1]            | ![Crop Type][img1]            |
## How it works
```bash
pip install openmapflow
openmapflow generate
```
Generates the following workflow: 
Adding data ➞ Training a model ➞ Creating a map

[cb]: https://colab.research.google.com/assets/colab-badge.svg
[1]: https://colab.research.google.com/github/nasaharvest/openmapflow/blob/main/openmapflow/notebooks/new_data.ipynb
[2]: https://colab.research.google.com/github/nasaharvest/openmapflow/blob/main/openmapflow/notebooks/train.ipynb
[3]: https://colab.research.google.com/github/nasaharvest/openmapflow/blob/main/openmapflow/notebooks/create_map.ipynb


Each step of the workflow is accessible through a notebook:
| Adding data  | Training a model | Creating a map |
| ------------ | ---------------- | -------------- |
| [![cb]][1]   | [![cb]][2]       | [![cb]][3]     |

Notebooks can also be run locally by pulling them from the package 
```
openmapflow cp notebooks .
```


