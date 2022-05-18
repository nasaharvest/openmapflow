# OpenMapFlow 🌍
[![CI Status](https://github.com/nasaharvest/openmapflow/actions/workflows/ci.yaml/badge.svg)](https://github.com/nasaharvest/openmapflow/actions/workflows/ci.yaml)
[![Docker Status](https://github.com/nasaharvest/openmapflow/actions/workflows/docker.yaml/badge.svg)](https://github.com/nasaharvest/openmapflow/actions/workflows/docker.yaml)

Rapid map creation with machine learning and earth observation data.

## How it works
```bash
pip install openmapflow
openmapflow generate
```
Generates a project structure for: Adding data ➞ Training a model ➞ Creating a map (all steps accessible through Colab)

[cb]: https://colab.research.google.com/assets/colab-badge.svg
[1]: https://colab.research.google.com/github/nasaharvest/openmapflow/blob/main/openmapflow/notebooks/new_data.ipynb
[2]: https://colab.research.google.com/github/nasaharvest/openmapflow/blob/main/openmapflow/notebooks/train.ipynb
[3]: https://colab.research.google.com/github/nasaharvest/openmapflow/blob/main/openmapflow/notebooks/create_map.ipynb

| Adding data  | Training a model | Creating a map |
| ------------ | ---------------- | -------------- |
| [![cb]][1]   | [![cb]][2]       | [![cb]][3]     |


Notebooks can also be run locally by pulling them from the package: `openmapflow cp notebooks .`

## Examples

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

![3maps-gif](assets/3maps.gif)
| [Cropland](https://github.com/nasaharvest/openmapflow/tree/main/crop-mask-example) | [Buildings](https://github.com/nasaharvest/openmapflow/tree/main/buildings-example) | [Maize](https://github.com/nasaharvest/openmapflow/tree/main/maize-example) |
| ---------| --------   | ------|
| [![tb1]][ta1] [![db1]][da1] &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; | [![tb2]][ta2] [![db2]][da2] &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; | [![tb3]][ta3] [![db3]][da3] &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;  |





