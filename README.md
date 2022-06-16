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

# OpenMapFlow 🌍
[![CI Status](https://github.com/nasaharvest/openmapflow/actions/workflows/ci.yaml/badge.svg)](https://github.com/nasaharvest/openmapflow/actions/workflows/ci.yaml)
[![Docker Status](https://github.com/nasaharvest/openmapflow/actions/workflows/docker.yaml/badge.svg)](https://github.com/nasaharvest/openmapflow/actions/workflows/docker.yaml)
[![tb1]][ta1] [![db1]][da1]
[![tb2]][ta2] [![db2]][da2]
[![tb3]][ta3] [![db3]][da3]


Rapid map creation with machine learning and earth observation data.

[cb]: https://colab.research.google.com/assets/colab-badge.svg

**Examples:** [Cropland](https://github.com/nasaharvest/openmapflow/tree/main/crop-mask-example), [Buildings](https://github.com/nasaharvest/openmapflow/tree/main/buildings-example), [Maize](https://github.com/nasaharvest/openmapflow/tree/main/maize-example)

![3maps-gif](assets/3maps.gif)

* [Tutorial](#tutorial-)
* [Generating a project](#generating-a-project-)
* [Adding data](#adding-data-)
* [Training a model](#training-a-model-)
* [Creating a map](#creating-a-map-)

## Tutorial [![cb]](https://colab.research.google.com/github/nasaharvest/openmapflow/blob/main/openmapflow/notebooks/tutorial.ipynb)
Colab notebook tutorial demonstrating data exploration, model training, and inference over small region.


## Generating a project [![cb]](https://colab.research.google.com/github/nasaharvest/openmapflow/blob/main/openmapflow/notebooks/generate_project.ipynb)
Inside a Github repository run:
```bash
pip install openmapflow
openmapflow generate
```
This generates a project for: Adding data ➞ Training a model ➞ Creating a map 

## Adding data [![cb]](https://colab.research.google.com/github/nasaharvest/openmapflow/blob/main/openmapflow/notebooks/new_data.ipynb)

Move raw labels into project:
```bash
export RAW_LABEL_DIR=$(openmapflow datapath RAW_LABELS)
mkdir RAW_LABEL_DIR/<my dataset name>
cp -r <path to my raw data files> RAW_LABEL_DIR/<my dataset name>
```
Add reference to data using a `LabeledDataset` object in datasets.py:
```python
datasets = [
    LabeledDataset(
        dataset="example_dataset",
        country="Togo",
        raw_labels=(
            RawLabels(
                filename="Togo_2019.csv",
                longitude_col="longitude",
                latitude_col="latitude",
                class_prob=lambda df: df["crop"],
                start_year=2019,
                x_y_from_centroid=False,
            ),
        ),
    ),
    ...
]
```
Run feature creation:
```bash
earthengine authenticate    # For getting new earth observation data
gcloud auth login           # For getting cached earth observation data

openmapflow create-features # Initiatiates or checks progress of features creation
# May take long time depending on amount of labels in dataset 
# TODO make the end more obvious

openmapflow datasets        # Shows the status of datasets

dvc commit && dvc push      # Push new data to data version control

git add .
git commit -m'Created new features'
git push
```

## Training a model [![cb]](https://colab.research.google.com/github/nasaharvest/openmapflow/blob/main/openmapflow/notebooks/train.ipynb)
```bash
# Pull in latest data
dvc pull    
tar -xzf $(openmapflow datapath COMPRESSED_FEATURES) -C data

export MODEL_NAME=<model_name>              # Set model name
python train.py --model_name $MODEL_NAME    # Train a model
python evaluate.py --model_name $MODEL_NAME # Record test metrics

dvc commit && dvc push  # Push new models to data version control

git checkout -b"$MODEL_NAME"
git add .
git commit -m "$MODEL_NAME"
git push --set-upstream origin "$MODEL_NAME"
```

## Creating a map [![cb]](https://colab.research.google.com/github/nasaharvest/openmapflow/blob/main/openmapflow/notebooks/create_map.ipynb)

Only available through Colab. Cloud Architecture must be deployed using the deploy.yaml Github Action.






