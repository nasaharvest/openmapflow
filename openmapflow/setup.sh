#- take unique project name - set it inside the notebooks
#- bucket creation: remote storage, tifs, earth engine, press, preds-merged
#- container registry creation
# env creation


# If project is directly in git repo
# dvc init

# If project is in subdirectory
# cd <subdir> dvc init --subdir

# Make data directories: [models, features, processed_labels, raw_labels]

# Add dvc data: [models, features, processed_labels, raw_labels]
# dvc add data/raw_labels ...
# dvc commit

# Set dvc remote, Google Drive by default for simplicity
# https://dvc.org/doc/user-guide/setup-google-drive-remote

# dvc remote add -d gdrive gdrive://1EMHILcNFwdusMHHs4eC4OVIJ0Ncp9fiY/crop-mask-example-dvc-store
