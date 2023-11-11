# Regression and integration for FMI

## Data management

The training data is managed with dvc. The repo is at DIR Isilon drive:
```
git clone /export/Lab-Xue/data/FM_data_repo ./FM_data_repo

# to get mri data

cd ./FM_data_repo

dvc remote add -d fm_data /export/Lab-Xue/data/FM_data_repo

dvc get fm_data mri
```