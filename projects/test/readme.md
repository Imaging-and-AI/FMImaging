# Regression and integration for FMI

## Install dvc
```
pip3 install dvc
```

## Data management

The training data is managed with dvc. The repo is at DIR Isilon drive:
```
git clone git@github.com:NHLBI-MR/FM_data_repo.git .

# to get mri data
cd /data/FM_data_repo
dvc pull
```

## Regression test

```
export FMI_DATA_ROOT=/data/FM_data_repo
export FMI_LOG_ROOT=/data/logs

pytest -s ./projects/test/test_mri_training.py

```