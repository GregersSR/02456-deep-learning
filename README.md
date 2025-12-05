# 02456 Deep learning project
This repository contains the code for the project part of 02456 Deep Learning at DTU, fall 2025.
We investigate ship trajectories in and around Danish waters, aiming to predict future vessel movement based on past trajectory.

## Data acquisition and preprocessing
We use data from the entire year 2024.
1. Running `02456_dl_project/download.py` fetches all the data, preprocesses it and stores it under `processed_data` in the repository root. It takes considerable time and resources (~8 hours, 30 Gb memory) to run this.
2. The `02456_dl_project/split_data.ipynb` notebook partitions the data into train, validation and test splits and saves them under `data_splits` under the repository root.

The three files `train.parquet`, `val.parquet` and `test.parquet` are the data sets we base our models on.
The `dataloader.py`
