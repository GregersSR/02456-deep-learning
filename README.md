# 02456 Deep learning project
This repository contains the code for the project part of 02456 Deep Learning at DTU, fall 2025.
We investigate ship trajectories in and around Danish waters, aiming to predict future vessel movement based on past trajectory.

## Problem statement
The objective of this project is to predict the short-term trajectories of vessels using diverse models (linear, LSTM and transformer).
* **Prediction Window:** Predict the **next 50 minutes** (10 timesteps).
* **Context Window:** Based on the **last 150 minutes** (30 data points) of a vessel's history

## Data acquisition and preprocessing
We use data from the entire year 2024.
1. Running `02456_dl_project/download.py` fetches all the data, preprocesses it and stores it under `processed_data` in the repository root. It takes considerable time and resources (~8 hours, 30 Gb memory) to run this.
2. The `02456_dl_project/split_data.ipynb` notebook partitions the data into train, validation and test splits and saves them under `data_splits` under the repository root.
The three files `train.parquet`, `val.parquet` and `test.parquet` are the data sets we base our models on.
3. The script in `dataloader.py` is responsible for transforming the raw processed data into standardized, ready-to-use tensors for PyTorch models.

### Key Preprocessing Steps

* **Downsampling:** The raw data was downsampled to a **5-minute interval**.
* **Feature Selection:** We exclusively used **latitude and longitude** as positional input features. Speed Over Ground (SOG) and Course Over Ground (COG) were removed because their skewed distributions and missing values negatively affected model performance.
* **Stationary Segment Removal:** Segments classified as non-moving were completely removed from the dataset. A segment was considered stationary if all data points were within a **20-meter radius**. This data is not of interest for collision prevention.
* **Sequence Conversion:** Vessel segments were split using a **sliding window** to convert variable-length sequences to a fixed input/output length of **30 input observations** predicting **10 target observations**.
* **Splits:** Data was split into Train/Validation/Test sets using **70%-15%-15%** portions.


## Training
The core logic for executing model training is handled by `training.py`. The module contains shared utilities for efficiently training and evaluating **all models**. The central function, `train_model`, handles the entire training and validation loop. It automatically selects the optimal device (MPS, CUDA, or CPU), utilizes the AdamW optimizer, and calculates essential metrics (MSE, RMSE, MAE) for both training and validation sets. Crucially, it implements an early stopping mechanism based on validation RMSE, halting training after 20 epochs without improvement, and manages checkpointing by saving the model state with the lowest recorded validation RMSE.




