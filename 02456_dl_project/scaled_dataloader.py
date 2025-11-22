from collections.abc import Generator
import itertools
from typing import TypeVar
import pandas as pd
import numpy as np
import torch
import tqdm
from torch.utils.data import Dataset
import pyarrow.parquet as pq
from paths import SPLITS_DIR
from sklearn.preprocessing import StandardScaler

LOOKBACK = 30
N_PREDICT = 10

A = TypeVar('A')
Pair = tuple[A, A]
Tensor = torch.Tensor

def sliding_windows(segment: np.array) -> Generator[Pair[np.array]]:
    full_batches = (len(segment) - LOOKBACK) // N_PREDICT
    for i in range(full_batches):
        x_start = N_PREDICT * i
        y_start = x_start + LOOKBACK
        y_end = y_start + N_PREDICT
        yield segment[x_start:y_start], segment[y_start:y_end]

def to_tensors(df: pd.DataFrame) -> Pair[torch.Tensor]:
    # sort by ts so we can easily compute the positional encoding
    df.sort_values(by='Timestamp', inplace=True)
    df.drop(columns=['Timestamp', 'MMSI', 'SOG', 'COG'], inplace=True)  # DROP SOG and COG
    segments = df.groupby('segment_id').apply(pd.DataFrame.to_numpy, include_groups=False).to_numpy()

    windows = itertools.chain.from_iterable(sliding_windows(segment) for segment in tqdm.tqdm(segments))
    xs = []
    ys = []
    for x, y in windows:
        xs.append(torch.Tensor(x))
        ys.append(torch.Tensor(y))
    xs = torch.stack(xs)
    ys = torch.stack(ys)
    return xs, ys

# dataset with scaling
class AisDataset(Dataset): 
    def __init__(self, x_tensor: Tensor, y_tensor: Tensor, scaler: StandardScaler | None = None):
        self.scaler = scaler
        self.x = self._process(x_tensor)
        self.y = self._process(y_tensor)

    def _process(self, tensor: Tensor) -> Tensor:
        arr = tensor.numpy()
        # Only LAT/LON
        arr = arr[:, :, :2]  # shape (N, seq, 2)

        # scale LAT/LON
        if self.scaler is not None:
            N, seq_len, _ = arr.shape
            flat = arr.reshape(-1, 2)
            flat = self.scaler.transform(flat)
            arr = flat.reshape(N, seq_len, 2)

        return torch.tensor(arr, dtype=torch.float32)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


# raw tensor loaders (for fitting scaler)
def _load_raw(split_name: str) -> Pair[Tensor]:
    df = pq.read_table(SPLITS_DIR / f"{split_name}.parquet").to_pandas()
    return to_tensors(df)


# public API for use in other files
def load_train() -> tuple[AisDataset, StandardScaler]:
    print("Loading TRAIN...")
    x_train, y_train = _load_raw("train")

    # fit scaler on LAT/LON only
    flat_train = x_train.numpy().reshape(-1, 2)
    scaler = StandardScaler()
    scaler.fit(flat_train)

    ds = AisDataset(x_train, y_train, scaler=scaler)
    return ds, scaler

def load_val(scaler: StandardScaler) -> AisDataset:
    print("Loading VAL...")
    x_val, y_val = _load_raw("val")
    return AisDataset(x_val, y_val, scaler=scaler)


if __name__ == '__main__':
    train_ds, scaler = load_train()
    val_ds = load_val(scaler)
    print("Train windows:", len(train_ds))
    print("Val windows:", len(val_ds))
    x_batch, y_batch = train_ds[0]
    print("x_batch shape:", x_batch.shape)
    print("y_batch shape:", y_batch.shape)
