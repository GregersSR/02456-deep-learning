from collections.abc import Generator
import itertools
from typing import TypeVar
import pandas as pd
import numpy as np
import torch
import tqdm
from torch.utils.data import Dataset, TensorDataset
import pyarrow.parquet as pq
from paths import SPLITS_DIR

LOOKBACK = 30
N_PREDICT = 10

A = TypeVar('A')
Pair = tuple[A, A]
Tensor = torch.Tensor

def sliding_windows(segment: np.array) -> Generator[Pair[np.array]]:
    """Split a segment into sliding window rows.
    
    Uses the following rule:
    Output observations 0 through 29 as x, then 30 through 39 as y.
    Output obs. 10 through 39 as x, then 40 through 49 as y.
    ...
    As soon as there is not enough values to output an observation, stop.
    This means that some observations may not be used.
    """

    full_batches = (len(segment) - LOOKBACK) // N_PREDICT
    for i in range(full_batches):
        x_start = N_PREDICT*i
        y_start = x_start + LOOKBACK
        y_end = y_start + N_PREDICT
        yield segment[x_start:y_start], segment[y_start:y_end]

def to_tensors(df: pd.DataFrame) -> Pair[torch.Tensor]:
    """Extracts features from a DataFrame
    
    The DataFrame should have a row for each Segment (as produced by preprocessing.py).
    There must be no missing values.
    Each Segment is converted into a variable number of observations using sliding_window.
    Then the timestamps are replaced by a positional encoding: the first observation for each segment gets index 0 and so forth.
    
    The resulting train tensor has dimensions n_windows x 30 x 4.
    The first dimension indexes over the sliding windows, the second over the sequence of 30 data points and the third over the 4 features of each data point (LAT, LONG, SOG, COG).
    """
    # sort by ts so we can easily compute the positional encoding
    df.sort_values(by='Timestamp')
    df.drop(columns=['Timestamp', 'MMSI'], inplace=True)
    segments = df.groupby('segment_id').apply(pd.DataFrame.to_numpy, include_groups=False).to_numpy()
    # Now we have a Numpy array of length (n_segments) in the first layer, (segment length) in the second layer
    print("Initial data manipulation done, computing sliding windows and building x and y tensors . . .")
    windows = itertools.chain.from_iterable(sliding_windows(segment) for segment in tqdm.tqdm(segments))
    xs = []
    ys = []
    for x, y in windows:
        xs.append(torch.Tensor(x))
        ys.append(torch.Tensor(y))
    print("Built tensor lists")
    xs = torch.stack(xs)
    ys = torch.stack(ys)
    print("Built full tensors")
    print(xs.shape, ys.shape)
    return xs, ys

class AisDataset(TensorDataset):
    def __init__(self, inp, target):
        super().__init__(inp, target)

    @classmethod
    def from_pq(cls, pq_source):
        """Loads a data set from a Parquet source
        
        Uses same syntax as pq.read_table(...).
        """
        return cls(*to_tensors(pq.read_table(pq_source).to_pandas()))
    
    @classmethod
    def train(cls):
        return cls.from_pq(SPLITS_DIR / 'train.parquet')
    
    @classmethod
    def val(cls):
        return cls.from_pq(SPLITS_DIR / 'val.parquet')


def load_train() -> Dataset:
    return AisDataset.train()

def load_val() -> Dataset:
    return AisDataset.val()

if __name__ == '__main__':
    # Load datasets
    train_dataset = load_train()
    val_dataset = load_val()
    
    def print_first_n_windows(dataset, name: str, n: int = 100):
        print(f"\n{name} dataset: first {n} windows")
        for i in range(min(n, len(dataset))):
            x, y = dataset[i]
            print(f"\nWindow {i}:")
            print("x.shape =", x.shape)
            print(x)
            print("y.shape =", y.shape)
            print(y)
    
    print_first_n_windows(train_dataset, "Train")


