import pandas as pd
import pyarrow.parquet as pq
import pyarrow as pa
from sklearn.model_selection import train_test_split
from paths import PROCESSED_DATA_DIR, SPLITS_DIR

df = pq.read_table(PROCESSED_DATA_DIR).to_pandas()

# create segments, a smaller dataframe, with one row per segment, looking like this: MMSI, Segment, Timestamp (start time of segment)
segments = df.groupby(['MMSI', 'Segment']).agg({
    'Timestamp': 'min'  # take the start time of the segment
}).reset_index()

# First split: train vs temp (train 70%, temp 30%)
train_segments, temp_segments = train_test_split(
    segments,
    test_size=0.3,
    random_state=42,
    shuffle=True
)

# Second split: temp -> validation and test (each 15% of total)
val_segments, test_segments = train_test_split(
    temp_segments,
    test_size=0.5,
    random_state=42,
    shuffle=True
)

train_df = df.merge(train_segments[['MMSI','Segment']], on=['MMSI','Segment'])
val_df   = df.merge(val_segments[['MMSI','Segment']], on=['MMSI','Segment'])
test_df  = df.merge(test_segments[['MMSI','Segment']], on=['MMSI','Segment'])

train_df.to_parquet(f"{SPLITS_DIR}/train.parquet", index=False)
val_df.to_parquet(f"{SPLITS_DIR}/val.parquet", index=False)
test_df.to_parquet(f"{SPLITS_DIR}/test.parquet", index=False)
