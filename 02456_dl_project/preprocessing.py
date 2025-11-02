"""Preprocessing functionality.

The data set is read one day at a time. Some preprocessing is
done one the DataFrame for the single day. When each day has been
preprocessed, it is added to an aggregate DataFrame, which is then
preprocessed again in the end.
"""

import pandas as pd

BBOX = [60, 0, 50, 20]
KNOTS_TO_MS = 0.514444

def track_filter(g):
        len_filt = len(g) > 256  # Min required length of track/segment
        sog_filt = 1 <= g["SOG"].max() <= 50  # Remove stationary tracks/segments
        time_filt = (g["Timestamp"].max() - g["Timestamp"].min()).total_seconds() >= 60 * 60  # Min required timespan
        return len_filt and sog_filt and time_filt

def preprocess_partial(df: pd.DataFrame) -> pd.DataFrame:
    """This preprocessing is applied to each day-df individually"""
    # Remove errors
    north, west, south, east = BBOX
    df = df[(df["Latitude"] <= north) & (df["Latitude"] >= south) & (df["Longitude"] >= west) & (
            df["Longitude"] <= east)]
    
    # Filter out irrelevant vessel types and drop the column
    df = df[df["Type of mobile"].isin(["Class A", "Class B"])].drop(columns=["Type of mobile"])
    df = df[df["MMSI"].str.len() == 9]  # Adhere to MMSI format
    df = df[df["MMSI"].str[:3].astype(int).between(200, 775)]  # Adhere to MID standard

    df = df.rename(columns={"# Timestamp": "Timestamp"})
    # raises errors as opposed to Peder's code so we see any data quality issues
    df["Timestamp"] = pd.to_datetime(df["Timestamp"], format="%d/%m/%Y %H:%M:%S", errors="raise")


    df.drop_duplicates(["Timestamp", "MMSI", ], keep="first", inplace=True)

    # Track filtering
    df = df.groupby("MMSI").filter(track_filter)
    df = df.sort_values(['MMSI', 'Timestamp'])
    df["SOG"] = KNOTS_TO_MS * df["SOG"]
    return df

def preprocess_full(df: pd.DataFrame) -> pd.DataFrame:
      # Divide track into segments based on timegap
    df['Segment'] = df.groupby('MMSI')['Timestamp'].transform(
        lambda x: (x.diff().dt.total_seconds().fillna(0) >= 15 * 60).cumsum())  # Max allowed timegap

    # Segment filtering
    df = df.groupby(["MMSI", "Segment"]).filter(track_filter)
    df.reset_index(drop=True, inplace=True)
    return df
