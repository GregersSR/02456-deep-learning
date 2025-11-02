from pathlib import Path
from collections.abc import Generator
from zipfile import ZipFile
import pandas as pd

DTYPES = {
        "MMSI": "object",
        "SOG": float,
        "COG": float,
        "Longitude": float,
        "Latitude": float,
        "# Timestamp": "object",
        "Type of mobile": "object",
}

def read_zip(p: Path) -> pd.DataFrame:
    with ZipFile(p) as zip:
        files = zip.infolist()
        if len(files) != 1:
            raise ValueError(f"Expected zip file to have a single member, not {len(files)}.")
        with zip.open(files[0]) as fp:
            usecols = list(DTYPES.keys())
            return pd.read_csv(fp, usecols=usecols, dtype=DTYPES)

def read_all(dir: Path) -> Generator[pd.DataFrame]:
    zips = sorted(filter(lambda p: p.suffix == '.zip', dir.iterdir()), key=lambda p: p.name)
    for day_f in zips:
        yield read_zip(day_f)


def main():
    zips = sorted(filter(lambda p: p.suffix == '.zip', DATA_DIR.iterdir()), key=lambda p: p.name)
    for day_f in zips:
        df = read_zip(day_f)

        for entry in filter(lambda entry: entry[10] == '8317954', read_zip(day_f)):
            print("\t".join(entry))

if __name__ == '__main__':
    main()
