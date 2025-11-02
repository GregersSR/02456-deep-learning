from pathlib import Path

HERE = Path(__file__).parent
ROOT = HERE.parent
RAW_DATA_DIR = ROOT / 'data'
PROCESSED_DATA_DIR = ROOT / 'processed_data'

assert ROOT.is_dir()
RAW_DATA_DIR.mkdir(exist_ok=True)
PROCESSED_DATA_DIR.mkdir(exist_ok=True)
