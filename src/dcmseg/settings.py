from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]

DATA_DIR = BASE_DIR / 'src/dcmseg/data'

print(DATA_DIR)
