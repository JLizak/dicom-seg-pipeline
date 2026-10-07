import pydicom
from pathlib import Path
from collections import Counter

case = Path("~/projects/dicom-seg-pipeline/data/mama-mia/dicom/DUKE_040").expanduser()

for phase in sorted(case.iterdir()):
    files = sorted(phase.glob("*.dcm"))
    ds = [pydicom.dcmread(f, stop_before_pixels=True) for f in files]
    z = sorted(float(d.ImagePositionPatient[2]) for d in ds)
    gaps = {round(b - a, 3) for a, b in zip(z, z[1:])}
    print(f"{phase.name}: {len(ds)} plikow | "
          f"study {len({d.StudyInstanceUID for d in ds})} | "
          f"series {len({d.SeriesInstanceUID for d in ds})} | "
          f"unikalnych Z: {len(set(z))} | odstepy: {sorted(gaps)[:3]}")

studies = {pydicom.dcmread(next(p.glob("*.dcm")), stop_before_pixels=True).StudyInstanceUID
           for p in sorted(case.iterdir())}
print("wspolnych study:", len(studies))   # ma byc 1
