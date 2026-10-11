import pyplastimatch as pypla
from pyplastimatch.utils.install import install_precompiled_binaries
from pathlib import Path
import os
import pydicom
from pydicom.uid import generate_uid

PROJECT_ROOT = Path(__file__).resolve().parents[0]

NIFTI_DIR = PROJECT_ROOT / "data/mama-mia/raw/images"
DICOM_DIR = PROJECT_ROOT / "data/mama-mia/dicom"

def convert_case(case_dir):
    case = case_dir.name
    destination = DICOM_DIR / case
    
    if destination.exists():
        return

    for nii in sorted(case_dir.glob("*.nii.gz")):
        phase = nii.name.split("_")[-1].split(".")[0]

        pypla.convert(
            verbose=False,
            input=str(nii),
            **{"patient-id": case, "output-dicom": str(destination / phase)},
                )
    study_uid = generate_uid()
    for phase_dir in sorted(destination.iterdir()):
        series_uid = generate_uid()
        num = int(phase_dir.name)

        for i, dcm in enumerate(sorted(phase_dir.glob("*.dcm")), start=1):
            ds = pydicom.dcmread(dcm)
            ds.StudyInstanceUID = study_uid
            ds.SeriesInstanceUID = series_uid
            ds.SeriesNumber = num
            ds.InstanceNumber = i
            ds.SeriesDescription = f"DCE  phase {num}"
            ds.PatientID = ds.PatientName = case
            ds.StudyDescription = "MAMA-MIA"
            ds.save_as(dcm)
    print(f"ok {case}")

for case_dir in sorted(d for d in NIFTI_DIR.iterdir() if d.is_dir()):
    convert_case(case_dir)

