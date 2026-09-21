"""Seria DICOM → wolumen 3D (Faza 3.1 / 3.2 ścieżka B przewodnika).

Szkielet do samodzielnego uzupełnienia — to jest ta część, która uczy DICOM-a.
"""

from pathlib import Path

import numpy as np
import pydicom


def load_series(files: list[Path]) -> tuple[np.ndarray, np.ndarray, list[pydicom.Dataset]]:
    """Wczytaj instancje jednej serii i złóż je w wolumen.

    Zwraca: (wolumen [z, y, x], macierz affine 4x4 w mm, posortowane datasety).
    Datasety będą potrzebne później jako `source_images` dla DICOM-SEG.
    """
    # TODO: wczytaj pliki pydicom.dcmread
    # TODO: posortuj przekroje — NIE po nazwie pliku ani InstanceNumber, tylko po rzucie
    #       ImagePositionPatient na normalną płaszczyzny (iloczyn wektorowy dwóch
    #       wektorów z ImageOrientationPatient). Dlaczego? Zapisz w journal.md.
    # TODO: pixel_array * RescaleSlope + RescaleIntercept
    # TODO: zbuduj affine z PixelSpacing, odstępu między przekrojami i orientacji
    raise NotImplementedError


def select_series(series_datasets: dict[str, pydicom.Dataset]) -> str:
    """Wybierz SeriesInstanceUID właściwej fazy DCE (np. pre-contrast, 1. post-contrast).

    W Duke opisy serii (SeriesDescription) są różne między pacjentami — sprawdź, jakie
    reguły są odporne. To samo robi DICOMSeriesSelectorOperator w MONAI Deploy.
    """
    # TODO
    raise NotImplementedError
