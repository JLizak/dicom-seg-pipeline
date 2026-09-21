"""Maska → DICOM-SEG odwołujący się do serii źródłowej (Faza 3.2 ścieżka B, 3.4).

Szkielet do samodzielnego uzupełnienia. Enkodera nie pisz od zera — użyj highdicom.
"""

import numpy as np
import pydicom


def mask_to_dicom_seg(mask: np.ndarray, source_images: list[pydicom.Dataset], model_version: str) -> pydicom.Dataset:
    """Zbuduj obiekt DICOM-SEG z maski binarnej o kształcie [z, y, x].

    `mask` musi być w tej samej kolejności przekrojów co `source_images`
    (zwrócone przez `dcmseg.io.series.load_series`).
    """
    # TODO: highdicom.seg.SegmentDescription — nazwa segmentu, kod SNOMED CT
    #       (kategoria i typ, np. "Tissue"/"Neoplasm"), algorithm_type AUTOMATIC,
    #       AlgorithmIdentificationSequence z nazwą i `model_version`
    # TODO: highdicom.seg.Segmentation(source_images=..., pixel_array=..., segment_descriptions=[...],
    #       series_instance_uid=hd.UID(), sop_instance_uid=hd.UID(), ...)
    # TODO: sprawdź w OHIF, że nakładka nie jest przesunięta ani odbita (Zadanie 3.4)
    raise NotImplementedError
