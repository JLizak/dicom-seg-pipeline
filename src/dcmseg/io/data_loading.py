"""

    data/
    ├── images/
    │   ├── DUKE_009/
    │   │   ├── DUKE_009_0000.nii.gz   <- faza pre-kontrastowa
    │   │   ├── DUKE_009_0001.nii.gz   <- 1. faza post-kontrastowa
    │   │   ├── DUKE_009_0002.nii.gz   <- 2. faza post
    │   │   └── DUKE_009_0003.nii.gz   <- 3. faza post
    │   └── NACT_45/
    │       ├── NACT_45_0000.nii.gz
    │       ├── NACT_45_0001.nii.gz
    │       └── NACT_45_0002.nii.gz    <- tylko 3 fazy!
    └── segmentations/
        ├── DUKE_009.nii.gz            <- maski leżą płasko
        └── NACT_45.nii.gz

"""

from __future__ import annotations

import csv
import random
import re
from collections import Counter
from collections.abc import Collection, Sequence
from pathlib import Path
from dcmseg.settings import DATA_DIR

class CaseNotFoundError(Exception):
    """Folder przypadku nie zawiera oczekiwanych plików NIfTI."""


PHASE_RE = re.compile(r"_(\d{4})\.nii\.gz$")


def parse_phase_index(path: Path) -> int:
    m = PHASE_RE.search(path.name)
    if m is None:
        raise CaseNotFoundError()
    return int(m.group(1))


def find_phases(
    case_dir: Path,
    phases: Sequence[int] | None = None,
    pattern: str = "*.nii.gz",
) -> list[Path]:
    
    found = {}
    for f in case_dir.glob(pattern):
        found[parse_phase_index(f)] = f

    if not found:
        raise CaseNotFoundError()

    if phases is None:
        return [found[i] for i in sorted(found)]

    missing = [p for p in phases if p not in found]
    if missing:
        raise CaseNotFoundError(
            f"{case_dir}: brak faz {missing}; dostępne: {sorted(found)}"
        )
    return [found[p] for p in phases]

def _is_junk(name: str) -> bool:
    """Katalogi służbowe, które nie są przypadkami."""
    return name.startswith((".", "__"))

def _scan_images(
    root: Path,
    phases: Sequence[int] | None,
    pattern: str,
) -> dict[str, list[Path]]:
    """Mapuje nazwę folderu przypadku -> lista plików faz."""
    if not root.is_dir():
        raise FileNotFoundError(f"Nie znaleziono katalogu: {root.resolve()}")

    return {
        d.name: find_phases(d, phases, pattern)
        for d in sorted(root.iterdir())
        if d.is_dir() and not _is_junk(d.name)
    }


def strip_nifti_suffix(name: str) -> str:
    """DUKE_009.nii.gz -> DUKE_009 (Path.stem zdjęłoby tylko .gz)."""
    for suffix in (".nii.gz", ".nii"):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return name


def _scan_segs(
    root: Path,
    pattern: str,
    known_ids: Collection[str] | None = None,
) -> dict[str, list[Path]]:

    if not root.is_dir():
        raise FileNotFoundError(f"Nie znaleziono katalogu: {root.resolve()}")

    known = set(known_ids or ())
    out: dict[str, list[Path]] = {}

    # układ z podfolderami (pomijając .ipynb_checkpoints, __MACOSX itp.)
    for d in sorted(root.iterdir()):
        if not d.is_dir() or _is_junk(d.name):
            continue
        if files := sorted(d.glob(pattern)):
            out[d.name] = files

    # układ płaski — czytany zawsze, niezależnie od obecności podfolderów
    for f in sorted(root.glob(pattern)):
        cid = strip_nifti_suffix(f.name)
        if cid not in known:
            stripped = re.sub(r"_\d{4}$", "", cid)
            if stripped in known:
                cid = stripped
        out.setdefault(cid, []).append(f)

    return out

def build_data_dicts(
    data_dir: Path,
    images_subdir: str = "images",
    segs_subdir: str = "segmentations",
    phases: Sequence[int] | None = (0, 1),
    pattern: str = "*.nii.gz",
    as_str: bool = False,
    strict: bool = False,
) -> list[dict]:
 
    phases = list(phases) if phases is not None else None
    image_map = _scan_images(data_dir / images_subdir, phases, pattern)
    seg_map = _scan_segs(data_dir / segs_subdir, pattern, image_map.keys())
    
    paired = sorted(image_map.keys() & seg_map.keys())

    if not paired:
        img_dir = data_dir / images_subdir
        seg_dir = data_dir / segs_subdir
        raise CaseNotFoundError(
            "Nie znaleziono ani jednej pary.\n"
            f"  obrazy  ({img_dir}): {len(image_map)} ID, "
            f"np. {sorted(image_map)[:3]}\n"
            f"  maski   ({seg_dir}): {len(seg_map)} ID, "
            f"np. {sorted(seg_map)[:3]}\n"
            f"  pliki w katalogu masek: "
            f"{[f.name for f in sorted(seg_dir.iterdir())[:3]]}\n"
            "ID muszą być identyczne po obu stronach: w obrazach to nazwa "
            "podfolderu, w maskach nazwa pliku (bez .nii.gz) albo nazwa podfolderu."
        )

    for cid in paired:
        if len(seg_map[cid]) != 1:
            raise CaseNotFoundError(
                f"{cid}: znaleziono {len(seg_map[cid])} masek, oczekiwano 1"
            )

    conv = str if as_str else (lambda p: p)
    data = []
    for cid in paired:
        case_files = image_map[cid]
        data.append(
            {
                "image": (
                    [conv(p) for p in case_files]
                    if len(case_files) > 1
                    else conv(case_files[0])
                ),
                "label": conv(seg_map[cid][0]),
                "case_id": cid,
            }
        )

    return data


def read_split_csv(
    csv_path: str | Path,
    train_col: str = "train_split",
    test_col: str = "test_split",
) -> tuple[list[str], list[str]]:

    csv_path = DATA_DIR / csv_path
    if not csv_path.is_file():
        raise FileNotFoundError(f"Nie znaleziono pliku podziału: {csv_path.resolve()}")

    with csv_path.open(newline="", encoding="utf-8-sig") as fh:
        reader = csv.DictReader(fh)
        if reader.fieldnames is None:
            raise ValueError(f"{csv_path}: pusty plik")
        missing = {train_col, test_col} - set(reader.fieldnames)
        if missing:
            raise ValueError(
                f"{csv_path}: brak kolumn {sorted(missing)}; "
                f"są: {reader.fieldnames}"
            )
        train, test = [], []
        for row in reader:
            if cid := (row[train_col] or "").strip():
                train.append(cid)
            if cid := (row[test_col] or "").strip():
                test.append(cid)

    overlap = set(train) & set(test)
    if overlap:
        raise ValueError(
            f"{csv_path}: {len(overlap)} ID w obu kolumnach, np. "
            f"{sorted(overlap)[:5]} — podział jest nieważny"
        )
    return train, test


def split_by_csv(
    data: list[dict],
    csv_path: str | Path = 'train_test_splits.csv',
    train_col: str = "train_split",
    test_col: str = "test_split",
    verbose: bool = True,
) -> tuple[list[dict], list[dict]]:
    
    csv_path = DATA_DIR / csv_path
    train_ids, test_ids = read_split_csv(csv_path, train_col, test_col)
    by_id = {d["case_id"]: d for d in data}

    train = [by_id[cid] for cid in train_ids if cid in by_id]
    test = [by_id[cid] for cid in test_ids if cid in by_id]

    if verbose:
        on_disk = set(by_id)
        in_csv = set(train_ids) | set(test_ids)
        print(
            f"[split_by_csv] train: {len(train)}/{len(train_ids)} | "
            f"test: {len(test)}/{len(test_ids)}"
        )
        if brak := in_csv - on_disk:
            print(f"[uwaga] {len(brak)} ID z CSV nie ma na dysku")
        if nadmiar := on_disk - in_csv:
            print(
                f"[uwaga] {len(nadmiar)} przypadków z dysku nie ma w CSV "
                f"(pominięte): {sorted(nadmiar)[:5]}"
            )

    if not train:
        raise CaseNotFoundError(
            "Żadne ID treningowe z CSV nie pasuje do przypadków na dysku — "
            "sprawdź, czy nazwy folderów zgadzają się z kolumną CSV"
        )
    return train, test




if __name__ == "__main__":

    data = build_data_dicts(DATA_DIR)
    
    csv_path = DATA_DIR / "train_test_splits.csv"
   
    if csv_path.is_file():
        train, test = split_by_csv(data, csv_path)
