# dicom-seg-pipeline

Segmentacja raka piersi w DCE-MRI + pipeline DICOM → PACS (Orthanc) → OHIF.
Plan pracy: [`docs/Przewodnik_DICOM_PACS_pipeline.md`](docs/Przewodnik_DICOM_PACS_pipeline.md).

## Struktura

```
deploy/            konfiguracja Orthanc i OHIF (montowana przez docker-compose.yml)
src/dcmseg/        kod pakietu
  pacs/            klient REST Orthanca
  io/series.py     seria DICOM → wolumen 3D                  (Faza 3, TODO)
  data/            datalist JSON → transformy MONAI → DataLoader
  models/          fabryka architektur: segresnet | unet | dynunet | swinunetr
  train.py         trening (CPU lokalnie / GPU w chmurze — różni się tylko config)
  infer.py         inferencja z checkpointu na NIfTI
  export/          maska → DICOM-SEG (highdicom)             (Faza 3, TODO)
configs/           configi treningu (local_cpu.yaml, kaggle_gpu.yaml)
scripts/           małe narzędzia CLI (eksploracja, dane syntetyczne)
notebooks/         eksploracja danych
tests/             pytest
docs/              przewodnik, journal.md, notatki
data/              (gitignore) surowe DICOM, NIfTI, datalisty
experiments/       (gitignore) checkpointy i metryki
```

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu   # laptop bez GPU NVIDIA
pip install -e ".[train,dicom,dev]"
```

> Uwaga: jeśli w powłoce masz ustawiony `PYTHONPATH` (np. z ROS), przesłania on
> pakiety z `.venv`. Przed pracą w tym projekcie: `unset PYTHONPATH`.

## PACS + viewer

```bash
docker compose up -d     # Orthanc: http://localhost:8042, OHIF: http://localhost:3000
python scripts/explore.py
```

## Trening

Test „czy w ogóle działa” na CPU (dane syntetyczne, ok. 1–2 min):

```bash
python scripts/make_synthetic_data.py --out data/synthetic
python -m dcmseg.train --config configs/local_cpu.yaml
python -m dcmseg.infer --checkpoint experiments/segresnet_local_smoke/best.pt \
    --image data/synthetic/case006_pre.nii.gz data/synthetic/case006_post1.nii.gz --output pred.nii.gz
```

Prawdziwy trening (GPU): przygotuj datalistę dla MAMA-MIA w tym samym formacie
(`image` = lista faz DCE, `label` = maska), popraw ścieżki w `configs/kaggle_gpu.yaml`:

```bash
python -m dcmseg.train --config configs/kaggle_gpu.yaml            # pierwszy start
python -m dcmseg.train --config configs/kaggle_gpu.yaml --resume   # po zerwanej sesji
python -m dcmseg.train --config configs/kaggle_gpu.yaml --set model.name=dynunet experiment=dynunet_mamamia
```

## Testy

```bash
python -m pytest
```
