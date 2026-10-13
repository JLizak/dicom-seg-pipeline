# dicom-seg-pipeline

End-to-end breast tumour segmentation in DCE-MRI, wired into a PACS: a study goes
into Orthanc, the model's result comes back as DICOM-SEG / DICOM-SR and shows up as
an overlay in the OHIF viewer.

This is a learn-by-building project on public data (MAMA-MIA), built along production
patterns. It is not a clinical deployment. The full work plan, written in Polish, is
in [`Przewodnik_DICOM_PACS_pipeline.md`](Przewodnik_DICOM_PACS_pipeline.md).

> **Status: work in progress.** The Orthanc + OHIF stack, the Orthanc REST client and
> the MAMA-MIA NIfTI → DICOM conversion are in place. Loading a DICOM series into a 3D
> volume is in progress; DICOM-SEG export and training are still skeletons.

## Why DICOM in, DICOM out

In a clinical setting a model does not output a PNG with a mask. The result of an AI
model is itself a DICOM object, so that it can go back to the PACS and be displayed in
any viewer:

- **DICOM-SEG** — the segmentation mask as a DICOM object that references the source
  series, which is what makes the overlay land on the right image.
- **DICOM-SR** — a structured, machine-readable report with values such as lesion
  volume, number of foci and model confidence.

Two architectural rules run through the whole project:

- **One contract.** A series in the PACS goes in, DICOM-SEG and DICOM-SR in the PACS
  come out. Everything in between (queue, worker, model) can be swapped without
  touching that contract.
- **Two tracks.** Training runs offline on prepared volumes (usually NIfTI), serving
  runs on DICOM. They share only the model artifact (weights + configuration) and the
  preprocessing code.

## Target architecture

Runtime:

```
[modality / storescu]
        │ C-STORE (DIMSE)
        ▼
   ┌─────────┐  OnStableStudy  ┌─────────┐  job  ┌──────────┐  gRPC  ┌──────────────┐
   │ Orthanc │ ──────────────► │  queue  │ ────► │  worker  │ ─────► │    Triton    │
   │ (PACS)  │                 └─────────┘       │ (DICOM,  │        │  champion +  │
   │         │ ◄── STOW-RS: SEG + SR ─────────── │ pre/post)│        │  challenger  │
   └────┬────┘     (champion only)               └────┬─────┘        └──────────────┘
        │ DICOMweb (QIDO/WADO/STOW)                   │ metrics, drift features,
        ▼                                             │ shadow results
   ┌─────────┐                                        ▼
   │  OHIF   │  ← overlay + report         [Prometheus / Grafana / Evidently]
   └────┬────┘
        │ radiologist's correction (SEG, MANUAL) → Orthanc → [collector] → data/feedback (DVC)

   [API / orchestrator + Postgres]  ← receives the event from Orthanc, stores the job,
                                      puts it on the queue; keeps statuses and audit trail
```

Model lifecycle:

```
  data + splits (DVC)  ◄─────────────────────────────────┐
        │                                                │
        ▼                                                │
  training + evaluation (dvc repro → MLflow)             │
        │                                                │
        ▼                                                │  retraining
  MLflow Registry: version N                             │  (new labels,
        │                                                │   drift alert
        ▼                                                │   or manual)
  CI gate: golden set + geometry test ──(no)──► stop     │
        │ (yes) → alias: challenger                      │
        ▼                                                │
  shadow: challenger runs next to the champion           │
        │                                                │
        ▼                                                │
  human decision → alias: champion → PACS                │
        │                                                │
        ▼                                                │
  drift monitoring + radiologist's corrections ──────────┘
```

## Roadmap

**Part I — clinical pipeline.** A study enters the PACS, the model's result returns as
DICOM-SEG/SR and is displayed in OHIF.

| Phase | Scope |
|---|---|
| 0 | Mental model: DICOM data hierarchy, PACS, DIMSE vs DICOMweb, SEG / SR objects |
| 1 | Run Orthanc and push data through it |
| 2 | Connect the OHIF viewer |
| 3 | The core: a model that takes DICOM and produces DICOM-SEG |
| 4 | Close the loop: automatic triggering, job queue, job database |
| 5 | DICOM-SR: a report with numbers |
| 6 | Production layer: CI/CD, observability, drift monitoring, audit trail, security |

**Part II — model lifecycle (ML and MLOps).** Where the model comes from, how you know
it is good, how to replace it safely and how to notice it has stopped working.

| Phase | Scope |
|---|---|
| 7 | Data as code: per-patient splits, DVC, MLflow tracking, reproducible training |
| 8 | Trustworthy evaluation: metrics, bootstrap confidence intervals, external test set, nnU-Net baseline |
| 9 | Model registry, quality gate in CI, geometry regression test, shadow deployment |
| 10 | Model uncertainty and fast inference (ONNX → TensorRT, Triton) |
| 11 | Monitoring, radiologist feedback loop, active learning, retraining |

**Part III — portfolio.** Phase 12: write-up, demos and the path to production.

## Repository layout

```
deploy/            Orthanc and OHIF configuration (mounted by docker-compose.yml)
src/dcmseg/        package code
  pacs/            Orthanc REST client
  io/              NIfTI → DICOM conversion, DICOM series → 3D volume
  models/          model and training code
  export/          mask → DICOM-SEG (highdicom)
configs/           training configs (local_cpu.yaml, kaggle_gpu.yaml)
scripts/           small CLI tools
notebooks/         data exploration
tests/             pytest
docs/              guide, journal, notes
data/              (gitignored) raw DICOM, NIfTI
experiments/       (gitignored) checkpoints and metrics
```

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu   # laptop without an NVIDIA GPU
pip install -e ".[train,dicom,dev]"
```

> If `PYTHONPATH` is set in your shell (e.g. by ROS), it shadows the packages from
> `.venv`. Run `unset PYTHONPATH` before working in this project.

## PACS + viewer

```bash
docker compose up -d     # Orthanc: http://localhost:8042, OHIF: http://localhost:3000
python scripts/explore.py
```

The Orthanc client reads `ORTHANC_URL`, `ORTHANC_USER` and `ORTHANC_PASSWORD` from the
environment; see [`.env.example`](.env.example).

Load a series stored in Orthanc as a 3D volume:

```bash
python -m dcmseg.io.series <orthanc-series-id>
```

## Tests

```bash
python -m pytest
```

## Data and limitations

The project uses the public [MAMA-MIA](https://www.synapse.org/Synapse:syn60868042)
breast DCE-MRI dataset, not hospital data. It does not cover RIS/HIS integration over
HL7/FHIR, certification (MDR, FDA), work on real patient data or traffic from real
scanners across multiple sites.
