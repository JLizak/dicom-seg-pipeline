"""Inferencja z checkpointu na wolumenach NIfTI.

    python -m dcmseg.infer --checkpoint experiments/<exp>/best.pt \
        --image pre.nii.gz post1.nii.gz --output maska.nii.gz

To wersja „badawcza”. W pipeline DICOM (Faza 3) wejściem będzie wolumen złożony
z serii przez `dcmseg.io.series`, a wyjściem DICOM-SEG z `dcmseg.export.dicom_seg`.
"""

import argparse

import nibabel as nib
import numpy as np
import torch
from monai.inferers import sliding_window_inference
from monai.transforms import AsDiscreted, Compose, Invertd

from dcmseg.data import val_transforms
from dcmseg.models import build_model
from dcmseg.train import resolve_device


def load_model(checkpoint: str, device: torch.device) -> tuple[torch.nn.Module, dict]:
    ckpt = torch.load(checkpoint, map_location=device)
    cfg = ckpt["config"]
    m = cfg["model"]
    model = build_model(m["name"], cfg["data"]["in_channels"], m["out_channels"], **m.get("kwargs", {}))
    model.load_state_dict(ckpt["model"])
    return model.to(device).eval(), cfg


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--image", nargs="+", required=True, help="fazy DCE w tej samej kolejności co w treningu")
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    device = resolve_device(args.device)
    model, cfg = load_model(args.checkpoint, device)
    pre = val_transforms(cfg)
    # Po predykcji cofamy resampling/orientację, żeby maska pasowała 1:1 do oryginału.
    post = Compose(
        [
            AsDiscreted(keys="pred", argmax=True),
            Invertd(keys="pred", transform=pre, orig_keys="image", nearest_interp=True),
        ]
    )

    data = pre({"image": args.image})
    with torch.no_grad():
        image = data["image"].unsqueeze(0).to(device)
        data["pred"] = sliding_window_inference(image, cfg["data"]["patch_size"], 1, model, overlap=0.5)[0].cpu()
    pred = post(data)["pred"]
    mask = pred[0].numpy().astype(np.uint8)
    nib.save(nib.Nifti1Image(mask, pred.affine.numpy()), args.output)
    print(f"Zapisano maskę {mask.shape}, voxele guza: {int(mask.sum())} → {args.output}")


if __name__ == "__main__":
    main()
