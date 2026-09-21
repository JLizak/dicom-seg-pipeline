"""Wygeneruj mały syntetyczny zbiór (2 kanały „DCE” + kulista „zmiana”) do testów pipeline'u.

Służy do sprawdzenia, że trening działa end-to-end na CPU, zanim zapłacisz za GPU:
model powinien szybko dojść do wysokiego Dice. Jeśli nie dochodzi — błąd jest w kodzie,
nie w danych.

    python scripts/make_synthetic_data.py --out data/synthetic
"""

import argparse
import json
from pathlib import Path

import nibabel as nib
import numpy as np


def make_case(rng: np.random.Generator, shape=(96, 96, 48), spacing=(1.0, 1.0, 2.0)):
    zz, yy, xx = np.meshgrid(*[np.arange(s) for s in shape], indexing="ij")
    center = [rng.integers(s // 4, 3 * s // 4) for s in shape]
    radius = rng.uniform(5, 10)
    # Promień w mm, więc uwzględniamy anizotropowy spacing.
    dist = np.sqrt(sum(((g - c) * sp) ** 2 for g, c, sp in zip((zz, yy, xx), center, spacing)))
    label = (dist < radius).astype(np.uint8)

    tissue = 100 + 20 * rng.standard_normal(shape)
    pre = tissue
    post = tissue + 80 * label + 10 * rng.standard_normal(shape)  # zmiana „wzmacnia się” po kontraście
    affine = np.diag([*spacing, 1.0])
    return pre.astype(np.float32), post.astype(np.float32), label, affine


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="data/synthetic")
    parser.add_argument("--n-train", type=int, default=6)
    parser.add_argument("--n-val", type=int, default=2)
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    entries = []
    for i in range(args.n_train + args.n_val):
        pre, post, label, affine = make_case(rng)
        names = {k: f"case{i:03d}_{k}.nii.gz" for k in ("pre", "post1", "seg")}
        for key, arr in zip(("pre", "post1", "seg"), (pre, post, label)):
            nib.save(nib.Nifti1Image(arr, affine), out / names[key])
        entries.append({"image": [names["pre"], names["post1"]], "label": names["seg"]})

    datalist = {"training": entries[: args.n_train], "validation": entries[args.n_train :]}
    (out / "datalist.json").write_text(json.dumps(datalist, indent=2))
    print(f"Zapisano {len(entries)} przypadków i datalist.json w {out}")


if __name__ == "__main__":
    main()
