"""Trening segmentacji 3D. Ten sam skrypt lokalnie (CPU) i na GPU w chmurze.

    python -m dcmseg.train --config configs/local_cpu.yaml
    python -m dcmseg.train --config configs/kaggle_gpu.yaml --resume

Każda epoka zapisuje `last.pt` (do wznowienia po zerwanej sesji Kaggle/Colab),
najlepsza walidacja — `best.pt`, a metryki lecą do `metrics.jsonl`.
"""

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from monai.inferers import sliding_window_inference
from monai.losses import DiceCELoss
from monai.metrics import DiceMetric
from monai.transforms import AsDiscrete
from tqdm import tqdm

from dcmseg.config import load_config
from dcmseg.data import build_loaders
from dcmseg.models import build_model


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def validate(model, loader, cfg, device, use_amp) -> float:
    model.eval()
    metric = DiceMetric(include_background=False, reduction="mean")
    to_pred = AsDiscrete(argmax=True, to_onehot=cfg["model"]["out_channels"])
    to_label = AsDiscrete(to_onehot=cfg["model"]["out_channels"])
    with torch.no_grad(), torch.autocast(device.type, enabled=use_amp):
        for batch in loader:
            image, label = batch["image"].to(device), batch["label"].to(device)
            logits = sliding_window_inference(
                image, cfg["data"]["patch_size"], cfg["train"].get("sw_batch_size", 1), model, overlap=0.25
            )
            metric(y_pred=[to_pred(p) for p in logits], y=[to_label(y) for y in label])
    return metric.aggregate().item()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--set", nargs="*", default=[], metavar="KLUCZ=WARTOŚĆ", help="nadpisz pola configu")
    parser.add_argument("--resume", action="store_true", help="wznów z last.pt w katalogu eksperymentu")
    args = parser.parse_args()

    cfg = load_config(args.config, args.set)
    t = cfg["train"]
    seed_everything(t.get("seed", 0))
    device = resolve_device(t.get("device", "auto"))
    use_amp = bool(t.get("amp", False)) and device.type == "cuda"

    out_dir = Path(cfg.get("output_dir", "experiments")) / cfg["experiment"]
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "config.yaml", "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)

    train_loader, val_loader = build_loaders(cfg)
    m = cfg["model"]
    model = build_model(m["name"], cfg["data"]["in_channels"], m["out_channels"], **m.get("kwargs", {})).to(device)
    loss_fn = DiceCELoss(to_onehot_y=True, softmax=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=t["lr"], weight_decay=t.get("weight_decay", 1e-5))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=t["epochs"])
    scaler = torch.amp.GradScaler(enabled=use_amp)

    start_epoch, best_dice = 0, -1.0
    if args.resume and (out_dir / "last.pt").exists():
        ckpt = torch.load(out_dir / "last.pt", map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        scaler.load_state_dict(ckpt["scaler"])
        start_epoch, best_dice = ckpt["epoch"] + 1, ckpt["best_dice"]
        print(f"Wznawiam od epoki {start_epoch} (best Dice {best_dice:.4f})")

    print(f"Urządzenie: {device}, AMP: {use_amp}, parametry: {sum(p.numel() for p in model.parameters()):,}")
    for epoch in range(start_epoch, t["epochs"]):
        model.train()
        epoch_loss, t0 = 0.0, time.time()
        for batch in tqdm(train_loader, desc=f"epoka {epoch}", leave=False):
            image, label = batch["image"].to(device), batch["label"].to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device.type, enabled=use_amp):
                loss = loss_fn(model(image), label)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            epoch_loss += loss.item()
        scheduler.step()

        record = {
            "epoch": epoch,
            "loss": epoch_loss / len(train_loader),
            "lr": scheduler.get_last_lr()[0],
            "time_s": round(time.time() - t0, 1),
        }
        if (epoch + 1) % t.get("val_every", 1) == 0 or epoch + 1 == t["epochs"]:
            record["val_dice"] = validate(model, val_loader, cfg, device, use_amp)
            if record["val_dice"] > best_dice:
                best_dice = record["val_dice"]
                torch.save({"model": model.state_dict(), "config": cfg, "epoch": epoch}, out_dir / "best.pt")
        print(json.dumps(record))
        with open(out_dir / "metrics.jsonl", "a") as f:
            f.write(json.dumps(record) + "\n")

        torch.save(
            {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "scaler": scaler.state_dict(),
                "epoch": epoch,
                "best_dice": best_dice,
                "config": cfg,
            },
            out_dir / "last.pt",
        )
    print(f"Koniec. Najlepszy Dice walidacyjny: {best_dice:.4f} → {out_dir / 'best.pt'}")


if __name__ == "__main__":
    main()
