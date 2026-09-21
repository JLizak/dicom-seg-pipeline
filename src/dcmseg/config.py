"""Wczytywanie konfiguracji YAML z nadpisywaniem z linii poleceń.

Ten sam kod treningu działa lokalnie i na Kaggle/chmurze — różni się tylko config
(ścieżki, rozmiar patcha, urządzenie). Pojedyncze pola nadpiszesz bez edycji pliku:

    python -m dcmseg.train --config configs/local_cpu.yaml --set train.epochs=5
"""

from pathlib import Path

import yaml


def load_config(path: str | Path, overrides: list[str] | None = None) -> dict:
    with open(path) as f:
        cfg = yaml.safe_load(f)
    for item in overrides or []:
        key, _, raw = item.partition("=")
        if not _:
            raise ValueError(f"Nadpisanie musi mieć postać klucz=wartość, dostałem: {item!r}")
        _set_nested(cfg, key.split("."), yaml.safe_load(raw))
    return cfg


def _set_nested(cfg: dict, keys: list[str], value) -> None:
    node = cfg
    for k in keys[:-1]:
        node = node.setdefault(k, {})
    node[keys[-1]] = value
