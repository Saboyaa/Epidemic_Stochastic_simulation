"""Escrita de tabelas em CSV e Markdown (sem dependências extras)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .config import TAB_DIR, DATA_PROC


def _fmt(v) -> str:
    if isinstance(v, (bool, np.bool_)):
        return "sim" if v else "não"
    if isinstance(v, (float, np.floating)):
        if np.isnan(v):
            return "—"
        if v == 0:
            return "0"
        a = abs(v)
        if a >= 1000:
            return f"{v:,.1f}"
        if a >= 10:
            return f"{v:.2f}"
        if a >= 0.01:
            return f"{v:.4f}"
        return f"{v:.3g}"
    return str(v)


def to_markdown(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(_fmt(row[c]) for c in cols) + " |")
    return "\n".join(lines) + "\n"


def write_table(df: pd.DataFrame, name: str, tab_dir: Path = TAB_DIR) -> dict:
    tab_dir.mkdir(parents=True, exist_ok=True)
    csv = tab_dir / f"{name}.csv"
    md = tab_dir / f"{name}.md"
    df.to_csv(csv, index=False)
    md.write_text(to_markdown(df), encoding="utf-8")
    return dict(csv=str(csv), md=str(md))


def save_json(obj, name: str, out_dir: Path = DATA_PROC):
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / f"{name}.json", "w") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, default=_json_default)


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.bool_,)):
        return bool(o)
    raise TypeError(str(type(o)))


def load_json(name: str, out_dir: Path = DATA_PROC):
    with open(out_dir / f"{name}.json") as f:
        return json.load(f)
