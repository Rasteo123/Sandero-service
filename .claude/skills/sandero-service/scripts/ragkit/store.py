"""Пути скилла и загрузка корпуса."""

from __future__ import annotations

import json
from pathlib import Path

SKILL_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = SKILL_ROOT / "data"
CHUNKS_PATH = DATA_DIR / "chunks.jsonl"
SYNONYMS_PATH = DATA_DIR / "synonyms.json"
SOURCES_PATH = DATA_DIR / "sources.json"
VEHICLE_PATH = DATA_DIR / "vehicle.json"
# Исходники (PDF/JPG) в git не хранятся — их восстанавливает scripts/drive_import.py
CORPUS_DIR = SKILL_ROOT.parents[2] / "corpus"


def load_chunks(path: Path | None = None) -> list[dict]:
    path = path or CHUNKS_PATH
    if not path.exists():
        raise SystemExit(
            f"Корпус не найден: {path}\n"
            "Соберите его: python3 scripts/ingest.py (см. README)."
        )
    chunks = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                chunks.append(json.loads(line))
    return chunks


def load_synonyms(path: Path | None = None) -> dict[str, list[str]]:
    path = path or SYNONYMS_PATH
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def load_vehicle(path: Path | None = None) -> dict:
    """Справочник применимости: какие двигатели покрывает каждый документ."""
    path = path or VEHICLE_PATH
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def load_sources(path: Path | None = None) -> dict:
    path = path or SOURCES_PATH
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def build_index(chunks: list[dict] | None = None):
    from .bm25 import Bm25Index

    return Bm25Index(chunks if chunks is not None else load_chunks(), load_synonyms())


def _span(start, end, prefix: str) -> str | None:
    """«стр. 4.6-4.9» для единицы на несколько страниц, «стр. 4.7» для одной."""
    if start is None:
        return None
    if end is None or str(end) == str(start):
        return f"{prefix} {start}"
    return f"{prefix} {start}-{end}"


def citation(chunk: dict) -> str:
    """Человекочитаемая ссылка на источник — то, что скилл обязан показывать."""
    bits = [str(chunk.get("doc_title", chunk.get("doc", "?")))]
    section = chunk.get("section")
    if section:
        bits.append(str(section))
    label = _span(chunk.get("page_label"), chunk.get("page_label_end"), "стр.")
    if label:
        bits.append(label)
    pdf = _span(chunk.get("pdf_page"), chunk.get("pdf_page_end"), "PDF-стр.")
    if pdf:
        bits.append(pdf)
    image = chunk.get("image")
    if image:
        bits.append(f"иллюстрация {image}")
    parts = chunk.get("unit_parts") or 1
    if parts > 1:
        bits.append(f"фрагмент {int(chunk['id'].rsplit(':', 1)[1]) + 1} из {parts}")
    return " — ".join(bits)
