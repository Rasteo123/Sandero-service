#!/usr/bin/env python3
"""Сборка поискового корпуса скилла.

    python3 scripts/ingest.py            # пересобрать data/chunks.jsonl
    python3 scripts/ingest.py --stats    # что получилось

Корпус нарезается атомарно: единица — это законченный смысловой кусок
документа (процедура ремонта, раздел руководства, таблица на иллюстрации),
а не страница. Длинная единица дробится на части по границам абзацев, но
все части несут её название, поэтому фрагмент никогда не остаётся без
контекста, а ссылка указывает на диапазон страниц целиком.

Источники описаны в data/sources.json. Разбор PDF требует pymupdf
(`pip install -r requirements.txt`); поиску он не нужен.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from ragkit.chunker import (
    CHAPTERS_EN,
    CHAPTERS_RU,
    SM_TORQUE,
    best_title,
    chunk_unit,
    parse_page,
    parse_service_page,
    parse_toc,
    section_key,
    source_key,
    toc_lookup,
)
from ragkit.store import CHUNKS_PATH, CORPUS_DIR, DATA_DIR, SOURCES_PATH

CYRILLIC = re.compile(r"[а-яё]", re.IGNORECASE)

# Оглавления и алфавитный указатель только зашумляют выдачу.
TOC_RATIO_SKIP = 0.5
INDEX_CHAPTER = 7


def read_pdf(path: Path):
    """Возвращает (страницы, оглавление). Оглавление у сервисного мануала есть."""
    try:
        import pymupdf
    except ImportError:  # pragma: no cover - зависит от окружения
        raise SystemExit(
            "Нужен pymupdf: pip install -r requirements.txt\n"
            "(он требуется только для пересборки корпуса, поиск работает без него)"
        )
    with pymupdf.open(str(path)) as document:
        pages = [page.get_text("text") for page in document]
        toc = document.get_toc()
    return pages, toc


def owner_units(source: dict, path: Path) -> list[dict]:
    """Руководство по эксплуатации: единица — раздел со всеми продолжениями.

    Границу задаёт идентификатор раздела из исходной вёрстки («Ceintures de
    sécurité»), который типография оставила на каждой странице: он надёжнее
    заголовков, потому что на страницах с одними таблицами (крепление детских
    кресел) русского заголовка нет вовсе. Где такого идентификатора нет —
    в английском руководстве его не печатали — граница берётся по заголовку.
    """
    chapters = CHAPTERS_RU if source.get("lang") == "ru" else CHAPTERS_EN
    pages, _ = read_pdf(path)
    toc = parse_toc(pages)
    units: list[dict] = []
    current: dict | None = None

    for number, raw in enumerate(pages, start=1):
        page = parse_page(raw)
        if not page["text"].strip():
            continue
        if page["toc_ratio"] >= TOC_RATIO_SKIP or page["chapter"] == INDEX_CHAPTER:
            continue

        source_id = source_key(page["section_src"]) if page["section_src"] else None
        key = source_id or (section_key(page["section"]) if page["section"] else None)
        starts_new = current is None or (key is not None and key != current["key"])
        if starts_new:
            current = {
                "key": key,
                "doc": source["doc"],
                "doc_title": source["title"],
                "lang": source.get("lang", "ru"),
                "kind": "manual",
                "titles": [],
                "chapter": page["chapter"],
                "chapter_title": chapters.get(page["chapter"] or -1),
                "page_start": number,
                "label_start": page["page_label"],
                "system": source_id,
                "parts": [],
            }
            units.append(current)
        if page["section"]:
            current["titles"].append(page["section"])
        current["page_end"] = number
        current["label_end"] = page["page_label"] or current.get("label_end")
        current["parts"].append(page["text"])

    for unit in units:
        unit["text"] = "\n".join(unit.pop("parts"))
        # Название берём русское, если типография его напечатала; иначе
        # остаётся идентификатор раздела из вёрстки — он хотя бы точен.
        title, reliable = best_title(unit.pop("titles"))
        if not reliable:
            # Заголовка с меткой продолжения нет: собственное оглавление
            # документа надёжнее одиночной строки капсом, которая может
            # оказаться переносом предупреждения.
            title = toc_lookup(toc, unit.get("label_start")) or title
        # Титульная страница и предисловие своего заголовка не имеют.
        unit["title"] = title or unit["key"] or unit["chapter_title"] or chapters[0]
    return units


def service_units(source: dict, corpus_dir: Path) -> tuple[list[dict], list[str]]:
    """Сервисный мануал: единица — процедура целиком, по закладкам PDF.

    В закладках лежат название процедуры и страница её начала — это точнее
    любой эвристики по заголовкам в тексте. Процедура может продолжаться в
    следующей части файла, поэтому единица переносится через границу частей.
    """
    units: list[dict] = []
    missing: list[str] = []
    current: dict | None = None

    for part in source["parts"]:
        path = corpus_dir / part["file"]
        if not path.exists():
            missing.append(f"часть {part['part']}")
            current = None  # разрыв нумерации: продолжать процедуру нельзя
            continue

        pages, toc = read_pdf(path)
        starts = {page: title.strip() for _level, title, page in toc if page >= 1}

        for local, raw in enumerate(pages, start=1):
            page = parse_service_page(raw)
            global_page = part["page_offset"] + local
            if local in starts:
                current = {
                    "doc": source["doc"],
                    "doc_title": source["title"],
                    "lang": source.get("lang", "en"),
                    "kind": "service",
                    "title": starts[local],
                    "chapter": None,
                    "chapter_title": None,
                    "page_start": global_page,
                    "label_start": None,
                    "system": page["section_src"],
                    "part": part["part"],
                    "parts": [],
                }
                units.append(current)
            if current is None or not page["text"].strip():
                continue
            current["page_end"] = global_page
            current["system"] = current["system"] or page["section_src"]
            current["parts"].append(page["text"])

    for unit in units:
        unit["text"] = "\n".join(unit.pop("parts"))
    return [u for u in units if u.get("text", "").strip()], missing


def image_units(images_path: Path) -> list[dict]:
    """Иллюстрация — уже атомарная единица: одна таблица или одна легенда."""
    if not images_path.exists():
        return []
    payload = json.loads(images_path.read_text(encoding="utf-8"))
    return [
        {
            "doc": payload["doc"],
            "doc_title": payload["title"],
            "lang": payload.get("lang", "ru"),
            "kind": "image",
            "title": item.get("title"),
            "chapter": None,
            "chapter_title": item.get("group"),
            "image": item["file"],
            "drive_url": item.get("drive_url"),
            "labels": item.get("labels"),
            "text": item.get("description", ""),
        }
        for item in payload["images"]
    ]


def units_to_records(units: list[dict]) -> list[dict]:
    """Единицы -> чанки. Каждая часть единицы несёт её название и диапазон страниц."""
    records: list[dict] = []
    for number, unit in enumerate(units, start=1):
        pieces = chunk_unit(unit["text"])
        for position, text in enumerate(pieces):
            record = {
                "id": f"{unit['doc']}:u{number}:{position}",
                "doc": unit["doc"],
                "doc_title": unit["doc_title"],
                "lang": unit["lang"],
                "kind": unit["kind"],
                "section": unit.get("title"),
                "chapter": unit.get("chapter"),
                "chapter_title": unit.get("chapter_title"),
                "section_src": unit.get("system"),
                "unit_parts": len(pieces),
                "text": text,
            }
            for field, key in (
                ("pdf_page", "page_start"),
                ("pdf_page_end", "page_end"),
                ("page_label", "label_start"),
                ("page_label_end", "label_end"),
                ("part", "part"),
                ("image", "image"),
                ("drive_url", "drive_url"),
                ("labels", "labels"),
            ):
                if unit.get(key) is not None:
                    record[field] = unit[key]
            if SM_TORQUE.search(text):
                record["has_torque"] = True
            records.append(record)
    return records


def main() -> int:
    parser = argparse.ArgumentParser(description="Сборка корпуса для RAG-поиска")
    parser.add_argument("--out", type=Path, default=CHUNKS_PATH)
    parser.add_argument("--stats", action="store_true", help="показать статистику корпуса")
    args = parser.parse_args()

    sources = json.loads(SOURCES_PATH.read_text(encoding="utf-8"))
    records: list[dict] = []
    skipped: list[str] = []

    for source in sources["documents"]:
        if source.get("type") != "pdf" or not source.get("indexed"):
            if source.get("type") == "pdf":
                skipped.append(f"{source['doc']} ({source.get('reason', 'не индексируется')})")
            continue

        if source.get("format") == "service_manual":
            units, missing = service_units(source, CORPUS_DIR)
            if missing:
                skipped.append(f"{source['doc']}: нет локально — {', '.join(missing)}")
        else:
            path = CORPUS_DIR / source["file"]
            if not path.exists():
                skipped.append(f"{source['file']} (нет локальной копии, см. drive_import.py)")
                continue
            units = owner_units(source, path)

        found = units_to_records(units)
        records.extend(found)
        print(f"{source['doc']}: {len(units)} единиц -> {len(found)} чанков")

    images = image_units(DATA_DIR / "images.json")
    if images:
        found = units_to_records(images)
        records.extend(found)
        print(f"иллюстрации: {len(images)} единиц -> {len(found)} чанков")

    if not records:
        raise SystemExit("Нечего индексировать. Восстановите исходники: scripts/drive_import.py")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as fh:
        for record in records:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"\nЗаписано {len(records)} чанков -> {args.out}")

    if skipped:
        print("Пропущено: " + "; ".join(skipped))
    if args.stats:
        for doc, count in Counter(r["doc"] for r in records).most_common():
            single = sum(1 for r in records if r["doc"] == doc and r["unit_parts"] == 1)
            print(f"  {doc:16} {count:5} чанков, из них цельных единиц: {single}")
        lengths = [len(r["text"]) for r in records]
        print(f"  средняя длина чанка: {sum(lengths) // len(lengths)} символов")
        print(f"  с моментами затяжки: {sum(1 for r in records if r.get('has_torque'))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
