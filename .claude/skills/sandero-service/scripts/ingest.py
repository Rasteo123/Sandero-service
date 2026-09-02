#!/usr/bin/env python3
"""Сборка поискового корпуса скилла.

    python3 scripts/ingest.py            # пересобрать data/chunks.jsonl
    python3 scripts/ingest.py --stats    # что получилось

Источники описаны в data/sources.json. PDF читаются через pypdf
(`pip install -r requirements.txt`), подписи к иллюстрациям берутся
из data/images.json.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from ragkit.chunker import CHAPTERS_EN, CHAPTERS_RU, chunk_page, parse_page
from ragkit.store import CHUNKS_PATH, CORPUS_DIR, DATA_DIR, SOURCES_PATH

# Оглавления и алфавитный указатель только зашумляют выдачу.
TOC_RATIO_SKIP = 0.5
INDEX_CHAPTER = 7


def read_pdf_pages(path: Path) -> list[str]:
    try:
        from pypdf import PdfReader
    except ImportError:  # pragma: no cover - зависит от окружения
        raise SystemExit(
            "Нужен pypdf: pip install -r requirements.txt\n"
            "(он требуется только для пересборки корпуса, поиск работает без него)"
        )
    reader = PdfReader(str(path))
    return [page.extract_text() or "" for page in reader.pages]


def ingest_pdf(source: dict, path: Path) -> list[dict]:
    chapters = CHAPTERS_RU if source.get("lang") == "ru" else CHAPTERS_EN
    records: list[dict] = []
    last_section = None
    last_section_page = -99

    for page_number, raw in enumerate(read_pdf_pages(path), start=1):
        page = parse_page(raw)
        if not page["text"].strip():
            continue
        if page["toc_ratio"] >= TOC_RATIO_SKIP:
            continue
        if page["chapter"] == INDEX_CHAPTER:
            continue
        # Раздел переносится только на соседнюю страницу: продолжение разворота
        # своего заголовка не имеет, а через страницу это уже другая тема.
        if page["section"]:
            last_section = page["section"]
            last_section_page = page_number
        inherited = not page["section"] and page_number - last_section_page <= 1
        section = page["section"] or (last_section if inherited else None)

        for position, text in enumerate(chunk_page(page)):
            records.append(
                {
                    "id": f"{source['doc']}:p{page_number}:{position}",
                    "doc": source["doc"],
                    "doc_title": source["title"],
                    "lang": source.get("lang", "ru"),
                    "kind": "manual",
                    "pdf_page": page_number,
                    "page_label": page["page_label"],
                    "chapter": page["chapter"],
                    "chapter_title": chapters.get(page["chapter"] or -1),
                    "section": section,
                    "section_inherited": inherited or None,
                    "section_src": page["section_src"],
                    "text": text,
                }
            )
    return records


def ingest_images(images_path: Path) -> list[dict]:
    """Иллюстрации ищутся по описанию: сам JPG искать нечем.

    Длинные описания (таблицы предохранителей) режутся так же, как страницы
    PDF, — иначе нормировка BM25 по длине занижает их в выдаче.
    """
    if not images_path.exists():
        return []
    payload = json.loads(images_path.read_text(encoding="utf-8"))
    records = []
    for item in payload["images"]:
        page = {"text": item.get("description", "")}
        for position, text in enumerate(chunk_page(page)) or [(0, page["text"])]:
            records.append(
                {
                    "id": f"{payload['doc']}:{item['file']}:{position}",
                    "doc": payload["doc"],
                    "doc_title": payload["title"],
                    "lang": payload.get("lang", "ru"),
                    "kind": "image",
                    "image": item["file"],
                    "drive_url": item.get("drive_url"),
                    "section": item.get("title"),
                    "chapter_title": item.get("group"),
                    # Ключевые слова индексируются, но в ответ не показываются.
                    "labels": item.get("labels"),
                    "text": text,
                }
            )
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
        path = CORPUS_DIR / source["file"]
        if not path.exists():
            skipped.append(f"{source['file']} (нет локальной копии, см. scripts/drive_import.py)")
            continue
        found = ingest_pdf(source, path)
        records.extend(found)
        print(f"{source['doc']}: {len(found)} чанков из {path.name}")

    images = ingest_images(DATA_DIR / "images.json")
    if images:
        records.extend(images)
        print(f"иллюстрации: {len(images)} чанков")

    if not records:
        raise SystemExit(
            "Нечего индексировать. Восстановите исходники: см. scripts/drive_import.py"
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as fh:
        for record in records:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"\nЗаписано {len(records)} чанков -> {args.out}")

    if skipped:
        print("Пропущено: " + ", ".join(skipped))
    if args.stats:
        by_doc = Counter(r["doc"] for r in records)
        for doc, count in by_doc.most_common():
            print(f"  {doc:24} {count:5} чанков")
        lengths = [len(r["text"]) for r in records]
        print(f"  средняя длина чанка: {sum(lengths) // len(lengths)} символов")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
