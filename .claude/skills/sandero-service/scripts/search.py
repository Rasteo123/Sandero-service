#!/usr/bin/env python3
"""Поиск по руководствам Renault Sandero II.

    python3 scripts/search.py "как проверить уровень масла"
    python3 scripts/search.py "давление в шинах" --top 8 --lang ru
    python3 scripts/search.py "fuse box" --doc en_owner --json
    python3 scripts/search.py --list-sources

Работает на голой стандартной библиотеке python3: ни установок, ни сети.
"""

from __future__ import annotations

import argparse
import json
import sys
import textwrap
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from ragkit.bm25 import Bm25Index
from ragkit.store import citation, load_chunks, load_sources, load_synonyms

SNIPPET_WIDTH = 96


def render(hit, index: int, *, full: bool) -> str:
    chunk = hit.chunk
    text = chunk["text"] if full else textwrap.shorten(chunk["text"], 700, placeholder=" …")
    body = "\n".join(
        textwrap.fill(line, SNIPPET_WIDTH, initial_indent="   ", subsequent_indent="   ")
        for line in text.split("\n")
    )
    header = f"[{index}] {citation(chunk)}"
    meta = f"    id={chunk['id']}  score={hit.score:.2f}"
    if chunk.get("section_inherited"):
        meta += "  (раздел определён по соседней странице)"
    if chunk.get("drive_url"):
        meta += f"\n    {chunk['drive_url']}"
    return f"{header}\n{meta}\n{body}"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="RAG-поиск по руководствам Renault Sandero II",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("query", nargs="*", help="запрос на русском или английском")
    parser.add_argument("-k", "--top", type=int, default=5, help="сколько фрагментов вернуть")
    parser.add_argument(
        "--doc", help="ограничить документом (ru_owner, en_owner, service_manual, wiring)"
    )
    parser.add_argument("--lang", choices=["ru", "en"], help="ограничить языком")
    parser.add_argument("--chapter", type=int, help="ограничить главой руководства (1-6)")
    parser.add_argument(
        "--kind",
        choices=["manual", "service", "image"],
        help="тип материала: руководство, ремонтная процедура, иллюстрация",
    )
    parser.add_argument(
        "--per-doc", type=int, default=None, help="не более N фрагментов из одного документа"
    )
    parser.add_argument("--full", action="store_true", help="показать фрагменты целиком")
    parser.add_argument("--json", action="store_true", help="машиночитаемый вывод")
    parser.add_argument("--list-sources", action="store_true", help="состав корпуса и что не вошло")
    args = parser.parse_args()

    if args.list_sources:
        sources = load_sources()
        chunks = load_chunks()
        counts: dict[str, int] = {}
        for chunk in chunks:
            counts[chunk["doc"]] = counts.get(chunk["doc"], 0) + 1
        print(f"Автомобиль: {sources.get('vehicle', '—')}\n")
        for document in sources["documents"]:
            mark = "✓" if document.get("indexed") else "✗"
            print(f"{mark} {document['doc']:16} {document['title']}")
            if document.get("indexed"):
                print(f"    фрагментов в индексе: {counts.get(document['doc'], 0)}")
            else:
                print(f"    не в индексе: {document.get('reason', '—')}")
        return 0

    if not args.query:
        parser.error("нужен текст запроса (или --list-sources)")

    query = " ".join(args.query)
    chunks = load_chunks()
    index = Bm25Index(chunks, load_synonyms())

    where: dict[str, object] = {}
    if args.doc:
        where["doc"] = args.doc
    if args.lang:
        where["lang"] = args.lang
    if args.chapter is not None:
        where["chapter"] = args.chapter
    if args.kind:
        where["kind"] = args.kind

    hits = index.search(query, top_k=args.top, where=where or None, per_doc=args.per_doc)

    if args.json:
        print(
            json.dumps(
                [
                    {"score": round(h.score, 4), "citation": citation(h.chunk), **h.chunk}
                    for h in hits
                ],
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0

    if not hits:
        print(f"Ничего не найдено: {query!r}")
        print("Попробуйте другие слова или снимите фильтры (--doc/--lang/--chapter).")
        return 1

    print(f"Запрос: {query}   найдено фрагментов: {len(hits)}\n")
    for position, hit in enumerate(hits, start=1):
        print(render(hit, position, full=args.full))
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
