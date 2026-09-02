#!/usr/bin/env python3
"""Отрисовка страницы документа в картинку — чтобы показать иллюстрацию.

    python3 scripts/page_image.py service_manual 356 357
    python3 scripts/page_image.py ru_owner --preset maintenance --dpi 120
    python3 scripts/page_image.py ru_owner --label 4.7
    python3 scripts/page_image.py en_owner 178 --dpi 150

Зачем: в корпусе лежит только текст, а процедуры сервисного мануала полны
ссылок на выноски — «отверните болты (3)», «отсоедините колодку (12)».
Без рисунка такая процедура наполовину бесполезна. Отрисовывать весь
мануал заранее нельзя: 3508 страниц это сотни мегабайт, поэтому страница
рендерится тогда, когда она понадобилась.

Нужны исходные PDF в corpus/pdf (их восстанавливает scripts/drive_import.py)
и pymupdf. Уже отрисованные страницы кэшируются и второй раз не считаются.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from ragkit.store import CACHE_DIR, CORPUS_DIR, PAGES_DIR, SOURCES_PATH, load_chunks


def locate(doc: str, page: int, sources: dict) -> tuple[Path, int]:
    """Сквозной номер страницы -> файл на диске и номер страницы внутри него.

    Сервисный мануал разрезан на части, у каждой своё смещение; у остальных
    документов файл один и смещения нет.
    """
    document = next((d for d in sources["documents"] if d["doc"] == doc), None)
    if document is None:
        raise SystemExit(f"Неизвестный документ: {doc}")

    if document.get("format") == "service_manual":
        parts = sorted(document["parts"], key=lambda p: p["page_offset"])
        for part in reversed(parts):
            if page > part["page_offset"]:
                return CORPUS_DIR / part["file"], page - part["page_offset"]
        raise SystemExit(f"Страница {page} вне нумерации {doc}")

    if not document.get("file"):
        raise SystemExit(f"У документа {doc} нет локального файла")
    return CORPUS_DIR / document["file"], page


def page_for_label(doc: str, label: str) -> int:
    """Метка бумажной страницы («4.7») -> сквозной номер страницы PDF.

    У чанка хранится метка начала единицы и её конца, а не каждой страницы,
    поэтому нужную страницу вычисляем по смещению внутри диапазона.
    """

    def parts(value: str) -> tuple[int, int] | None:
        try:
            chapter, page = value.split(".", 1)
            return int(chapter), int(page)
        except (ValueError, AttributeError):
            return None

    wanted = parts(label)
    if wanted is None:
        raise SystemExit(f"Непонятная метка страницы: {label}")

    for chunk in load_chunks():
        if chunk["doc"] != doc:
            continue
        start = parts(chunk.get("page_label") or "")
        if start is None or start[0] != wanted[0]:
            continue
        end = parts(chunk.get("page_label_end") or "") or start
        if start[1] <= wanted[1] <= end[1]:
            return chunk["pdf_page"] + (wanted[1] - start[1])
    raise SystemExit(f"В корпусе нет страницы {label} документа {doc}")


def render(
    doc: str, page: int, sources: dict, *, dpi: int, out_dir: Path, fmt: str = "png"
) -> Path:
    suffix = "jpg" if fmt == "jpg" else "png"
    name = f"p{page:04d}.{suffix}" if out_dir == PAGES_DIR else f"p{page:04d}-{dpi}dpi.{suffix}"
    out_path = out_dir / doc / name
    if out_path.exists():
        return out_path
    try:
        import pymupdf
    except ImportError:  # pragma: no cover - зависит от окружения
        raise SystemExit("Нужен pymupdf: pip install -r requirements.txt")

    path, local = locate(doc, page, sources)
    if not path.exists():
        raise SystemExit(
            f"Нет файла {path}. Восстановите исходники: scripts/drive_import.py\n"
            "Для поиска они не нужны, но без них страницу не отрисовать."
        )
    with pymupdf.open(str(path)) as document:
        if not 1 <= local <= document.page_count:
            raise SystemExit(f"В {path.name} нет страницы {local}")
        pixmap = document[local - 1].get_pixmap(dpi=dpi)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(
        pixmap.tobytes("jpg", jpg_quality=80) if fmt == "jpg" else pixmap.tobytes("png")
    )
    return out_path


def pages_with_figures(doc: str, sources: dict, chapters: tuple[int, ...] | None = None) -> list[int]:
    """Страницы документа, на которых есть рисунки; можно сузить до глав.

    Отрисовывать всё подряд бессмысленно: страница без единого рисунка
    ничего не добавляет к тексту, который уже в корпусе.
    """
    import pymupdf

    document = next(d for d in sources["documents"] if d["doc"] == doc)
    first, last = 1, None
    if chapters:
        chunks = [c for c in load_chunks() if c["doc"] == doc and c.get("chapter") in chapters]
        first = min(c["pdf_page"] for c in chunks)
        last = max(c.get("pdf_page_end") or c["pdf_page"] for c in chunks)
    with pymupdf.open(str(CORPUS_DIR / document["file"])) as pdf:
        last = last or pdf.page_count
        return [p for p in range(first, last + 1) if pdf[p - 1].get_images()]


# Наборы страниц, которые имеет смысл отрисовать заранее и положить в скилл.
# Сервисного мануала здесь нет намеренно: 3170 страниц с рисунками — это
# больше 200 МБ, их рендерим по требованию.
PRESETS = {
    "maintenance": [("ru_owner", (4, 5))],
    "owner": [("ru_owner", None), ("en_owner", None)],
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("doc", help="ru_owner, en_owner, service_manual")
    parser.add_argument("pages", nargs="*", type=int, help="сквозные номера страниц PDF")
    parser.add_argument("--label", help="метка бумажной страницы, например 4.7")
    parser.add_argument("--dpi", type=int, default=140, help="разрешение (по умолчанию 140)")
    parser.add_argument(
        "--preset",
        choices=sorted(PRESETS),
        help="отрисовать набор, который идёт вместе со скиллом",
    )
    parser.add_argument("--out", type=Path, default=CACHE_DIR, help="куда складывать")
    args = parser.parse_args()

    sources = json.loads(SOURCES_PATH.read_text(encoding="utf-8"))
    pages = list(args.pages)
    if args.label:
        pages.append(page_for_label(args.doc, args.label))
    fmt = "png"
    out_dir = args.out
    if args.preset:
        out_dir, fmt = PAGES_DIR, "jpg"
        for doc, chapters in PRESETS[args.preset]:
            for page in pages_with_figures(doc, sources, chapters):
                print(render(doc, page, sources, dpi=args.dpi, out_dir=out_dir, fmt=fmt))
        return 0

    if not pages:
        parser.error("укажите номера страниц, --label или --preset")

    for page in pages:
        print(render(args.doc, page, sources, dpi=args.dpi, out_dir=out_dir, fmt=fmt))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
