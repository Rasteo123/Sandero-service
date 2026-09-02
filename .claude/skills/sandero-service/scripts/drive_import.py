#!/usr/bin/env python3
"""Восстановление исходников (PDF, JPG) из выгрузок коннектора Google Drive.

Зачем: контейнеры Claude Code обычно не имеют прямого доступа к drive.google.com,
а MCP-инструмент `download_file_content` возвращает файл в base64 и складывает
большие ответы на диск. Этот скрипт превращает такие ответы в нормальные файлы.

Как пользоваться:

  1. В сессии с подключённым коннектором Google Drive попросите Claude вызвать
     `download_file_content` для нужных fileId (они перечислены в
     data/sources.json). Ответы осядут в каталоге tool-results.
  2. Запустите:

         python3 scripts/drive_import.py ~/.claude/projects/<проект>/<сессия>/tool-results

     Файлы разложатся в corpus/pdf и corpus/img по MIME-типу.
  3. Пересоберите корпус: python3 scripts/ingest.py

Ограничение коннектора: файлы больше 10 МБ он не отдаёт (см. sources.json,
документы service_manual и logan2).
"""

from __future__ import annotations

import argparse
import base64
import binascii
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))

from ragkit.store import CORPUS_DIR, SOURCES_PATH

SUBDIR_BY_MIME = {
    "application/pdf": "pdf",
    "image/jpeg": "img",
    "image/png": "img",
}


def target_name(file_id: str, title: str, sources: dict) -> str:
    """Имя из sources.json, если файл известен; иначе имя с Диска."""
    for document in sources.get("documents", []):
        if document.get("drive_file_id") == file_id and document.get("file"):
            return Path(document["file"]).name
    return title


def import_dump(path: Path, sources: dict, out_root: Path) -> tuple[Path, int] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8", errors="strict"))
    except (json.JSONDecodeError, UnicodeDecodeError):
        return None
    if not isinstance(payload, dict) or "content" not in payload:
        return None

    mime = payload.get("mimeType", "")
    subdir = SUBDIR_BY_MIME.get(mime)
    if subdir is None:
        return None

    try:
        blob = base64.b64decode(payload["content"], validate=True)
    except (binascii.Error, ValueError):
        return None

    name = target_name(payload.get("id", ""), payload.get("title", path.stem), sources)
    out_path = out_root / subdir / name
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(blob)
    return out_path, len(blob)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "dumps", type=Path, help="каталог с ответами MCP-инструмента (tool-results)"
    )
    parser.add_argument("--out", type=Path, default=CORPUS_DIR, help="куда класть файлы")
    args = parser.parse_args()

    if not args.dumps.is_dir():
        raise SystemExit(f"Не каталог: {args.dumps}")

    sources = json.loads(SOURCES_PATH.read_text(encoding="utf-8"))
    imported = 0
    for dump in sorted(args.dumps.iterdir()):
        if not dump.is_file():
            continue
        result = import_dump(dump, sources, args.out)
        if result:
            out_path, size = result
            print(f"{out_path.relative_to(args.out)}  ({size / 1024:.0f} КБ)")
            imported += 1

    print(f"\nВосстановлено файлов: {imported}")
    if imported:
        print("Дальше: python3 scripts/ingest.py")
    return 0 if imported else 1


if __name__ == "__main__":
    raise SystemExit(main())
