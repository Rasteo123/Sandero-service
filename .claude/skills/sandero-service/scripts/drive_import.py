#!/usr/bin/env python3
"""Восстановление исходников (PDF, JPG) из выгрузок коннектора Google Drive.

Зачем: контейнеры Claude Code обычно не имеют прямого доступа к drive.google.com,
а MCP-инструмент `download_file_content` возвращает файл в base64 и складывает
большие ответы на диск. Этот скрипт превращает такие ответы в нормальные файлы.

Как пользоваться:

  1. В сессии с подключённым коннектором Google Drive попросите Claude вызвать
     `download_file_content` для нужных fileId (они перечислены в
     data/sources.json). Ответы осядут в каталоге tool-results. Zip-архивы
     распаковываются автоматически: файл больше предела коннектора удобно
     разрезать и отдать архивом из нескольких кусков.
  2. Запустите:

         python3 scripts/drive_import.py ~/.claude/projects/<проект>/<сессия>/tool-results

     Файлы разложатся в corpus/pdf и corpus/img по MIME-типу.
  3. Пересоберите корпус: python3 scripts/ingest.py

Ограничение коннектора: практический предел около 6,3 МБ на файл — более
крупные выгрузки обрываются по таймауту (см. sources.json, service_manual
и logan2).
"""

from __future__ import annotations

import argparse
import base64
import binascii
import io
import json
import zipfile
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))

from ragkit.store import CORPUS_DIR, SOURCES_PATH

SUBDIR_BY_MIME = {
    "application/pdf": "pdf",
    "image/jpeg": "img",
    "image/png": "img",
}
SUBDIR_BY_SUFFIX = {".pdf": "pdf", ".jpg": "img", ".jpeg": "img", ".png": "img"}
# Файл крупнее предела коннектора удобно отдавать zip-архивом из нескольких
# кусков — распаковываем их так же, как обычные выгрузки.
ZIP_MIMES = {"application/zip", "application/x-zip-compressed"}


def target_name(file_id: str, title: str, sources: dict) -> str:
    """Имя из sources.json, если файл известен; иначе имя с Диска."""
    for document in sources.get("documents", []):
        if document.get("drive_file_id") == file_id and document.get("file"):
            return Path(document["file"]).name
    return title


def unpack_zip(blob: bytes, out_root: Path) -> list[tuple[Path, int]]:
    """Раскладывает содержимое архива по corpus/pdf и corpus/img."""
    written: list[tuple[Path, int]] = []
    with zipfile.ZipFile(io.BytesIO(blob)) as archive:
        for info in archive.infolist():
            if info.is_dir():
                continue
            name = Path(info.filename).name
            subdir = SUBDIR_BY_SUFFIX.get(Path(name).suffix.lower())
            if subdir is None or name.startswith("."):
                continue
            out_path = out_root / subdir / name
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_bytes(archive.read(info))
            written.append((out_path, info.file_size))
    return written


def import_dump(path: Path, sources: dict, out_root: Path) -> list[tuple[Path, int]]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8", errors="strict"))
    except (json.JSONDecodeError, UnicodeDecodeError):
        return []
    if not isinstance(payload, dict) or "content" not in payload:
        return []

    mime = payload.get("mimeType", "")
    if mime not in ZIP_MIMES and mime not in SUBDIR_BY_MIME:
        return []

    try:
        blob = base64.b64decode(payload["content"], validate=True)
    except (binascii.Error, ValueError):
        return []

    if mime in ZIP_MIMES:
        try:
            return unpack_zip(blob, out_root)
        except zipfile.BadZipFile:
            return []

    name = target_name(payload.get("id", ""), payload.get("title", path.stem), sources)
    out_path = out_root / SUBDIR_BY_MIME[mime] / name
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(blob)
    return [(out_path, len(blob))]


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
        for out_path, size in import_dump(dump, sources, args.out):
            print(f"{out_path.relative_to(args.out)}  ({size / 1024:.0f} КБ)")
            imported += 1

    print(f"\nВосстановлено файлов: {imported}")
    if imported:
        print("Дальше: python3 scripts/ingest.py")
    return 0 if imported else 1


if __name__ == "__main__":
    raise SystemExit(main())
