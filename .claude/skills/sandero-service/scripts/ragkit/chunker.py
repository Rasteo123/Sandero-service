"""Разбор страниц руководств Renault и нарезка на чанки для поиска."""

from __future__ import annotations

import re

# Служебный мусор типографики, попадающий в текстовый слой PDF.
NOISE_LINE = re.compile(
    r"^(?:"
    r"jaune\s+noir(?:\s+noir\s+texte)?|cyan\s+magenta.*|noir\s+texte"
    r"|(?:rus|eng|fra)_ud\d+.*|(?:rus|eng|fra)_nu[_\s].*"
    r"|nu\d+-\d+\s*\|.*|\d+\s+de\s+couv.*"
    r")$",
    re.IGNORECASE,
)
# Выноски к иллюстрациям: одиночная цифра или ряд небольших чисел («1 2», «2 1 2»).
# Числа из таблиц характеристик (75, 520, 0,827) при этом сохраняются.
CALLOUT_LINE = re.compile(r"^(?:\d|(?:\d{1,2}[\s,]+){1,}\d{1,2})$")
# Строка оглавления: «Название . . . . . . 4.12»
TOC_LINE = re.compile(r"\.\s?\.\s?\.\s?\.")
# Метка страницы бумажного руководства: 4.12, 0.1, 6.7
PAGE_LABEL = re.compile(r"^(\d{1,2})\.(\d{1,3})$")
PAGE_LABEL_INLINE = re.compile(r"\b(\d{1,2})\.(\d{1,3})\b")
# Французский идентификатор раздела из исходной вёрстки — полезная подсказка темы.
SECTION_SRC = re.compile(r"^[A-Za-zÀ-ÿ][^\n]{4,90}\((?:X52|X52 Ph2|X90)[^)]*\)$")
# Переносы: «поврежде -\nния», «main-\ntenance»
HYPHEN_BREAK = re.compile(r"([a-zа-яё])\s*-\s*\n\s*([a-zа-яё])")
# Код вёрстки, прилипший к содержательной строке
INLINE_CODE = re.compile(r"(?:RUS|ENG|FRA)_(?:NU|UD)[_\w.\-]*", re.IGNORECASE)
# Уточнение в скобках не должно мешать распознать заголовок: «(в метрах)», «(2/4)»
PARENTHETICAL = re.compile(r"\([^)]*\)")

CHAPTERS_RU = {
    0: "Введение",
    1: "Знакомство с автомобилем",
    2: "Вождение автомобиля",
    3: "Комфорт",
    4: "Техническое обслуживание",
    5: "Практические советы",
    6: "Технические характеристики",
    7: "Алфавитный указатель",
}
CHAPTERS_EN = {
    0: "Introduction",
    1: "Getting to know your vehicle",
    2: "Driving",
    3: "Your comfort",
    4: "Maintenance",
    5: "Practical advice",
    6: "Technical specifications",
    7: "Alphabetical index",
}


def _looks_like_a_title(line: str) -> bool:
    """Отсеивает обрывки предупреждений и буквенные выноски («A B C D», «ЗАПРЕЩЕНА.»).

    Заголовок — это либо два и более значащих слова, либо одно длинное слово
    («ПРЕДОХРАНИТЕЛИ»), но не оборванная фраза с точкой или восклицанием.
    """
    words = [w for w in re.split(r"[\s,]+", line) if len(w) >= 3 and any(c.isalpha() for c in w)]
    if len(words) >= 2:
        return True
    return len(words) == 1 and len(words[0]) >= 8 and not line.rstrip().endswith((".", "!"))


def _is_heading(line: str) -> bool:
    """Заголовок раздела: строка целиком капсом либо «КАПС: уточнение»."""
    letters = [c for c in line if c.isalpha()]
    if len(letters) < 6 or not _looks_like_a_title(line):
        return False
    core = PARENTHETICAL.sub("", line)
    core_letters = [c for c in core if c.isalpha()]
    if core_letters and sum(1 for c in core_letters if c.isupper()) / len(core_letters) >= 0.75:
        return True
    head, sep, _ = line.partition(":")
    if not sep:
        return False
    head_letters = [c for c in head if c.isalpha()]
    return len(head_letters) >= 4 and all(c.isupper() for c in head_letters)


def parse_page(text: str) -> dict:
    """Чистит текстовый слой страницы и достаёт метаданные."""
    text = HYPHEN_BREAK.sub(r"\1\2", text)
    raw_lines = [ln.strip() for ln in text.split("\n")]

    page_label = None
    section = None
    section_src = None
    toc_lines = 0
    kept: list[str] = []
    seen: set[str] = set()

    for line in raw_lines:
        line = INLINE_CODE.sub("", line).strip()
        if not line:
            continue
        match = PAGE_LABEL.match(line)
        if match:
            page_label = page_label or line
            continue
        if TOC_LINE.search(line):
            toc_lines += 1
            continue
        if NOISE_LINE.match(line) or CALLOUT_LINE.match(line):
            continue
        if SECTION_SRC.match(line):
            section_src = section_src or line
            continue
        if _is_heading(line) and section is None:
            section = line
        key = line.lower()
        if key in seen:
            continue
        seen.add(key)
        kept.append(line)

    body = "\n".join(kept)
    if page_label is None:
        inline = PAGE_LABEL_INLINE.search(text)
        if inline:
            page_label = inline.group(0)

    chapter = int(page_label.split(".")[0]) if page_label else None
    return {
        "page_label": page_label,
        "chapter": chapter,
        "section": section,
        "section_src": section_src,
        "text": body,
        "toc_ratio": toc_lines / max(len(kept) + toc_lines, 1),
    }


def _paragraphs(text: str) -> list[str]:
    """Склеивает строки в абзацы и выкидывает повторы вёрстки."""
    paragraphs: list[str] = []
    buffer: list[str] = []
    for line in text.split("\n"):
        buffer.append(line)
        # Абзац закончился, если строка завершается знаком конца предложения.
        if re.search(r"[.;:!?]$", line) or _is_heading(line):
            paragraphs.append(" ".join(buffer).strip())
            buffer = []
    if buffer:
        paragraphs.append(" ".join(buffer).strip())

    unique: list[str] = []
    seen: set[str] = set()
    for paragraph in paragraphs:
        key = re.sub(r"\s+", " ", paragraph.lower())
        if not key or key in seen:
            continue
        seen.add(key)
        unique.append(re.sub(r"\s+", " ", paragraph))
    return unique


def _split_oversized(paragraph: str, target: int) -> list[str]:
    """Абзац длиннее чанка режется по предложениям, а сверхдлинное — по словам."""
    if len(paragraph) <= target:
        return [paragraph]
    pieces: list[str] = []
    for sentence in re.split(r"(?<=[.!?;])\s+", paragraph):
        if len(sentence) <= target:
            pieces.append(sentence)
            continue
        buffer = ""
        for word in sentence.split(" "):
            if buffer and len(buffer) + len(word) + 1 > target:
                pieces.append(buffer)
                buffer = word
            else:
                buffer = f"{buffer} {word}".strip()
        if buffer:
            pieces.append(buffer)
    return [piece for piece in pieces if piece]


def chunk_page(page: dict, *, target: int = 900, overlap: int = 160) -> list[str]:
    """Режет страницу на чанки ~target символов по границам абзацев."""
    paragraphs = [
        piece
        for paragraph in _paragraphs(page["text"])
        for piece in _split_oversized(paragraph, target)
    ]
    chunks: list[str] = []
    current = ""
    for paragraph in paragraphs:
        if current and len(current) + len(paragraph) + 1 > target:
            chunks.append(current.strip())
            tail = current[-overlap:] if overlap else ""
            # Хвост предыдущего чанка сохраняет контекст на стыке.
            current = (tail.split(" ", 1)[-1] + " " + paragraph).strip() if tail else paragraph
        else:
            current = f"{current} {paragraph}".strip() if current else paragraph
    if current.strip():
        chunks.append(current.strip())
    return [c for c in chunks if len(c) >= 40]
