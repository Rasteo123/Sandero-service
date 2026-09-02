"""Нормализация текста для поиска по русско-английскому корпусу.

Модуль сознательно не тянет внешних зависимостей: скилл должен работать
на любой машине с python3 без `pip install`.
"""

from __future__ import annotations

import re
import unicodedata

RU_VOWELS = "аеиоуыэюя"

CYRILLIC_RE = re.compile(r"[а-яёА-ЯЁ]")
LATIN_RE = re.compile(r"[a-zA-Z]")

# --- Стеммер Портера для русского языка ---------------------------------
_RVRE = re.compile(r"^(.*?[аеиоуыэюя])(.*)$")
_PERFECTIVE_GERUND = re.compile(
    r"((ив|ыв|ившись|ывшись)|((?<=[ая])(в|вши|вшись)))$"
)
_REFLEXIVE = re.compile(r"(с[яь])$")
_ADJECTIVE = re.compile(
    r"(ее|ие|ые|ое|ими|ыми|ей|ий|ый|ой|ем|им|ым|ом|его|ого|ему|ому|их|ых"
    r"|ую|юю|ая|яя|ою|ею)$"
)
_PARTICIPLE = re.compile(r"((ивш|ывш|ующ)|((?<=[ая])(ем|нн|вш|ющ|щ)))$")
_VERB = re.compile(
    r"((ила|ыла|ена|ейте|уйте|ите|или|ыли|ей|уй|ил|ыл|им|ым|ен|ило|ыло|ено"
    r"|ят|ует|уют|ит|ыт|ены|ить|ыть|ишь|ую|ю)"
    r"|((?<=[ая])(ла|на|ете|йте|ли|й|л|ем|н|ло|но|ет|ют|ны|ть|ешь|нно)))$"
)
_NOUN = re.compile(
    r"(а|ев|ов|ие|ье|е|иями|ями|ами|еи|ии|и|ией|ей|ой|ий|й|иям|ям|ием|ем|ам"
    r"|ом|о|у|ах|иях|ях|ы|ь|ию|ью|ю|ия|ья|я)$"
)
_SUPERLATIVE = re.compile(r"(ейше|ейш)$")
_DERIVATIONAL = re.compile(r"[^аеиоуыэюя][аеиоуыэюя]+[^аеиоуыэюя]+[аеиоуыэюя].*(?<=о)сть?$")
_DER = re.compile(r"ость?$")
_I = re.compile(r"и$")
_SOFT = re.compile(r"ь$")
_NN = re.compile(r"нн$")


def stem_ru(word: str) -> str:
    """Стеммер Портера (Snowball) для русского. Возвращает основу слова."""
    m = _RVRE.match(word)
    if not m:
        return word
    pre, rv = m.groups()

    temp, n = _PERFECTIVE_GERUND.subn("", rv)
    if n == 0:
        rv = _REFLEXIVE.sub("", rv)
        temp, n = _ADJECTIVE.subn("", rv)
        if n > 0:
            rv = temp
            rv = _PARTICIPLE.sub("", rv)
        else:
            temp, n = _VERB.subn("", rv)
            rv = temp if n > 0 else _NOUN.sub("", rv)
    else:
        rv = temp

    rv = _I.sub("", rv)
    if _DERIVATIONAL.match(pre + rv):
        rv = _DER.sub("", rv)
    temp, n = _SUPERLATIVE.subn("", rv)
    rv = temp
    temp, n = _NN.subn("н", rv)
    rv = temp if n > 0 else _SOFT.sub("", rv)
    return pre + rv


# --- Лёгкий стеммер для английского -------------------------------------
# (суффикс, замена, минимальная длина основы). Разный минимум не прихоть:
# окончание множественного числа безопасно и на коротком слове (pads -> pad),
# а глагольное -ed на нём ломает смысл (bleed -> ble), и формы расходятся.
_EN_RULES = (
    ("ational", "ate", 4), ("iveness", "ive", 4), ("fulness", "ful", 4),
    ("ousness", "ous", 4), ("ization", "ize", 4), ("ations", "ate", 4),
    ("tional", "tion", 4), ("alism", "al", 4), ("ities", "ity", 4),
    ("ements", "ement", 4), ("ances", "ance", 4), ("ences", "ence", 4),
    ("ingly", "", 4), ("ings", "", 4), ("ness", "", 4), ("ions", "ion", 4),
    ("ing", "", 4), ("ers", "er", 4), ("ies", "y", 3), ("ied", "y", 4),
    ("ses", "se", 3), ("ed", "", 4), ("es", "", 3), ("s", "", 3),
)


def _drop_silent_e(word: str) -> str:
    """Немое «e» снимается всегда: иначе fuse и fuses дают разные основы."""
    return word[:-1] if len(word) > 3 and word.endswith("e") else word


def stem_en(word: str) -> str:
    if len(word) <= 3:
        return word
    for suffix, repl, min_len in _EN_RULES:
        if not word.endswith(suffix):
            continue
        candidate = word[: -len(suffix)] + repl
        if len(candidate) >= min_len:
            return _drop_silent_e(candidate)
    return _drop_silent_e(word)


def stem(token: str) -> str:
    """Стеммит токен, выбирая правила по алфавиту. Числа и коды не трогает."""
    if not token or any(ch.isdigit() for ch in token):
        return token
    if any("а" <= ch <= "я" or ch == "ё" for ch in token):
        return stem_ru(token)
    return stem_en(token)


# --- Токенизация ---------------------------------------------------------
# Токен: буквенно-цифровая последовательность, внутри допустимы `-` и `/`
# (10w-40, стоп-старт, вкл/выкл, k9k-796).
_TOKEN_RE = re.compile(r"[a-zа-яё0-9]+(?:[-/.][a-zа-яё0-9]+)*")
_SPLIT_RE = re.compile(r"[-/]")


def normalize(text: str) -> str:
    """Приводит текст к нижнему регистру, чинит ё и юникодные пробелы/дефисы."""
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("­", "").replace("‑", "-")
    text = re.sub(r"[‐‒–—―]", "-", text)
    text = re.sub(r"[«»„“”‘’]", '"', text)
    return text.lower().replace("ё", "е")


def tokenize(text: str, *, expand_compounds: bool = True) -> list[str]:
    """Текст -> список основ. Составные токены дополнительно разбиваются.

    'стоп-старт' -> ['стоп-старт', 'стоп', 'старт'] — так находится и
    точная форма, и части по отдельности.
    """
    out: list[str] = []
    for raw in _TOKEN_RE.findall(normalize(text)):
        raw = raw.strip(".")
        if not raw:
            continue
        out.append(stem(raw))
        if expand_compounds and _SPLIT_RE.search(raw):
            for part in _SPLIT_RE.split(raw):
                if len(part) > 1:
                    out.append(stem(part))
    return out


def trigrams(token: str) -> set[str]:
    """Символьные триграммы с якорями — основа нечёткого сопоставления."""
    padded = f"  {token} "
    return {padded[i : i + 3] for i in range(len(padded) - 2)}
