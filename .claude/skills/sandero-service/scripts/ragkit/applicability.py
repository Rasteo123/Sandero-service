"""Проверка применимости документов к модификации автомобиля.

Корпус собран из разных изданий, и они покрывают разные двигатели.
Сервисный мануал 2012-2016 не застал H4M, который ставили на Sandero II
с 2016 года, — но по словам «замена ремня ГРМ» он отвечает уверенно и
красиво. Для ремонтного поиска это опаснее любой потери полноты: человек
получит заводскую процедуру с моментами затяжки, которая относится к
другому мотору.

Поэтому запрос, называющий двигатель, отсекает документы, этот двигатель
не покрывающие, и поиск говорит об этом вслух.
"""

from __future__ import annotations

import re

# Коды двигателей Renault/Nissan, встречающиеся у Sandero II.
ENGINE_RE = re.compile(r"\b(K4M|K7M|K9K|D4F|H4M|H4Dt|B4D|HR16DE)\b", re.IGNORECASE)
# Слова, по которым видно, что спрашивают про привод ГРМ, и чем именно.
TIMING_RE = re.compile(r"\b(грм|timing)\b", re.IGNORECASE)
BELT_RE = re.compile(r"\b(ремень|ремня|ремнём|ремнем|belt)\b", re.IGNORECASE)
CHAIN_RE = re.compile(r"\b(цепь|цепи|цепью|chain)\b", re.IGNORECASE)
DRIVE_RU = {"chain": "цепной", "belt": "ременной"}


def engines_in_text(text: str) -> list[str]:
    """Коды двигателей, названные в тексте, в каноническом написании."""
    found = []
    for match in ENGINE_RE.findall(text):
        code = match.upper()
        code = "H4M" if code == "HR16DE" else code
        canonical = {"H4DT": "H4Dt"}.get(code, code)
        if canonical not in found:
            found.append(canonical)
    return found


def check(query: str, vehicle: dict, *, profile_engine: str | None = None) -> dict:
    """Решает, какие документы применимы к запросу, и объясняет почему.

    Возвращает {"engine", "allowed", "blocked", "warnings"}. Пустой allowed
    означает, что применимых документов нет, — это честный ответ «нет данных»,
    а не повод показать процедуру от другого мотора.
    """
    engines = vehicle.get("engines", {})
    coverage = vehicle.get("document_engines", {})
    named = engines_in_text(query)
    engine = named[0] if named else profile_engine
    result = {"engine": engine, "allowed": None, "blocked": [], "warnings": []}
    if not engine:
        return result

    if engine not in engines:
        result["warnings"].append(f"Двигатель {engine} неизвестен справочнику применимости.")
        return result

    allowed, blocked = [], []
    for doc, info in coverage.items():
        (allowed if engine in info.get("engines", []) else blocked).append(doc)
    result["allowed"] = allowed
    result["blocked"] = blocked

    for doc in blocked:
        info = coverage[doc]
        result["warnings"].append(
            f"{doc}: не покрывает {engine} ({', '.join(info.get('engines', [])) or 'нет данных'}) — исключён. "
            f"Основание: {info.get('evidence', 'не указано')}."
        )

    not_covered = vehicle.get("not_covered", {}).get(engine)
    repair_docs = [d for d in allowed if not coverage[d].get("note")]
    if not repair_docs and not_covered:
        result["warnings"].append(f"ВНИМАНИЕ. {not_covered}")

    drive = engines[engine].get("timing_drive")
    if drive:
        wrong = CHAIN_RE if drive == "belt" else BELT_RE
        if wrong.search(query):
            result["warnings"].append(
                f"ВНИМАНИЕ. У {engine} привод ГРМ {DRIVE_RU[drive]} — процедуры по "
                f"{'цепи' if drive == 'belt' else 'ремню'} ГРМ к нему неприменимы."
            )
        elif TIMING_RE.search(query) or BELT_RE.search(query) or CHAIN_RE.search(query):
            result["warnings"].append(f"Справка: у {engine} привод ГРМ {DRIVE_RU[drive]}.")
    return result


def restrict(where: dict | None, result: dict) -> dict | None:
    """Добавляет к фильтру поиска ограничение по применимым документам."""
    allowed = result.get("allowed")
    if allowed is None:
        return where
    where = dict(where or {})
    requested = where.get("doc")
    where["doc"] = [doc for doc in allowed if requested in (None, doc)]
    return where
