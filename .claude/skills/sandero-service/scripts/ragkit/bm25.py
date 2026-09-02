"""Поиск BM25 с расширением запроса по синонимам и нечётким совпадениям."""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from dataclasses import dataclass, field

from .textnorm import CYRILLIC_RE, LATIN_RE, tokenize, trigrams

K1 = 1.5
B = 0.75
# Порог схожести триграмм для «спасения» неизвестного слова запроса.
FUZZY_THRESHOLD = 0.62
FUZZY_MAX_MATCHES = 3
# Вес расширенных терминов относительно исходных слов запроса.
EXPANSION_WEIGHT = 0.55
# Надбавка за фразу: слова запроса стоят в тексте рядом и в том же порядке.
PHRASE_BONUS = 0.9
# Надбавка документу на языке запроса — русский вопрос предпочитает русский оригинал.
LANG_BONUS = 0.3
# Сколько кандидатов BM25 переоценивать фразовой близостью.
RERANK_DEPTH = 60


def detect_lang(text: str) -> str | None:
    """Язык запроса по преобладающему алфавиту: ru, en или None при ничьей."""
    cyrillic = len(CYRILLIC_RE.findall(text))
    latin = len(LATIN_RE.findall(text))
    if cyrillic > latin:
        return "ru"
    if latin > cyrillic:
        return "en"
    return None


@dataclass
class Hit:
    chunk: dict
    score: float
    matched: list[str] = field(default_factory=list)


class Bm25Index:
    """Инвертированный индекс над списком чанков.

    Корпус небольшой (тысячи чанков), поэтому индекс строится в памяти при
    каждом запуске: нет файлов кэша — нет рассинхронизации с корпусом.
    """

    def __init__(self, chunks: list[dict], synonyms: dict[str, list[str]] | None = None):
        self.chunks = chunks
        self.postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
        self.doc_len: list[int] = []
        self.doc_tokens: list[list[str]] = []
        self._trigram_buckets: dict[str, set[str]] = defaultdict(set)

        for idx, chunk in enumerate(chunks):
            tokens = tokenize(self._indexable_text(chunk))
            self.doc_len.append(len(tokens))
            self.doc_tokens.append(tokens)
            for term, tf in Counter(tokens).items():
                self.postings[term].append((idx, tf))

        self.n_docs = len(chunks)
        self.avgdl = (sum(self.doc_len) / self.n_docs) if self.n_docs else 0.0
        for term in self.postings:
            for tri in trigrams(term):
                self._trigram_buckets[tri].add(term)

        self.synonyms = self._compile_synonyms(synonyms or {})

    @staticmethod
    def _indexable_text(chunk: dict) -> str:
        """Заголовки весят больше: раздел и название документа идут в индекс дважды.

        Поле labels индексируется, но в выдаче не показывается — это ключевые
        слова к иллюстрациям, а не текст документа.
        """
        parts = [
            chunk.get("section", ""),
            chunk.get("section", ""),
            chunk.get("chapter_title", ""),
            chunk.get("doc_title", ""),
            chunk.get("labels", ""),
            chunk.get("text", ""),
        ]
        return "\n".join(p for p in parts if p)

    def _compile_synonyms(self, raw: dict[str, list[str]]) -> dict[str, list[str]]:
        """Слова синонимов -> основы; ключи тоже раскладываются на основы."""
        compiled: dict[str, list[str]] = defaultdict(list)
        for key, values in raw.items():
            key_tokens = tokenize(key)
            value_tokens: list[str] = []
            for value in values:
                value_tokens.extend(tokenize(value))
            group = list(dict.fromkeys(key_tokens + value_tokens))
            # Двусторонняя связь: любой член группы тянет за собой остальные.
            for member in group:
                for other in group:
                    if other != member and other not in compiled[member]:
                        compiled[member].append(other)
        return dict(compiled)

    def _idf(self, term: str) -> float:
        df = len(self.postings.get(term, ()))
        if df == 0:
            return 0.0
        return math.log(1 + (self.n_docs - df + 0.5) / (df + 0.5))

    def fuzzy_matches(self, term: str) -> list[str]:
        """Ищет похожие термины индекса по триграммам (опечатки, редкие формы)."""
        target = trigrams(term)
        candidates: Counter[str] = Counter()
        for tri in target:
            for candidate in self._trigram_buckets.get(tri, ()):
                candidates[candidate] += 1
        scored = []
        for candidate, shared in candidates.items():
            union = len(target | trigrams(candidate))
            jaccard = shared / union if union else 0.0
            if jaccard >= FUZZY_THRESHOLD:
                scored.append((jaccard, candidate))
        scored.sort(reverse=True)
        return [c for _, c in scored[:FUZZY_MAX_MATCHES]]

    def expand(self, query: str) -> dict[str, float]:
        """Запрос -> {основа: вес}. Синонимы и нечёткие формы идут с меньшим весом."""
        weights: dict[str, float] = {}
        for term in tokenize(query):
            weights[term] = max(weights.get(term, 0.0), 1.0)
        for term in list(weights):
            for synonym in self.synonyms.get(term, ()):
                if synonym not in weights:
                    weights[synonym] = EXPANSION_WEIGHT
            if term not in self.postings:
                for near in self.fuzzy_matches(term):
                    if near not in weights:
                        weights[near] = EXPANSION_WEIGHT
        return weights

    def search(
        self,
        query: str,
        top_k: int = 5,
        *,
        where: dict[str, object] | None = None,
        per_doc: int | None = None,
    ) -> list[Hit]:
        weights = self.expand(query)
        query_tokens = tokenize(query)
        scores: dict[int, float] = defaultdict(float)
        matched: dict[int, set[str]] = defaultdict(set)

        for term, weight in weights.items():
            postings = self.postings.get(term)
            if not postings:
                continue
            idf = self._idf(term)
            for idx, tf in postings:
                norm = 1 - B + B * (self.doc_len[idx] / self.avgdl if self.avgdl else 1)
                scores[idx] += weight * idf * (tf * (K1 + 1)) / (tf + K1 * norm)
                matched[idx].add(term)

        ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
        query_lang = detect_lang(query)
        for idx, score in ranked[:RERANK_DEPTH]:
            score += self._phrase_bonus(idx, query_tokens)
            if query_lang and self.chunks[idx].get("lang") == query_lang:
                score *= 1 + LANG_BONUS
            scores[idx] = score

        hits = [
            Hit(chunk=self.chunks[idx], score=score, matched=sorted(matched[idx]))
            for idx, score in scores.items()
            if self._passes(self.chunks[idx], where)
        ]
        hits.sort(key=lambda h: h.score, reverse=True)
        if per_doc:
            hits = self._limit_per_doc(hits, per_doc)
        return hits[:top_k]

    def _phrase_bonus(self, idx: int, query_tokens: list[str]) -> float:
        """Награда за то, что слова запроса стоят в тексте рядом и в том же порядке."""
        if len(query_tokens) < 2:
            return 0.0
        tokens = self.doc_tokens[idx]
        adjacent = set(zip(tokens, tokens[1:]))
        bonus = 0.0
        for first, second in zip(query_tokens, query_tokens[1:]):
            if (first, second) in adjacent:
                bonus += PHRASE_BONUS * (self._idf(first) + self._idf(second)) / 2
        return bonus

    @staticmethod
    def _passes(chunk: dict, where: dict[str, object] | None) -> bool:
        if not where:
            return True
        for key, expected in where.items():
            value = chunk.get(key)
            if isinstance(expected, (list, tuple, set)):
                if value not in expected:
                    return False
            elif str(value) != str(expected):
                return False
        return True

    @staticmethod
    def _limit_per_doc(hits: list[Hit], per_doc: int) -> list[Hit]:
        """Не даёт одному документу занять всю выдачу."""
        seen: Counter[str] = Counter()
        kept = []
        for hit in hits:
            doc = str(hit.chunk.get("doc", ""))
            if seen[doc] >= per_doc:
                continue
            seen[doc] += 1
            kept.append(hit)
        return kept
