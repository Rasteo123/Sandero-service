"""Поиск BM25 с расширением запроса по синонимам и нечётким совпадениям."""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from dataclasses import dataclass, field

from .textnorm import tokenize, trigrams

K1 = 1.5
B = 0.75
# Порог схожести триграмм для «спасения» неизвестного слова запроса.
FUZZY_THRESHOLD = 0.62
FUZZY_MAX_MATCHES = 3
# Словарь синонимов выверен вручную, поэтому его связи почти равноправны словам
# запроса: сервисный мануал англоязычный, и русский вопрос доходит до него
# только через кросс-языковые пары.
SYNONYM_WEIGHT = 0.95
# Догадки по триграммам менее надёжны, поэтому весят меньше.
FUZZY_WEIGHT = 0.55
# Надбавка за фразу: слова запроса стоят в тексте рядом и в том же порядке.
PHRASE_BONUS = 0.9
# Надбавка за понятие, попавшее в название единицы: для процедурного документа
# заголовок «CLUTCH: REMOVAL - REFITTING» — сильнейший признак релевантности.
TITLE_BONUS = 1.6
# Спрашивают число — предпочитаем фрагмент, где это число есть. Пока такой
# признак один: момент затяжки, который при сборке помечается has_torque.
ANSWER_BONUS = 0.8
TORQUE_TERMS = frozenset(tokenize("момент затяжки torque tightening n.m"))
# Сколько кандидатов BM25 переоценивать фразовой близостью.
RERANK_DEPTH = 60
# Насколько жёстко требовать, чтобы документ закрывал все понятия запроса.
COVERAGE_POWER = 1.5
# Служебные слова: связав через них группы, мы связали бы весь словарь.
STOPWORD_STEMS = frozenset(
    tokenize("и в на по с от для к или а the a of and to in for with on at")
)


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
        self.title_tokens: list[frozenset[str]] = []
        self._trigram_buckets: dict[str, set[str]] = defaultdict(set)

        for idx, chunk in enumerate(chunks):
            tokens = tokenize(self._indexable_text(chunk))
            self.doc_len.append(len(tokens))
            self.doc_tokens.append(tokens)
            self.title_tokens.append(frozenset(tokenize(chunk.get("section") or "")))
            for term, tf in Counter(tokens).items():
                self.postings[term].append((idx, tf))

        self.n_docs = len(chunks)
        self.avgdl = (sum(self.doc_len) / self.n_docs) if self.n_docs else 0.0
        for term in self.postings:
            for tri in trigrams(term):
                self._trigram_buckets[tri].add(term)

        self.synonyms: list[list[list[str]]] = self._compile_synonyms(synonyms or {})

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

    @staticmethod
    def _compile_synonyms(raw: dict[str, list[str]]) -> list[list[list[str]]]:
        """Словарь -> группы фраз, где каждая фраза это список основ.

        Расширение срабатывает по фразе целиком, а не по отдельным словам.
        Иначе группы слипаются через общее слово: «замена масла» выстреливала
        бы на запросе «замена сцепления» и тащила в выдачу масляные страницы.
        """
        groups: list[list[list[str]]] = []
        for key, values in raw.items():
            if key.startswith("_"):
                continue
            phrases = []
            for phrase in [key, *values]:
                stems = [t for t in dict.fromkeys(tokenize(phrase)) if t not in STOPWORD_STEMS]
                if not stems:
                    continue
                if stems not in phrases:
                    phrases.append(stems)
                # Длинное название обычно сокращают: «головка блока» вместо
                # «головка блока цилиндров». Первые два слова тоже открывают группу.
                if len(stems) > 2 and stems[:2] not in phrases:
                    phrases.append(stems[:2])
            if len(phrases) > 1:
                groups.append(phrases)
        return groups

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

    def expand(self, query: str) -> list[dict]:
        """Запрос -> список понятий; понятие это слово запроса со своими синонимами.

        Понятия нужны, чтобы документ не выигрывал повторами одного слова:
        «замена сцепления» это два понятия, и страница про замену ламп
        закрывает только одно из них.

        Многословный синоним хранится отдельно от одиночных и засчитывается
        только целиком: иначе «удаление воздуха» отдало бы в понятие «прокачка»
        слово «воздух», и страница про кондиционер считалась бы прокачкой.
        """
        tokens = [t for t in dict.fromkeys(tokenize(query)) if t not in STOPWORD_STEMS]
        if not tokens:
            tokens = list(dict.fromkeys(tokenize(query)))
        if not tokens:
            return []

        position = {token: index for index, token in enumerate(tokens)}
        parent = list(range(len(tokens)))

        def root(index: int) -> int:
            while parent[index] != index:
                parent[index] = parent[parent[index]]
                index = parent[index]
            return index

        def merge(left: int, right: int) -> None:
            left, right = root(left), root(right)
            if left != right:
                parent[right] = left

        singles: dict[int, set[str]] = {index: set() for index in range(len(tokens))}
        phrases: dict[int, set[tuple[str, ...]]] = {index: set() for index in range(len(tokens))}
        for group in self.synonyms:
            matched = [phrase for phrase in group if set(phrase) <= position.keys()]
            if not matched:
                continue
            # Многословная фраза («моторное масло») — одно понятие, а не два.
            anchor = position[matched[0][0]]
            for phrase in matched:
                for term in phrase:
                    merge(anchor, position[term])
            for phrase in group:
                if len(phrase) == 1:
                    singles[root(anchor)].add(phrase[0])
                else:
                    phrases[root(anchor)].add(tuple(phrase))

        concepts: dict[int, dict] = {}
        for token, index in position.items():
            concept = concepts.setdefault(root(index), {"terms": {}, "phrases": []})
            concept["terms"][token] = 1.0
        for index, terms in singles.items():
            if not terms:
                continue
            concept = concepts.setdefault(root(index), {"terms": {}, "phrases": []})
            for term in terms:
                concept["terms"].setdefault(term, SYNONYM_WEIGHT)
        for index, groups in phrases.items():
            if not groups:
                continue
            concept = concepts.setdefault(root(index), {"terms": {}, "phrases": []})
            concept["phrases"].extend((phrase, SYNONYM_WEIGHT) for phrase in groups)

        for token, index in position.items():
            if token in self.postings:
                continue
            concept = concepts[root(index)]
            for near in self.fuzzy_matches(token):
                concept["terms"].setdefault(near, FUZZY_WEIGHT)

        return list(concepts.values())

    def _title_hit(self, idx: int, concept: dict) -> bool:
        """Понятие попало в название единицы.

        Многословная фраза засчитывается только целиком — иначе название
        «HYDRAULIC BRAKE UNIT REMOVAL» получало бы надбавку за понятие «ABS»
        из-за одного слова «brake» в паре «anti-lock braking».
        """
        title = self.title_tokens[idx]
        if title & concept["terms"].keys():
            return True
        return any(set(phrase) <= title for phrase, _ in concept["phrases"])

    def _bm25(self, term: str, idx: int, tf: int) -> float:
        norm = 1 - B + B * (self.doc_len[idx] / self.avgdl if self.avgdl else 1)
        return self._idf(term) * (tf * (K1 + 1)) / (tf + K1 * norm)

    def _concept_scores(self, concept: dict) -> tuple[dict[int, float], dict[int, str]]:
        """Лучшее совпадение понятия в каждом документе.

        Синонимы внутри понятия конкурируют, а не складываются: три слова об
        одном и том же не должны весить втрое больше одного точного.
        """
        best: dict[int, float] = {}
        best_term: dict[int, str] = {}

        def offer(idx: int, value: float, label: str) -> None:
            if value > best.get(idx, 0.0):
                best[idx] = value
                best_term[idx] = label

        for term, weight in concept["terms"].items():
            for idx, tf in self.postings.get(term, ()):
                offer(idx, weight * self._bm25(term, idx, tf), term)

        for phrase, weight in concept["phrases"]:
            postings = [dict(self.postings.get(term, ())) for term in phrase]
            if not all(postings):
                continue
            common = set(postings[0])
            for other in postings[1:]:
                common &= other.keys()
            for idx in common:
                # Требование «все слова фразы на месте» — это защита от ложных
                # срабатываний; сама оценка складывается из всех её слов, иначе
                # кросс-языковая пара («timing belt») заведомо проигрывает
                # одному русскому слову с высоким весом.
                value = weight * sum(
                    self._bm25(term, idx, postings[n][idx]) for n, term in enumerate(phrase)
                )
                offer(idx, value, " ".join(phrase))
        return best, best_term

    def search(
        self,
        query: str,
        top_k: int = 5,
        *,
        where: dict[str, object] | None = None,
        per_doc: int | None = None,
        engine: str | None = None,
    ) -> list[Hit]:
        concepts = self.expand(query)
        if not concepts:
            return []
        query_tokens = tokenize(query)
        scores: dict[int, float] = defaultdict(float)
        covered: dict[int, int] = defaultdict(int)
        matched: dict[int, set[str]] = defaultdict(set)

        for concept in concepts:
            best, best_term = self._concept_scores(concept)
            for idx, value in best.items():
                if self._title_hit(idx, concept):
                    value *= 1 + TITLE_BONUS
                scores[idx] += value
                covered[idx] += 1
                matched[idx].add(best_term[idx])

        # Покрытие понятий: документ, ответивший на весь запрос, важнее того,
        # кто много раз повторил одно слово из него.
        total = len(concepts)
        for idx in list(scores):
            scores[idx] *= (covered[idx] / total) ** COVERAGE_POWER

        wants_torque = bool(TORQUE_TERMS & set(query_tokens))
        ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
        for idx, score in ranked[:RERANK_DEPTH]:
            score += self._phrase_bonus(idx, query_tokens)
            if wants_torque and self.chunks[idx].get("has_torque"):
                score *= 1 + ANSWER_BONUS
            scores[idx] = score

        hits = [
            Hit(chunk=self.chunks[idx], score=score, matched=sorted(matched[idx]))
            for idx, score in scores.items()
            if self._passes(self.chunks[idx], where) and self._fits_engine(self.chunks[idx], engine)
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
    def _fits_engine(chunk: dict, engine: str | None) -> bool:
        """Фрагмент, помеченный другим двигателем, не годится.

        Метка есть не у всех: схема кузова к мотору не привязана. Отсекаются
        только те фрагменты, про которые известно, что они про чужой двигатель.
        """
        engines = chunk.get("engines")
        return not (engine and engines) or engine in engines

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
