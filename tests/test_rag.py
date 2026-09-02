"""Тесты RAG-движка скилла sandero-service.

    python3 -m unittest discover -s tests -v

Только стандартная библиотека: тесты должны идти в том же окружении,
в котором работает сам скилл.
"""

import json
import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SKILL_SCRIPTS = REPO_ROOT / ".claude" / "skills" / "sandero-service" / "scripts"
sys.path.insert(0, str(SKILL_SCRIPTS))

from ragkit.bm25 import Bm25Index, detect_lang
from ragkit.chunker import chunk_page, parse_page
from ragkit.store import citation, load_chunks, load_synonyms
from ragkit.textnorm import stem, tokenize, trigrams


class TestTextNorm(unittest.TestCase):
    def test_russian_forms_share_a_stem(self):
        for group in (
            ("масло", "масла", "маслом", "маслу"),
            ("двигатель", "двигателя", "двигателем"),
            ("шина", "шины", "шинах"),
            ("предохранитель", "предохранители", "предохранителя"),
        ):
            stems = {stem(word) for word in group}
            self.assertEqual(len(stems), 1, f"{group} -> {stems}")

    def test_english_plurals_share_a_stem(self):
        self.assertEqual(stem("fuses"), stem("fuse"))
        self.assertEqual(stem("wipers"), stem("wiper"))

    def test_tokens_with_digits_are_left_intact(self):
        self.assertIn("10w-40", tokenize("масло 10W-40"))
        self.assertIn("k9k", tokenize("двигатель K9K"))

    def test_compound_tokens_are_also_split(self):
        tokens = tokenize("стоп-старт")
        self.assertIn("стоп-старт", tokens)
        self.assertIn("стоп", tokens)
        self.assertIn("старт", tokens)

    def test_yo_is_normalised(self):
        self.assertEqual(tokenize("ёмкость"), tokenize("емкость"))

    def test_trigrams_are_anchored(self):
        self.assertIn("  ф", trigrams("фара"))


class TestChunker(unittest.TestCase):
    PAGE = "\n".join(
        [
            "Jaune Noir Noir texte",
            "RUS_UD57140_3",
            "Niveau huile moteur : appoint / remplissage (X52 Ph2 - Dacia)",
            "1 2",
            "2",
            "RUS_NU_1232-8_X52Ph2_4",
            "4.7",
            "УРОВЕНЬ МОТОРНОГО МАСЛА: долив, заправка (2/4)",
            "Во избежание поврежде -",
            "ния узлов двигателя используйте воронку.",
            "75",
        ]
    )

    def setUp(self):
        self.page = parse_page(self.PAGE)

    def test_page_label_and_chapter(self):
        self.assertEqual(self.page["page_label"], "4.7")
        self.assertEqual(self.page["chapter"], 4)

    def test_section_heading_with_lowercase_tail(self):
        self.assertEqual(
            self.page["section"], "УРОВЕНЬ МОТОРНОГО МАСЛА: долив, заправка (2/4)"
        )

    def test_typographic_noise_is_dropped(self):
        self.assertNotIn("Jaune", self.page["text"])
        self.assertNotIn("RUS_UD57140_3", self.page["text"])

    def test_illustration_callouts_are_dropped_but_table_values_kept(self):
        lines = self.page["text"].split("\n")
        self.assertNotIn("1 2", lines)
        self.assertIn("75", lines, "трёхзначные и одиночные числа таблиц должны остаться")

    def test_hyphenated_line_break_is_joined(self):
        self.assertIn("повреждения узлов", self.page["text"])

    def test_uppercase_heading_without_colon(self):
        page = parse_page("6.7\nГАБАРИТНЫЕ РАЗМЕРЫ (в метрах) (1/3)\n4,357")
        self.assertEqual(page["section"], "ГАБАРИТНЫЕ РАЗМЕРЫ (в метрах) (1/3)")

    def test_chunks_respect_target_size(self):
        page = parse_page("5.1\nЗАГОЛОВОК РАЗДЕЛА\n" + "Это предложение про ремонт. " * 200)
        chunks = chunk_page(page, target=900, overlap=100)
        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(len(chunk) < 1400 for chunk in chunks))


class TestBm25(unittest.TestCase):
    CHUNKS = [
        {
            "id": "a",
            "doc": "ru_owner",
            "lang": "ru",
            "section": "УРОВЕНЬ МОТОРНОГО МАСЛА",
            "text": "Проверьте уровень масла щупом на холодном двигателе.",
        },
        {
            "id": "b",
            "doc": "ru_owner",
            "lang": "ru",
            "section": "ДАВЛЕНИЕ ВОЗДУХА В ШИНАХ",
            "text": "Давление в шинах проверяйте на холодных шинах.",
        },
        {
            "id": "c",
            "doc": "en_owner",
            "lang": "en",
            "section": "TYRE PRESSURE",
            "text": "Check the tyre pressure when the tyres are cold.",
        },
    ]

    def setUp(self):
        self.index = Bm25Index(self.CHUNKS, {"давление в шинах": ["tyre pressure"]})

    def test_finds_the_matching_chunk(self):
        hits = self.index.search("уровень масла", top_k=1)
        self.assertEqual(hits[0].chunk["id"], "a")

    def test_synonyms_cross_the_language_gap(self):
        ids = {hit.chunk["id"] for hit in self.index.search("давление в шинах", top_k=3)}
        self.assertIn("c", ids)

    def test_query_language_is_preferred(self):
        self.assertEqual(self.index.search("tyre pressure", top_k=1)[0].chunk["id"], "c")
        self.assertEqual(self.index.search("давление в шинах", top_k=1)[0].chunk["id"], "b")

    def test_filters(self):
        hits = self.index.search("cold", top_k=5, where={"doc": "en_owner"})
        self.assertTrue(all(hit.chunk["doc"] == "en_owner" for hit in hits))

    def test_per_doc_cap(self):
        hits = self.index.search("холодн", top_k=5, per_doc=1)
        docs = [hit.chunk["doc"] for hit in hits]
        self.assertEqual(len(docs), len(set(docs)))

    def test_typo_is_recovered_by_trigrams(self):
        hits = self.index.search("давлени в шынах", top_k=1)
        self.assertEqual(hits[0].chunk["id"], "b")

    def test_unknown_query_returns_nothing(self):
        self.assertEqual(self.index.search("карбюратор веберовский", top_k=3), [])

    def test_detect_lang(self):
        self.assertEqual(detect_lang("уровень масла"), "ru")
        self.assertEqual(detect_lang("oil level"), "en")
        self.assertIsNone(detect_lang("123"))


class TestCorpus(unittest.TestCase):
    """Проверки на настоящем корпусе — он лежит в data/chunks.jsonl."""

    @classmethod
    def setUpClass(cls):
        cls.chunks = load_chunks()
        cls.index = Bm25Index(cls.chunks, load_synonyms())

    def test_corpus_is_not_empty(self):
        self.assertGreater(len(self.chunks), 500)

    def test_every_chunk_can_be_cited(self):
        for chunk in self.chunks:
            self.assertTrue(citation(chunk).strip(), chunk["id"])

    def test_required_metadata(self):
        for chunk in self.chunks:
            for field in ("id", "doc", "doc_title", "lang", "text"):
                self.assertIn(field, chunk)
            self.assertGreaterEqual(len(chunk["text"]), 40, chunk["id"])

    def test_chunk_ids_are_unique(self):
        ids = [chunk["id"] for chunk in self.chunks]
        self.assertEqual(len(ids), len(set(ids)))

    def test_all_expected_documents_are_indexed(self):
        docs = {chunk["doc"] for chunk in self.chunks}
        self.assertEqual(docs, {"ru_owner", "en_owner", "wiring"})


class TestRetrievalQuality(unittest.TestCase):
    """Эвал: каждый запрос должен находить нужный раздел в топ-5."""

    @classmethod
    def setUpClass(cls):
        cls.index = Bm25Index(load_chunks(), load_synonyms())
        with (REPO_ROOT / "tests" / "eval_queries.jsonl").open(encoding="utf-8") as fh:
            cls.cases = [json.loads(line) for line in fh if line.strip()]

    def test_eval_set_is_meaningful(self):
        self.assertGreaterEqual(len(self.cases), 15)

    def test_every_query_hits_its_section(self):
        failures = []
        for case in self.cases:
            hits = self.index.search(
                case["query"], top_k=case.get("top_k", 5), where=case.get("filters")
            )
            haystack = " || ".join(
                citation(hit.chunk) + " " + hit.chunk["text"] for hit in hits
            ).lower()
            if not any(expected.lower() in haystack for expected in case["expect_any"]):
                failures.append(
                    f"{case['query']!r}: ждали {case['expect_any']}, "
                    f"получили {[citation(h.chunk) for h in hits]}"
                )
        self.assertEqual(failures, [], "\n" + "\n".join(failures))


if __name__ == "__main__":
    unittest.main()
