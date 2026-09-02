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

from ragkit.applicability import check as check_applicability
from ragkit.applicability import engines_in_text, restrict
from ragkit.bm25 import Bm25Index
from ragkit.chunker import chunk_page, chunk_unit, parse_page, parse_service_page, section_key
from ragkit.store import citation, image_path, load_chunks, load_synonyms, load_vehicle
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

    def test_english_forms_share_a_stem(self):
        for group in (
            ("fuse", "fuses"),
            ("pad", "pads"),
            ("bleed", "bleeding", "bleeds"),
            ("brake", "brakes"),
            ("remove", "removing", "removed"),
        ):
            stems = {stem(word) for word in group}
            self.assertEqual(len(stems), 1, f"{group} -> {stems}")

    def test_short_verbs_are_not_over_stemmed(self):
        # bleed -> ble сломало бы совпадение с bleeding
        for word in ("speed", "need", "seal", "head"):
            self.assertEqual(stem(word), word)

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
        self.assertIn("75", lines, "числа таблиц характеристик должны остаться")

    def test_hyphenated_line_break_is_joined(self):
        self.assertIn("повреждения узлов", self.page["text"])

    def test_uppercase_heading_without_colon(self):
        page = parse_page("6.7\nГАБАРИТНЫЕ РАЗМЕРЫ (в метрах) (1/3)\n4,357")
        self.assertEqual(page["section"], "ГАБАРИТНЫЕ РАЗМЕРЫ (в метрах) (1/3)")

    def test_section_key_merges_continuations(self):
        self.assertEqual(section_key("ЗАМЕНА КОЛЕСА (2/2)"), "ЗАМЕНА КОЛЕСА")
        self.assertEqual(section_key("ЗАМЕНА КОЛЕСА"), "ЗАМЕНА КОЛЕСА")

    def test_long_unit_is_split_at_paragraphs(self):
        chunks = chunk_unit("Это предложение про ремонт. " * 200, target=1300)
        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(len(chunk) <= 1400 for chunk in chunks))

    def test_page_chunking_still_works(self):
        chunks = chunk_page(self.page)
        self.assertTrue(chunks and all(chunks))


class TestServiceChunker(unittest.TestCase):
    PAGE = "\n".join(
        [
            "- 3 -",
            "Remove the clutch bolts (3) .",
            "2. REMOVAL OPERATION",
            "Torque tighten the clutch bolts 25 N.m .",
            "See (20A, Clutch) for details.",
            "XSL version : 3.02 du 22/07/11",
            "Repair-12x01x03x01-01x37-1-6-1.xml",
        ]
    )

    def setUp(self):
        self.page = parse_service_page(self.PAGE)

    def test_step_counter(self):
        self.assertEqual(self.page["step"], 3)

    def test_service_noise_is_dropped(self):
        self.assertNotIn("XSL version", self.page["text"])
        self.assertNotIn("Repair-12x01", self.page["text"])

    def test_numbered_substep_is_not_a_procedure_title(self):
        self.assertIsNone(self.page["section"])

    def test_renault_system_code(self):
        self.assertEqual(self.page["section_src"], "20A, Clutch")

    def test_torque_is_detected(self):
        self.assertTrue(self.page["has_torque"])


class TestBm25(unittest.TestCase):
    CHUNKS = [
        {
            "id": "a", "doc": "ru_owner", "lang": "ru", "section": "ЗАМЕНА КОЛЕСА",
            "text": "Затяните болты колеса. Сцепление шин с дорогой ухудшается.",
        },
        {
            "id": "b", "doc": "service_manual", "lang": "en", "section": "CLUTCH REMOVAL",
            "text": "Remove the clutch. Parts always to be replaced: clutch plate.",
        },
        {
            "id": "c", "doc": "en_owner", "lang": "en", "section": "TYRE PRESSURE",
            "text": "Check the tyre pressure when the tyres are cold.",
        },
        {
            "id": "d", "doc": "wiring", "lang": "ru", "section": "Схема двигателя 8V",
            "engines": ["K7M"], "text": "Схема жгута проводов двигателя, распиновка ЭБУ.",
        },
        {
            "id": "e", "doc": "wiring", "lang": "ru", "section": "Схема двигателя 16V",
            "engines": ["K4M"], "text": "Схема жгута проводов двигателя, распиновка ЭБУ.",
        },
    ]
    SYNONYMS = {
        "замена": ["removal", "replaced"],
        "сцепление": ["clutch"],
        "давление в шинах": ["tyre pressure"],
        "прокачка": ["удаление воздуха"],
    }

    def setUp(self):
        self.index = Bm25Index(self.CHUNKS, self.SYNONYMS)

    def test_finds_the_matching_chunk(self):
        self.assertEqual(self.index.search("болты колеса", top_k=1)[0].chunk["id"], "a")

    def test_cross_language_synonym_reaches_english_manual(self):
        self.assertEqual(self.index.search("замена сцепления", top_k=1)[0].chunk["id"], "b")

    def test_concept_coverage_beats_repetition(self):
        """Страница про замену колеса упоминает «сцепление» в смысле сцепления
        шин с дорогой — она не должна обойти процедуру про сцепление."""
        hits = self.index.search("замена сцепления", top_k=3)
        self.assertEqual(hits[0].chunk["id"], "b")

    def test_multiword_synonym_needs_all_of_its_words(self):
        """«удаление воздуха» не должно срабатывать по одному слову «воздух»."""
        concepts = self.index.expand("прокачка")
        phrases = [phrase for concept in concepts for phrase, _ in concept["phrases"]]
        self.assertTrue(any(len(phrase) > 1 for phrase in phrases))
        for concept in concepts:
            self.assertNotIn("воздух", concept["terms"])

    def test_filters(self):
        hits = self.index.search("cold", top_k=5, where={"doc": "en_owner"})
        self.assertTrue(all(hit.chunk["doc"] == "en_owner" for hit in hits))

    def test_per_doc_cap(self):
        hits = self.index.search("clutch tyre колеса", top_k=5, per_doc=1)
        docs = [hit.chunk["doc"] for hit in hits]
        self.assertEqual(len(docs), len(set(docs)))

    def test_chunk_of_another_engine_is_skipped(self):
        ids = {h.chunk["id"] for h in self.index.search("распиновка ЭБУ", top_k=5, engine="K7M")}
        self.assertIn("d", ids)
        self.assertNotIn("e", ids, "схема 16V не относится к K7M")

    def test_untagged_chunks_survive_the_engine_filter(self):
        """Метка есть не у всех: схема кузова к мотору не привязана."""
        ids = {h.chunk["id"] for h in self.index.search("tyre pressure", top_k=5, engine="K7M")}
        self.assertIn("c", ids)

    def test_typo_is_recovered_by_trigrams(self):
        self.assertEqual(self.index.search("сцеплени", top_k=1)[0].chunk["id"], "b")

    def test_unknown_query_returns_nothing(self):
        self.assertEqual(self.index.search("карбюратор веберовский", top_k=3), [])


class TestCorpus(unittest.TestCase):
    """Проверки на настоящем корпусе — он лежит в data/chunks.jsonl."""

    @classmethod
    def setUpClass(cls):
        cls.chunks = load_chunks()
        cls.index = Bm25Index(cls.chunks, load_synonyms())

    def test_corpus_is_not_empty(self):
        self.assertGreater(len(self.chunks), 1000)

    def test_every_chunk_can_be_cited(self):
        for chunk in self.chunks:
            self.assertTrue(citation(chunk).strip(), chunk["id"])

    def test_required_metadata(self):
        for chunk in self.chunks:
            for field in ("id", "doc", "doc_title", "lang", "text", "unit_parts"):
                self.assertIn(field, chunk)
            self.assertGreaterEqual(len(chunk["text"]), 40, chunk["id"])

    def test_chunk_ids_are_unique(self):
        ids = [chunk["id"] for chunk in self.chunks]
        self.assertEqual(len(ids), len(set(ids)))

    def test_all_expected_documents_are_indexed(self):
        docs = {chunk["doc"] for chunk in self.chunks}
        self.assertEqual(docs, {"ru_owner", "en_owner", "service_manual", "wiring"})

    def test_atomic_units_keep_their_title(self):
        """Каждая часть единицы несёт её название — фрагмент без контекста
        бесполезен и в выдаче, и в ссылке."""
        multi = [c for c in self.chunks if c["unit_parts"] > 1 and c["doc"] != "wiring"]
        self.assertGreater(len(multi), 100)
        without_title = [c["id"] for c in multi if not c.get("section")]
        self.assertLess(len(without_title) / len(multi), 0.05, without_title[:5])

    def test_every_illustration_ships_with_the_skill(self):
        """Скилл должен уметь показать схему, а не только сослаться на неё."""
        images = [c for c in self.chunks if c.get("image")]
        self.assertGreaterEqual(len({c["image"] for c in images}), 22)
        missing = sorted({c["image"] for c in images if image_path(c) is None})
        self.assertEqual(missing, [], f"нет файлов иллюстраций: {missing}")

    def test_engine_specific_diagrams_are_tagged(self):
        """Схемы 8V и 16V — это K7M и K4M: без метки поиск смешал бы их."""
        tagged = {c["image"]: c["engines"] for c in self.chunks if c.get("engines")}
        self.assertEqual(tagged.get("13.jpg"), ["K7M"])
        self.assertEqual(tagged.get("21.jpg"), ["K4M"])

    def test_service_manual_pages_are_globally_numbered(self):
        pages = [c["pdf_page"] for c in self.chunks if c["doc"] == "service_manual"]
        self.assertGreater(max(pages), 3000, "нумерация должна быть сквозной по всему PDF")


class TestApplicability(unittest.TestCase):
    """Документ, не покрывающий названный двигатель, не должен попадать в ответ."""

    @classmethod
    def setUpClass(cls):
        cls.vehicle = load_vehicle()

    def test_vehicle_reference_is_present(self):
        self.assertIn("engines", self.vehicle)
        self.assertIn("document_engines", self.vehicle)

    def test_engine_codes_are_recognised(self):
        self.assertEqual(engines_in_text("замена цепи грм h4m 1.6"), ["H4M"])
        self.assertEqual(engines_in_text("HR16DE"), ["H4M"], "ниссановское имя того же мотора")
        self.assertEqual(engines_in_text("моменты затяжки"), [])

    def test_service_manual_is_blocked_for_h4m(self):
        result = check_applicability("замена цепи ГРМ H4M", self.vehicle)
        self.assertEqual(result["engine"], "H4M")
        self.assertIn("service_manual", result["blocked"])
        self.assertIn("wiring", result["blocked"])
        self.assertTrue(any("H4M" in w for w in result["warnings"]))

    def test_service_manual_stays_for_covered_engine(self):
        result = check_applicability("замена ремня ГРМ K4M", self.vehicle)
        self.assertIn("service_manual", result["allowed"])
        self.assertEqual(result["blocked"], [])

    def test_wrong_timing_drive_is_called_out(self):
        warnings = check_applicability("замена ремня ГРМ H4M", self.vehicle)["warnings"]
        self.assertTrue(any("неприменим" in w for w in warnings))

    def test_matching_timing_drive_is_not_an_alarm(self):
        warnings = check_applicability("замена ремня ГРМ K4M", self.vehicle)["warnings"]
        self.assertFalse([w for w in warnings if "ВНИМАНИЕ" in w])

    def test_query_without_engine_restricts_nothing(self):
        result = check_applicability("прокачка тормозной системы", self.vehicle)
        self.assertIsNone(result["allowed"])
        self.assertEqual(restrict({"lang": "ru"}, result), {"lang": "ru"})

    def test_profile_engine_is_applied_without_a_code_in_the_query(self):
        profile = (self.vehicle.get("profile") or {}).get("engine")
        self.assertTrue(profile, "профиль автомобиля должен быть заполнен")
        result = check_applicability("распиновка ЭБУ", self.vehicle, profile_engine=profile)
        self.assertEqual(result["engine"], profile)

    def test_engine_named_in_the_query_overrides_the_profile(self):
        result = check_applicability("распиновка ЭБУ K4M", self.vehicle, profile_engine="K7M")
        self.assertEqual(result["engine"], "K4M")

    def test_restrict_intersects_with_requested_doc(self):
        result = check_applicability("цепь ГРМ H4M", self.vehicle)
        self.assertEqual(restrict({"doc": "service_manual"}, result)["doc"], [])
        self.assertIn("ru_owner", restrict(None, result)["doc"])


class TestRetrievalQuality(unittest.TestCase):
    """Эвал: каждый запрос должен находить нужный раздел в топ-5."""

    @classmethod
    def setUpClass(cls):
        cls.index = Bm25Index(load_chunks(), load_synonyms())
        cls.vehicle = load_vehicle()
        with (REPO_ROOT / "tests" / "eval_queries.jsonl").open(encoding="utf-8") as fh:
            cls.cases = [json.loads(line) for line in fh if line.strip()]

    def test_eval_set_is_meaningful(self):
        self.assertGreaterEqual(len(self.cases), 40)
        queries = [case["query"] for case in self.cases]
        self.assertEqual(len(queries), len(set(queries)), "дубли в эвале ничего не проверяют")
        for case in self.cases:
            self.assertTrue(case.get("expect_any") or case.get("forbid"), case["query"])

    def _search(self, case):
        """Ищет так же, как CLI: с проверкой применимости к двигателю."""
        applicability = check_applicability(case["query"], self.vehicle)
        where = restrict(case.get("filters"), applicability)
        return self.index.search(
            case["query"],
            top_k=case.get("top_k", 5),
            where=where,
            engine=applicability["engine"],
        )

    def test_every_query_hits_its_section(self):
        failures = []
        for case in self.cases:
            if not case.get("expect_any"):
                continue
            hits = self._search(case)
            haystack = " || ".join(
                citation(hit.chunk) + " " + hit.chunk["text"] for hit in hits
            ).lower()
            if not any(expected.lower() in haystack for expected in case["expect_any"]):
                failures.append(
                    f"{case['query']!r}: ждали {case['expect_any']}, "
                    f"получили {[citation(h.chunk) for h in hits]}"
                )
        self.assertEqual(failures, [], "\n" + "\n".join(failures))

    def test_inapplicable_documents_never_answer(self):
        """Процедура от другого двигателя опаснее, чем отсутствие ответа."""
        failures = []
        for case in self.cases:
            if not case.get("forbid"):
                continue
            hits = self._search(case)
            haystack = " || ".join(
                citation(hit.chunk) + " " + hit.chunk["text"] for hit in hits
            ).lower()
            for forbidden in case["forbid"]:
                if forbidden.lower() in haystack:
                    failures.append(
                        f"{case['query']!r}: в выдаче оказалось {forbidden!r} — "
                        f"{case.get('note', '')}"
                    )
        self.assertEqual(failures, [], "\n" + "\n".join(failures))


if __name__ == "__main__":
    unittest.main()
