from __future__ import annotations

import unittest

from zoterorag.search.query_parser import build_fts_query, parse_query, scoring_terms


class QueryParserTests(unittest.TestCase):
    def test_phrase_is_parsed_and_quoted(self) -> None:
        plan = parse_query('"machine learning" models')
        self.assertEqual(["machine learning"], plan.phrases)
        self.assertIn("models", plan.terms)
        fts = build_fts_query(plan)
        self.assertIn('"machine learning"', fts)
        self.assertIn('"models"', fts)
        self.assertIn(" AND ", fts)

    def test_prefix_uses_star_and_is_not_fully_quoted_closed(self) -> None:
        plan = parse_query("neur*")
        self.assertEqual(["neur"], plan.prefixes)
        fts = build_fts_query(plan)
        self.assertEqual('"neur"*', fts)

    def test_stopwords_removed_and_all_stopwords_yields_empty_plan(self) -> None:
        plan = parse_query("the of a")
        self.assertTrue(plan.is_empty)
        self.assertEqual("", build_fts_query(plan))

        mixed = parse_query("the transformer")
        self.assertEqual(["transformer"], mixed.terms)

    def test_any_mode_uses_or(self) -> None:
        plan = parse_query("alpha beta", match_mode="any")
        fts = build_fts_query(plan)
        self.assertIn(" OR ", fts)
        self.assertNotIn(" AND ", fts)

    def test_must_and_exclude_operators(self) -> None:
        plan = parse_query("+neural -survey topic")
        self.assertIn("neural", plan.must)
        self.assertIn("survey", plan.exclude)
        self.assertIn("topic", plan.terms)
        fts = build_fts_query(plan)
        self.assertIn('"neural"', fts)
        self.assertIn('NOT "survey"', fts)

    def test_exclude_only_query_is_empty(self) -> None:
        plan = parse_query("-only")
        self.assertTrue(plan.is_empty)
        self.assertEqual("", build_fts_query(plan))

    def test_scoring_terms_are_positive_and_deduplicated(self) -> None:
        plan = parse_query('"deep learning" deep +models neur*')
        terms = scoring_terms(plan)
        self.assertIn("deep", terms)
        self.assertIn("learning", terms)
        self.assertIn("models", terms)
        self.assertIn("neur", terms)
        self.assertEqual(len(terms), len(set(terms)))


if __name__ == "__main__":
    unittest.main()
