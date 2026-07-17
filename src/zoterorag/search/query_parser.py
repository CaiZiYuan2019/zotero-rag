"""Query preprocessing for FTS5 metadata and fulltext search.

The parser turns a raw user query into a :class:`QueryPlan` and can render that
plan into an FTS5 ``MATCH`` expression. It intentionally stays dependency-free
and does no stemming; it only understands quoted phrases, ``term*`` prefixes,
``+must`` / ``-exclude`` operators, and a small English stopword list.

Design constraints (see the retrieval enhancement plan):

* Default combination is ``AND`` so historical AND-only behaviour is preserved.
* ``match_mode="any"`` combines positive clauses with ``OR``.
* Plain terms are still quoted so FTS5 special characters cannot break the
  ``MATCH`` expression; prefix terms are quoted then suffixed with ``*``.
* If every meaningful token is a stopword the plan is *empty*; callers must
  treat an empty plan as "no text query" instead of emitting an illegal MATCH.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import re


# A deliberately small English stopword list. Chinese stopwords are out of
# scope for this phase.
DEFAULT_STOPWORDS = frozenset(
    {
        "a", "an", "and", "are", "as", "at", "be", "by", "for", "from",
        "in", "into", "is", "it", "of", "on", "or", "that", "the", "to",
        "was", "were", "with",
    }
)


_TOKEN_RE = re.compile(r'"[^"]*"|\S+')


@dataclass(frozen=True)
class QueryPlan:
    phrases: list[str] = field(default_factory=list)
    prefixes: list[str] = field(default_factory=list)
    must: list[str] = field(default_factory=list)
    exclude: list[str] = field(default_factory=list)
    terms: list[str] = field(default_factory=list)
    match_mode: str = "all"

    @property
    def is_empty(self) -> bool:
        # Excludes alone cannot form a valid MATCH; only positive clauses make
        # a plan searchable.
        return not (self.phrases or self.prefixes or self.must or self.terms)


def parse_query(
    query: str,
    *,
    match_mode: str = "all",
    stopwords: frozenset[str] | set[str] | None = None,
) -> QueryPlan:
    if match_mode not in {"all", "any"}:
        raise ValueError("match_mode must be 'all' or 'any'")
    stop = DEFAULT_STOPWORDS if stopwords is None else frozenset(w.casefold() for w in stopwords)

    phrases: list[str] = []
    prefixes: list[str] = []
    must: list[str] = []
    exclude: list[str] = []
    terms: list[str] = []

    for raw in _TOKEN_RE.findall(query or ""):
        token = raw.strip()
        if not token:
            continue
        if token.startswith('"') and token.endswith('"'):
            phrase = token[1:-1].strip()
            if phrase:
                phrases.append(phrase.casefold())
            continue
        sign = ""
        if token[0] in "+-":
            sign = token[0]
            token = token[1:].strip()
            if not token:
                continue
        is_prefix = token.endswith("*") and len(token) > 1
        if is_prefix:
            token = token[:-1]
        word = token.casefold()
        if not word:
            continue
        if sign == "-":
            exclude.append(word)
            continue
        if sign == "+":
            # Explicit must-terms are kept even if they are stopwords.
            (prefixes if is_prefix else must).append(word)
            continue
        if is_prefix:
            prefixes.append(word)
            continue
        if word in stop:
            continue
        terms.append(word)

    return QueryPlan(
        phrases=phrases,
        prefixes=prefixes,
        must=must,
        exclude=exclude,
        terms=terms,
        match_mode=match_mode,
    )


def _quote(term: str) -> str:
    return '"' + term.replace('"', '""') + '"'


def build_fts_query(plan: QueryPlan) -> str:
    """Render *plan* into an FTS5 MATCH expression (empty string if unusable)."""

    if plan.is_empty:
        return ""
    positives = [_quote(p) for p in plan.phrases]
    positives.extend(f"{_quote(p)}*" for p in plan.prefixes)
    positives.extend(_quote(t) for t in plan.terms)

    joiner = " AND " if plan.match_mode == "all" else " OR "
    expr = joiner.join(positives) if positives else ""

    must_clause = " AND ".join(_quote(m) for m in plan.must)
    if must_clause:
        expr = f"({expr}) AND {must_clause}" if expr else must_clause

    for term in plan.exclude:
        if expr:
            expr = f"{expr} NOT {_quote(term)}"
    return expr


def scoring_terms(plan: QueryPlan) -> list[str]:
    """Positive, casefolded tokens used for Python substring scoring."""

    terms: list[str] = []
    for phrase in plan.phrases:
        terms.extend(part for part in phrase.split() if part)
    terms.extend(plan.prefixes)
    terms.extend(plan.terms)
    terms.extend(plan.must)
    # Preserve order while de-duplicating.
    seen: set[str] = set()
    unique: list[str] = []
    for term in terms:
        if term not in seen:
            seen.add(term)
            unique.append(term)
    return unique
