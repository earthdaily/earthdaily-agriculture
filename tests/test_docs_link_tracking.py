"""Guards for UTM tagging of links to docs.earthdaily.com.

Convention: earthdaily-documentation-WIP ``contributing/11-link-tracking.md``. Every link
to the docs from an EarthDaily-owned surface carries UTM tags, so GA4 can attribute the
visit; links opened from a terminal, a notebook or a ``noreferrer`` report footer
otherwise arrive with no referrer and count as *Direct*. For this package that means
``utm_source=github&utm_medium=repo&utm_campaign=earthdaily-agriculture`` plus a
``utm_content`` naming the placement.

The inverse rule matters as much: **links inside the docs site are never tagged** — a
UTM tag starts a new GA4 session and overwrites the visitor's real source. The docs
pages this repo publishes must stay clean.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

pytestmark = pytest.mark.public

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC = PROJECT_ROOT / "src" / "earthdaily" / "agriculture"
PACKAGE_TAGS = ("utm_source=github", "utm_medium=repo", "utm_campaign=earthdaily-agriculture")
DOCS_URL_RE = re.compile(r"https://docs\.earthdaily\.com/\S*")


def _src_lines():
    for path in sorted(SRC.rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if "https://docs.earthdaily.com" in line:
                yield path.relative_to(SRC).as_posix(), lineno, line


def test_every_docstring_docs_link_is_tagged():
    """`Documentation:` lines feed help(), IDEs and every generated CLAUDE.md / skill."""
    untagged = []
    for rel, lineno, line in _src_lines():
        if "Documentation:" not in line:
            continue
        url = DOCS_URL_RE.search(line).group(0)
        if not all(tag in url for tag in PACKAGE_TAGS) or "utm_content=docstring" not in url:
            untagged.append(f"{rel}:{lineno}: {url}")
    assert not untagged, "docstring docs links missing the package UTM tags:\n" + "\n".join(untagged)


def test_no_untagged_docs_url_literal_in_code():
    """Outside docstrings, docs URLs are built by ``docs_link()``, never written by hand."""
    stray = [
        f"{rel}:{lineno}: {line.strip()}"
        for rel, lineno, line in _src_lines()
        if "Documentation:" not in line and not ("extraction_reporter.py" in rel and "{path}" in line)
    ]
    assert not stray, "route docs URLs through reporting.extraction_reporter.docs_link():\n" + "\n".join(stray)


def test_docs_link_helper_tags_every_link():
    from earthdaily.agriculture.reporting.extraction_reporter import docs_link

    url = docs_link("agro/cropid/", "example")
    assert url.startswith("https://docs.earthdaily.com/agro/cropid/?")
    assert all(tag in url for tag in PACKAGE_TAGS)
    assert url.endswith("utm_content=example")


def test_published_docs_pages_carry_no_utm_tags():
    """Never tag links inside the docs site — they would overwrite the visitor's source."""
    # docs/site/agriculture in this repo; the release remaps it to docs/ in the public one.
    site = PROJECT_ROOT / "docs" / "site"
    pages = site if site.is_dir() else PROJECT_ROOT / "docs"
    if not pages.is_dir():
        pytest.skip("no docs pages in this checkout")
    tagged = [
        p.relative_to(PROJECT_ROOT).as_posix()
        for p in pages.rglob("*.md")
        if "utm_" in p.read_text(encoding="utf-8", errors="replace")
    ]
    assert not tagged, f"UTM tags inside published docs pages: {tagged}"


# ---------------------------------------------------------------------------
# The AI-context generator: tagged links only where they leave this repo
# ---------------------------------------------------------------------------

TAGGED = (
    "https://docs.earthdaily.com/agro/library/Api_reference/"
    "?utm_source=github&utm_medium=repo&utm_campaign=earthdaily-agriculture&utm_content=docstring"
)


@pytest.fixture(scope="module")
def gen():
    from earthdaily.agriculture.ai_enablement import generate_ai_context

    return generate_ai_context


def _entry(gen):
    parsed = gen.parse_docstring(f"Extracts things for entities.\n\n    Documentation: {TAGGED}\n")
    return {
        "class_name": "DemoExtractor",
        "module": "earthdaily.agriculture.extractors.demo",
        "category": "Foundational",
        "description": "Extracts things for entities, long enough for the gap check.",
        "documentation_url": parsed["documentation_url"],
        "documentation_link": parsed["documentation_link"],
    }


def test_parse_keeps_both_the_canonical_url_and_the_tagged_link(gen):
    parsed = gen.parse_docstring(f"Desc.\n\n    Documentation: {TAGGED}\n")
    assert parsed["documentation_url"] == "https://docs.earthdaily.com/agro/library/Api_reference/"
    assert parsed["documentation_link"] == TAGGED


def test_repo_agents_md_uses_the_canonical_url(gen):
    """This repo's own CLAUDE.md / agents.md: the team's clicks are not package traffic."""
    out = gen.generate_agents_md([_entry(gen)])
    assert "utm_" not in out
    assert "https://docs.earthdaily.com/agro/library/Api_reference/" in out


def test_project_and_skill_extractors_md_keep_the_tags(gen):
    out = gen.generate_agents_md([_entry(gen)], tracked_links=True)
    assert TAGGED in out


def test_gap_check_still_sees_the_generic_api_reference(gen):
    """Regression: a query string hid the ``/Api_reference/`` suffix from the gap check."""
    gaps = gen.analyze_gaps([_entry(gen)])
    assert "generic Api_reference" in str(gaps), gaps
