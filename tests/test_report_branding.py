"""
Tests for the corporate mark on generated reports (docs/internal/19, §26):
    - embedded as a data: URI, never hotlinked — the reports are self-contained,
      so an external <img> breaks under a strict CSP and offline
    - linked to the corporate site, with a real alt and safe rel
    - exactly one mark per surface
    - a missing asset degrades to a text link rather than failing the report
"""

from unittest.mock import patch

import pandas as pd
import pytest

from earthdaily.agriculture.reporting.extraction_reporter import (
    _LOGO_RENDER_HEIGHT,
    BRAND_URL,
    DOCS_URL,
    FOOTER_LINKS,
    PRIVACY_URL,
    TERMS_URL,
    ExtractionReporter,
    _logo_data_uri,
    _render_brand,
)

pytestmark = pytest.mark.public


@pytest.fixture
def report_html():
    _logo_data_uri.cache_clear()
    reporter = ExtractionReporter(include_data_preview=False)
    reporter.set_extraction_context(extractor_name="CoverageExtractor", prefix="coverage", env="preprod")
    reporter.set_results(
        results_df=pd.DataFrame([{"entity_id": "f1", "coverage_percent": 98}]),
        total_entities=1,
        total_calculations=1,
        successful=1,
        failed=0,
        elapsed_seconds=1.0,
    )
    return reporter.render_html()


class TestMarkIsEmbedded:
    def test_logo_is_a_data_uri(self, report_html):
        assert "data:image/png;base64," in report_html

    def test_no_external_image_host(self, report_html):
        """A remote <img> is the failure §26 exists to prevent."""
        assert "earthdaily.com/hubfs" not in report_html
        assert "<img src='http" not in report_html and '<img src="http' not in report_html

    def test_asset_decodes_to_a_transparent_png(self):
        import base64
        import io

        from PIL import Image

        _logo_data_uri.cache_clear()
        uri = _logo_data_uri()
        assert uri is not None, "the packaged asset should be readable in a dev install"
        img = Image.open(io.BytesIO(base64.b64decode(uri.split(",", 1)[1])))
        rgba = img.convert("RGBA")
        w, h = rgba.size
        corners = [rgba.getchannel("A").getpixel(p) for p in ((0, 0), (w - 1, 0), (0, h - 1), (w - 1, h - 1))]
        assert max(corners) == 0, "the mark must keep its clear background"
        assert h == 2 * _LOGO_RENDER_HEIGHT, "the asset must stay 2x the render height for HiDPI"


class TestMarkIsALink:
    def test_links_to_the_corporate_site(self, report_html):
        assert BRAND_URL in report_html
        assert f"href='{BRAND_URL}'" in report_html

    def test_new_tab_link_is_not_a_tabnabbing_hole(self, report_html):
        assert "rel='noopener noreferrer'" in report_html

    def test_alt_text_is_present(self, report_html):
        assert "alt='EarthDaily'" in report_html


class TestPlacement:
    def test_exactly_one_mark_per_report(self, report_html):
        assert report_html.count("class='er-brand'") == 1

    def test_mark_sits_in_the_hero(self, report_html):
        hero = report_html.split("er-hero-head", 1)[1].split("</div>", 4)[0]
        assert "er-brand" in report_html.split("er-hero-head", 1)[1][: len(hero) + 400]

    def test_css_ships_the_brand_rules(self, report_html):
        assert ".er-brand" in report_html and ".er-hero-head" in report_html

    def test_render_height_matches_the_css(self, report_html):
        """The <img> height and the stylesheet must not drift apart."""
        assert f"height='{_LOGO_RENDER_HEIGHT}'" in report_html
        assert f"height: {_LOGO_RENDER_HEIGHT}px" in report_html


class TestFooterLinks:
    def test_footer_links_to_the_documentation(self, report_html):
        assert DOCS_URL in report_html
        assert f"href='{DOCS_URL}'" in report_html

    def test_docs_link_is_in_the_footer_not_the_hero(self, report_html):
        footer = report_html.split("class='er-footer'", 1)[1]
        assert DOCS_URL in footer
        hero = report_html.split("class='er-hero'", 1)[1].split("class='er-footer'", 1)[0]
        assert DOCS_URL not in hero

    def test_docs_link_opens_safely(self, report_html):
        segment = report_html.split(DOCS_URL, 1)[1][:80]
        assert "rel='noopener noreferrer'" in segment

    def test_terms_and_privacy_are_linked(self, report_html):
        """§26: a surface attributed to EarthDaily says where its terms live."""
        footer = report_html.split("class='er-footer'", 1)[1]
        for url in (TERMS_URL, PRIVACY_URL):
            assert url in footer, url

    def test_footer_link_order_is_useful_then_legal(self, report_html):
        footer = report_html.split("class='er-footer'", 1)[1]
        positions = [footer.index(url) for _, url in FOOTER_LINKS]
        assert positions == sorted(positions), "documentation first, then the legal pair"

    def test_every_footer_link_opens_safely(self, report_html):
        footer = report_html.split("class='er-footer'", 1)[1]
        for _, url in FOOTER_LINKS:
            assert "rel='noopener noreferrer'" in footer.split(url, 1)[1][:80], url


class TestGracefulDegradation:
    def test_missing_asset_falls_back_to_a_text_link(self):
        """Branding must never be the reason a report fails."""
        _logo_data_uri.cache_clear()
        with patch("earthdaily.agriculture.reporting.extraction_reporter._logo_data_uri", return_value=None):
            markup = _render_brand()
        assert "<img" not in markup
        assert BRAND_URL in markup and "EarthDaily" in markup

    def test_unreadable_asset_returns_none_rather_than_raising(self):
        _logo_data_uri.cache_clear()
        with patch("importlib.resources.files", side_effect=OSError("package data stripped")):
            assert _logo_data_uri() is None
        _logo_data_uri.cache_clear()

    def test_report_still_renders_without_the_asset(self):
        _logo_data_uri.cache_clear()
        with patch("earthdaily.agriculture.reporting.extraction_reporter._logo_data_uri", return_value=None):
            reporter = ExtractionReporter(include_data_preview=False)
            reporter.set_extraction_context(extractor_name="CoverageExtractor", prefix="coverage")
            reporter.set_results(results_df=pd.DataFrame(), total_entities=0, total_calculations=0)
            html = reporter.render_html()
        assert "CoverageExtractor Report" in html
        assert "data:image/png" not in html
