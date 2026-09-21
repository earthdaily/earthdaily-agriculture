"""Unit tests for api_utils.safe_parse_date.

Covers the ISO path, placeholder handling, the MM-DD historical-average
fallback (including the 02-29 leap-day case that a bare "%m-%d" — defaulting
to the non-leap year 1900 — used to drop), and the pass-through for
unrecognised input.
"""

import pytest

from earthdaily.agriculture.core.api_utils import safe_parse_date

pytestmark = pytest.mark.public


class TestSafeParseDate:
    def test_iso_date_returned_as_iso(self):
        assert safe_parse_date("2026-04-22") == "2026-04-22"

    @pytest.mark.parametrize("placeholder", ["", None, "0001-01-01", "0000-00-00"])
    def test_placeholders_return_none(self, placeholder):
        assert safe_parse_date(placeholder) is None

    def test_mm_dd_roundtrips(self):
        assert safe_parse_date("04-22") == "04-22"

    def test_mm_dd_leap_day_validates(self):
        # Regression: bare "%m-%d" defaults to 1900 (non-leap) and dropped 02-29.
        assert safe_parse_date("02-29") == "02-29"

    def test_invalid_mm_dd_returned_as_is(self):
        # 13-40 is neither a valid full date nor a valid MM-DD → left untouched.
        assert safe_parse_date("13-40") == "13-40"

    def test_unrecognised_string_returned_as_is(self):
        assert safe_parse_date("not-a-date") == "not-a-date"
