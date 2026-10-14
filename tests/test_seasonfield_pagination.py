"""
Tests for the seasonfield loading resilience (GEOSD-5964):
    - get_seasonfield_list() pages with $limit/$offset/$sort=id instead of one
      $limit=none request, and stops on the first short page
    - 5xx responses are retried, 4xx are not
    - an API ignoring $offset raises instead of looping forever
    - get_season_fields() retries its SEARCH batches on 5xx

The HTTP layer and the retry back-off sleep are mocked.
"""

from unittest.mock import MagicMock, patch

import pytest
import requests

from earthdaily.agriculture.services import entity_management as em

pytestmark = pytest.mark.public

GET = "earthdaily.agriculture.services.entity_management.requests.get"
REQUEST = "earthdaily.agriculture.services.entity_management.requests.request"
SLEEP = "earthdaily.agriculture.core.api_utils.time.sleep"


def _ok(rows):
    resp = MagicMock()
    resp.status_code = 200
    resp.json.return_value = rows
    resp.raise_for_status.return_value = None
    return resp


def _error(status):
    resp = MagicMock()
    resp.status_code = status
    resp.text = "error"
    resp.raise_for_status.side_effect = requests.exceptions.HTTPError(f"{status} error", response=resp)
    return resp


def _rows(*ids):
    return [{"id": i, "crop": {"id": "WINTER_OSR"}} for i in ids]


def _urls(mock_get):
    return [c.args[0] for c in mock_get.call_args_list]


@patch(SLEEP)
@patch(GET)
class TestGetSeasonfieldListPagination:
    def test_pages_until_a_short_page(self, mock_get, _sleep):
        mock_get.side_effect = [_ok(_rows("a", "b")), _ok(_rows("c", "d")), _ok(_rows("e"))]

        df = em.get_seasonfield_list("prod", "TOKEN", crop_id="WINTER_OSR", page_size=2)

        assert df["id"].tolist() == ["a", "b", "c", "d", "e"]
        urls = _urls(mock_get)
        assert len(urls) == 3
        assert all("$sort=id&$limit=2" in u for u in urls)
        assert [u.rsplit("$offset=", 1)[1] for u in urls] == ["0", "2", "4"]
        assert not any("$limit=none" in u for u in urls)
        assert all("crop.id=WINTER_OSR" in u for u in urls)  # filters kept on every page

    def test_exact_multiple_of_page_size_ends_on_empty_page(self, mock_get, _sleep):
        mock_get.side_effect = [_ok(_rows("a", "b")), _ok(_rows("c", "d")), _ok([])]

        df = em.get_seasonfield_list("prod", "TOKEN", page_size=2)

        assert df["id"].tolist() == ["a", "b", "c", "d"]
        assert mock_get.call_count == 3

    def test_no_seasonfield_returns_empty_dataframe(self, mock_get, _sleep):
        mock_get.side_effect = [_ok([])]

        assert em.get_seasonfield_list("prod", "TOKEN", page_size=2).empty

    def test_5xx_is_retried(self, mock_get, _sleep):
        mock_get.side_effect = [_error(500), _ok(_rows("a"))]

        df = em.get_seasonfield_list("prod", "TOKEN", page_size=2)

        assert df["id"].tolist() == ["a"]
        assert mock_get.call_count == 2

    def test_persistent_5xx_raises_after_max_attempts(self, mock_get, _sleep):
        mock_get.side_effect = [_error(500)] * em.SEASONFIELD_MAX_ATTEMPTS

        with pytest.raises(RuntimeError):
            em.get_seasonfield_list("prod", "TOKEN", page_size=2)
        assert mock_get.call_count == em.SEASONFIELD_MAX_ATTEMPTS

    def test_4xx_is_not_retried(self, mock_get, _sleep):
        mock_get.side_effect = [_error(403)]

        with pytest.raises(ValueError):
            em.get_seasonfield_list("prod", "TOKEN", page_size=2)
        assert mock_get.call_count == 1

    def test_offset_ignored_by_api_raises_instead_of_looping(self, mock_get, _sleep):
        mock_get.side_effect = [_ok(_rows("a", "b")), _ok(_rows("a", "b"))]

        with pytest.raises(RuntimeError, match="pagination not honoured"):
            em.get_seasonfield_list("prod", "TOKEN", page_size=2)


@patch(SLEEP)
@patch(REQUEST)
def test_get_season_fields_retries_search_batch_on_5xx(mock_request, _sleep):
    item = {"id": "a", "crop": {"id": "winter_osr"}, "sowingDate": "2026-09-01T00:00:00", "geometry": "POINT (0 0)"}
    mock_request.side_effect = [_error(500), _ok([item])]

    df = em.get_season_fields("https://fake", "TOKEN", ids=["a"], batch_size=5000)

    assert df["id"].tolist() == ["a"]
    assert mock_request.call_count == 2
    assert all(c.args[0] == "SEARCH" for c in mock_request.call_args_list)
