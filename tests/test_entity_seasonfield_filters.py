"""
Tests for EntityManager seasonfield farm filtering:
    - get_seasonfields() query-param construction for the farm_name filter
      (single string, list of names -> $in:, None/empty, single-element list)
    - id-level dedup of the returned DataFrame
    - load_seasonfields() threads str | list[str] through and logs both shapes

The HTTP layer is mocked; assertions are on the exact params sent to the MDM API.
"""

from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

import pytest

from earthdaily.agriculture.services.entity_management import EntityManager

pytestmark = pytest.mark.public


def _make_manager():
    """Minimal EntityManager that skips real auth/network setup.

    Sets only the attributes get_seasonfields / load_seasonfields touch; a future
    token_expiration + workflow_ref=None makes the @requires_token gate a no-op.
    """
    mgr = EntityManager.__new__(EntityManager)
    mgr.bearer_token = "FAKE_TOKEN"
    mgr.token_expiration = datetime.now() + timedelta(hours=1)
    mgr.workflow_ref = None
    mgr.env = "preprod"
    mgr.mdm_url = "https://fake.mdm/v6"
    mgr.logger = MagicMock()
    mgr.extractor_name = "EntityManager"
    return mgr


def _mock_response(rows):
    resp = MagicMock()
    resp.json.return_value = rows
    resp.raise_for_status.return_value = None
    return resp


def _sf(sf_id, farm):
    """One season-field record shaped like the json_normalize input."""
    return {"id": sf_id, "field": {"farm": {"name": farm}}}


# ===================================================================
# get_seasonfields() — farm_name param construction
# ===================================================================


class TestFarmNameParam:
    @patch("earthdaily.agriculture.services.entity_management.requests.get")
    def test_single_string_is_bare_value(self, mock_get):
        """A single farm name is sent unchanged (backward compatible)."""
        mock_get.return_value = _mock_response([_sf("1", "My Farm")])
        mgr = _make_manager()

        mgr.get_seasonfields(farm_name="My Farm")

        params = mock_get.call_args.kwargs["params"]
        assert params["field.farm.name"] == "My Farm"

    @patch("earthdaily.agriculture.services.entity_management.requests.get")
    def test_list_uses_in_operator(self, mock_get):
        """A list of names is sent as a single $in: filter joined on '|'."""
        mock_get.return_value = _mock_response([_sf("1", "Farm A"), _sf("2", "Farm B")])
        mgr = _make_manager()

        mgr.get_seasonfields(farm_name=["Farm A", "Farm B"])

        params = mock_get.call_args.kwargs["params"]
        assert params["field.farm.name"] == "$in:Farm A|Farm B"
        # exactly one request — no client-side fan-out
        assert mock_get.call_count == 1

    @patch("earthdaily.agriculture.services.entity_management.requests.get")
    def test_single_element_list_is_bare_value(self, mock_get):
        """A one-element list degrades to the bare value (no needless $in:)."""
        mock_get.return_value = _mock_response([_sf("1", "Solo Farm")])
        mgr = _make_manager()

        mgr.get_seasonfields(farm_name=["Solo Farm"])

        params = mock_get.call_args.kwargs["params"]
        assert params["field.farm.name"] == "Solo Farm"

    @patch("earthdaily.agriculture.services.entity_management.requests.get")
    def test_list_filters_blank_and_none_entries(self, mock_get):
        """None / whitespace-only entries are dropped before building $in:."""
        mock_get.return_value = _mock_response([_sf("1", "Farm A"), _sf("2", "Farm B")])
        mgr = _make_manager()

        mgr.get_seasonfields(farm_name=["Farm A", None, "  ", "Farm B"])

        params = mock_get.call_args.kwargs["params"]
        assert params["field.farm.name"] == "$in:Farm A|Farm B"

    @patch("earthdaily.agriculture.services.entity_management.requests.get")
    def test_none_omits_farm_filter(self, mock_get):
        """farm_name=None sends no farm filter at all."""
        mock_get.return_value = _mock_response([_sf("1", "Farm A")])
        mgr = _make_manager()

        mgr.get_seasonfields(farm_name=None)

        params = mock_get.call_args.kwargs["params"]
        assert "field.farm.name" not in params

    @patch("earthdaily.agriculture.services.entity_management.requests.get")
    def test_empty_list_omits_farm_filter(self, mock_get):
        """farm_name=[] (and an all-blank list) sends no farm filter."""
        mock_get.return_value = _mock_response([_sf("1", "Farm A")])
        mgr = _make_manager()

        mgr.get_seasonfields(farm_name=[])
        assert "field.farm.name" not in mock_get.call_args.kwargs["params"]

        mgr.get_seasonfields(farm_name=[None, "  "])
        assert "field.farm.name" not in mock_get.call_args.kwargs["params"]


# ===================================================================
# get_seasonfields() — external_ids filter (request side)
# ===================================================================


class TestExternalIdsParam:
    @patch("earthdaily.agriculture.services.entity_management.requests.get")
    def test_single_value_is_bare_value(self, mock_get):
        """A single external id value is sent as a bare externalIds.<system> filter."""
        mock_get.return_value = _mock_response([_sf("1", "Farm A")])
        mgr = _make_manager()

        mgr.get_seasonfields(external_ids={"smbsC_ID": "7073"})

        params = mock_get.call_args.kwargs["params"]
        assert params["externalIds.smbsC_ID"] == "7073"

    @patch("earthdaily.agriculture.services.entity_management.requests.get")
    def test_list_uses_in_operator(self, mock_get):
        """A list of external id values is sent as a single $in: filter joined on '|'."""
        mock_get.return_value = _mock_response([_sf("1", "Farm A"), _sf("2", "Farm B")])
        mgr = _make_manager()

        mgr.get_seasonfields(external_ids={"smbsC_ID": ["7073", "7074"]})

        params = mock_get.call_args.kwargs["params"]
        assert params["externalIds.smbsC_ID"] == "$in:7073|7074"
        assert mock_get.call_count == 1

    @patch("earthdaily.agriculture.services.entity_management.requests.get")
    def test_fully_qualified_key_not_double_prefixed(self, mock_get):
        """A key already prefixed with 'externalIds.' is used verbatim."""
        mock_get.return_value = _mock_response([_sf("1", "Farm A")])
        mgr = _make_manager()

        mgr.get_seasonfields(external_ids={"externalIds.legacY_ID_NA": "ajjwr7v"})

        params = mock_get.call_args.kwargs["params"]
        assert params["externalIds.legacY_ID_NA"] == "ajjwr7v"
        assert "externalIds.externalIds.legacY_ID_NA" not in params

    @patch("earthdaily.agriculture.services.entity_management.requests.get")
    def test_none_omits_external_ids_filter(self, mock_get):
        mock_get.return_value = _mock_response([_sf("1", "Farm A")])
        mgr = _make_manager()

        mgr.get_seasonfields(external_ids=None)

        params = mock_get.call_args.kwargs["params"]
        assert not any(k.startswith("externalIds.") for k in params)


# ===================================================================
# get_seasonfields() — externalIds response processing
# ===================================================================


class TestExternalIdsResponse:
    @patch("earthdaily.agriculture.services.entity_management.requests.get")
    def test_nested_external_ids_is_flattened(self, mock_get):
        """The nested externalIds object is flattened to externalIds.<system> columns."""
        mock_get.return_value = _mock_response(
            [{"externalIds": {"id": "abc", "smbsC_ID": "7073"}, "geometry": "POLYGON((...))"}]
        )
        mgr = _make_manager()

        df = mgr.get_seasonfields(fields="externalIds,geometry")

        assert "externalIds.id" in df.columns
        assert "externalIds.smbsC_ID" in df.columns
        assert df.loc[0, "externalIds.smbsC_ID"] == "7073"

    @patch("earthdaily.agriculture.services.entity_management.requests.get")
    def test_canonical_id_recovered_from_external_ids(self, mock_get):
        """When $fields omits id, the canonical id is recovered from externalIds.id."""
        mock_get.return_value = _mock_response(
            [
                {"externalIds": {"id": "abc", "smbsC_ID": "7073"}, "geometry": "P1"},
                {"externalIds": {"id": "def", "smbsC_ID": "7074"}, "geometry": "P2"},
            ]
        )
        mgr = _make_manager()

        df = mgr.get_seasonfields(fields="externalIds,geometry")

        assert "id" in df.columns
        assert df["id"].tolist() == ["abc", "def"]

    @patch("earthdaily.agriculture.services.entity_management.requests.get")
    def test_variable_external_id_systems_are_not_blocking(self, mock_get):
        """The number/set of externalIds systems is variable per row and per customer.

        Flattening must take the union of whatever keys appear (one column each, NaN
        where a row lacks one) — never assume a fixed set of systems.
        """
        import pandas as pd

        mock_get.return_value = _mock_response(
            [
                # 3 systems
                {"externalIds": {"id": "a", "legacY_ID_NA": "x1", "smbsC_ID": "7073"}},
                # only 1 system
                {"externalIds": {"id": "b"}},
                # a different/extra system not seen on the other rows
                {"externalIds": {"id": "c", "otherErpId": "ZZ"}},
            ]
        )
        mgr = _make_manager()

        df = mgr.get_seasonfields(fields="externalIds")

        # union of all systems present, no row dropped
        assert len(df) == 3
        for col in ["externalIds.id", "externalIds.legacY_ID_NA", "externalIds.smbsC_ID", "externalIds.otherErpId"]:
            assert col in df.columns
        # canonical id still recovered for every row
        assert df["id"].tolist() == ["a", "b", "c"]
        # missing systems are NaN, not an error
        assert pd.isna(df.loc[1, "externalIds.smbsC_ID"])
        assert pd.isna(df.loc[0, "externalIds.otherErpId"])

    @patch("earthdaily.agriculture.services.entity_management.requests.get")
    def test_explicit_id_not_overwritten(self, mock_get):
        """A returned native id takes precedence over externalIds.id recovery."""
        mock_get.return_value = _mock_response([{"id": "native", "externalIds": {"id": "external"}}])
        mgr = _make_manager()

        df = mgr.get_seasonfields(fields="id,externalIds")

        assert df.loc[0, "id"] == "native"


# ===================================================================
# get_seasonfields() — dedup on id
# ===================================================================


class TestDedup:
    @patch("earthdaily.agriculture.services.entity_management.requests.get")
    def test_duplicate_ids_are_dropped(self, mock_get):
        """A seasonfield id returned twice appears once in the result."""
        mock_get.return_value = _mock_response([_sf("1", "Farm A"), _sf("2", "Farm B"), _sf("1", "Farm A")])
        mgr = _make_manager()

        df = mgr.get_seasonfields(farm_name=["Farm A", "Farm B"])

        assert len(df) == 2
        assert sorted(df["id"].tolist()) == ["1", "2"]

    @patch("earthdaily.agriculture.services.entity_management.requests.get")
    def test_no_dedup_when_all_unique(self, mock_get):
        mock_get.return_value = _mock_response([_sf("1", "Farm A"), _sf("2", "Farm B")])
        mgr = _make_manager()

        df = mgr.get_seasonfields(farm_name=["Farm A", "Farm B"])

        assert len(df) == 2


# ===================================================================
# load_seasonfields() — wrapper threads the wider type + logs it
# ===================================================================


class TestLoadSeasonfieldsWrapper:
    def test_list_is_forwarded_to_get_seasonfields(self):
        mgr = _make_manager()
        with patch.object(mgr, "get_seasonfields", return_value=MagicMock(__len__=lambda s: 0)) as gs:
            mgr.load_seasonfields(farm_name=["Farm A", "Farm B"])
        assert gs.call_args.kwargs["farm_name"] == ["Farm A", "Farm B"]

    def test_string_is_forwarded_to_get_seasonfields(self):
        mgr = _make_manager()
        with patch.object(mgr, "get_seasonfields", return_value=MagicMock(__len__=lambda s: 0)) as gs:
            mgr.load_seasonfields(farm_name="My Farm")
        assert gs.call_args.kwargs["farm_name"] == "My Farm"

    def test_filters_log_handles_list(self, capsys):
        """The 'Filters applied' line renders a list as 'farm in [...]'."""
        mgr = _make_manager()
        with patch.object(mgr, "get_seasonfields", return_value=MagicMock(__len__=lambda s: 0)):
            mgr.load_seasonfields(farm_name=["Farm A", "Farm B"])
        out = capsys.readouterr().out
        assert "farm in [Farm A, Farm B]" in out

    def test_filters_log_handles_string(self, capsys):
        mgr = _make_manager()
        with patch.object(mgr, "get_seasonfields", return_value=MagicMock(__len__=lambda s: 0)):
            mgr.load_seasonfields(farm_name="My Farm")
        out = capsys.readouterr().out
        assert "farm = My Farm" in out
