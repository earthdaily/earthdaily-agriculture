"""
Tests for field-file ingestion: geometry normalisation
(``core.geometry.normalize_field_geometry``) and the row-level defaulting in
``EntityManager.prepare_field_dataframe``.

Why this exists: a client sends a shapefile or a CSV of borders and nothing
else. Everything the Farm -> Field -> Seasonfield hierarchy needs beyond the
geometry has to be defaulted or taken from the provisioning run, and the
geometry itself arrives in whatever CRS, validity and part-count the client's
GIS produced. The tests below pin each repair and each refusal, because the
failure mode that matters is the quiet one — a field created in the wrong place,
or a row dropped without a word.
"""

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
from shapely.wkt import loads as load_wkt

from earthdaily.agriculture.core.geometry import (
    close_wkt_rings,
    geodesic_area_ha,
    looks_like_lonlat,
    normalize_field_geometry,
    reproject_geometry,
)
from earthdaily.agriculture.services.entity_management import EntityManager

pytestmark = pytest.mark.public

# A ~64 ha rectangle in Iowa, and the same field in UTM 15N.
FIELD_4326 = "POLYGON((-93.70 42.00, -93.69 42.00, -93.69 42.007, -93.70 42.007, -93.70 42.00))"
FIELD_UTM15N = "POLYGON((442000 4650000, 442800 4650000, 442800 4650800, 442000 4650800, 442000 4650000))"


# ---------------------------------------------------------------------------
# CRS
# ---------------------------------------------------------------------------


class TestCrs:
    def test_clean_wgs84_passes_through_untouched(self):
        out, notes = normalize_field_geometry(FIELD_4326)
        assert notes == []
        assert load_wkt(out).equals(load_wkt(FIELD_4326))

    def test_projected_input_is_reprojected_when_epsg_given(self):
        out, notes = normalize_field_geometry(FIELD_UTM15N, source_epsg=32615)
        centroid = load_wkt(out).centroid
        assert any("reprojected" in n for n in notes)
        assert -94 < centroid.x < -93 and 41 < centroid.y < 43

    def test_projected_input_without_epsg_is_refused_not_guessed(self):
        """Silently accepting eastings would place the field in the Gulf of Guinea."""
        with pytest.raises(ValueError, match="outside WGS84 bounds"):
            normalize_field_geometry(FIELD_UTM15N)

    def test_reproject_is_a_noop_when_source_is_already_target(self):
        geom = load_wkt(FIELD_4326)
        assert reproject_geometry(geom, 4326, 4326) is geom

    def test_looks_like_lonlat(self):
        assert looks_like_lonlat(load_wkt(FIELD_4326))
        assert not looks_like_lonlat(load_wkt(FIELD_UTM15N))

    def test_area_is_geodesic_not_square_degrees(self):
        """shapely's .area over degrees is meaningless as a size."""
        geom = load_wkt(FIELD_4326)
        assert geom.area < 0.001  # square degrees
        assert 60 < geodesic_area_ha(geom) < 70  # hectares


# ---------------------------------------------------------------------------
# Shape repair
# ---------------------------------------------------------------------------


class TestGeometryRepair:
    def test_unclosed_ring_is_closed(self):
        """shapely 2 will not even parse these, so the fix must precede parsing."""
        unclosed = "POLYGON((-93.70 42.00, -93.69 42.00, -93.69 42.007, -93.70 42.007))"
        out, notes = normalize_field_geometry(unclosed)
        assert any("closed unclosed ring" in n for n in notes)
        assert load_wkt(out).equals(load_wkt(FIELD_4326))

    def test_close_wkt_rings_is_a_noop_on_valid_wkt(self):
        assert load_wkt(close_wkt_rings(FIELD_4326)).equals(load_wkt(FIELD_4326))

    def test_self_intersection_is_repaired_keeping_both_lobes(self):
        bowtie = "POLYGON((-93.70 42.00, -93.69 42.01, -93.69 42.00, -93.70 42.01, -93.70 42.00))"
        out, notes = normalize_field_geometry(bowtie)
        assert any("repaired" in n for n in notes)
        assert load_wkt(out).is_valid

    def test_multipolygon_keeps_every_part(self):
        """The platform supports multi-part borders — parts must never be dropped."""
        two_parts = (
            "MULTIPOLYGON("
            "((-93.70 42.00,-93.69 42.00,-93.69 42.01,-93.70 42.01,-93.70 42.00)),"
            "((-93.60 42.00,-93.59 42.00,-93.59 42.01,-93.60 42.01,-93.60 42.00)))"
        )
        out, _ = normalize_field_geometry(two_parts)
        geom = load_wkt(out)
        assert geom.geom_type == "MultiPolygon"
        assert len(geom.geoms) == 2

    def test_single_part_multipolygon_is_unwrapped(self):
        one_part = "MULTIPOLYGON(((-93.70 42.00,-93.69 42.00,-93.69 42.01,-93.70 42.01,-93.70 42.00)))"
        out, _ = normalize_field_geometry(one_part)
        assert load_wkt(out).geom_type == "Polygon"

    @pytest.mark.parametrize(
        "geometry",
        ["POINT(-93.7 42.0)", "LINESTRING(-93.7 42.0, -93.6 42.1)"],
    )
    def test_non_areal_geometry_is_refused(self, geometry):
        with pytest.raises(ValueError, match="must be areal|Expected a Polygon"):
            normalize_field_geometry(geometry)

    def test_unparseable_wkt_raises(self):
        with pytest.raises(ValueError, match="Unparseable WKT"):
            normalize_field_geometry("NOT WKT AT ALL")

    def test_empty_geometry_raises(self):
        with pytest.raises(ValueError, match="empty"):
            normalize_field_geometry("POLYGON EMPTY")


# ---------------------------------------------------------------------------
# Size floor
# ---------------------------------------------------------------------------


class TestMinimumArea:
    SLIVER = "POLYGON((-93.70 42.00, -93.6996 42.00, -93.6996 42.0002, -93.70 42.0002, -93.70 42.00))"

    def test_sliver_is_rejected(self):
        with pytest.raises(ValueError, match="below the 1.0 ha minimum"):
            normalize_field_geometry(self.SLIVER, min_area_ha=1.0)

    def test_real_field_passes(self):
        out, _ = normalize_field_geometry(FIELD_4326, min_area_ha=1.0)
        assert out

    def test_none_disables_the_check(self):
        out, _ = normalize_field_geometry(self.SLIVER, min_area_ha=None)
        assert out

    def test_threshold_is_measured_after_reprojection(self):
        """Square degrees would put every field far below any hectare threshold."""
        out, _ = normalize_field_geometry(FIELD_UTM15N, source_epsg=32615, min_area_ha=50.0)
        assert out


# ---------------------------------------------------------------------------
# prepare_field_dataframe
# ---------------------------------------------------------------------------


@pytest.fixture
def manager():
    with patch.object(EntityManager, "__init__", lambda self, *a, **k: None):
        m = EntityManager.__new__(EntityManager)
    m.get_contextualized_logger = lambda *a: MagicMock()
    return m


def _prepare(manager, rows, **kwargs):
    kwargs.setdefault("grower_id", "kxrx932")
    kwargs.setdefault("farm_name", "Test Farm")
    kwargs.setdefault("default_sowing_date", "2026-05-01")
    kwargs.setdefault("verbose", False)
    return manager.prepare_field_dataframe(pd.DataFrame(rows), **kwargs)


class TestPrepareFieldDataframe:
    def test_geometry_alone_is_enough(self):
        """The whole point: a client file with one column still provisions."""
        with patch.object(EntityManager, "__init__", lambda self, *a, **k: None):
            m = EntityManager.__new__(EntityManager)
        m.get_contextualized_logger = lambda *a: MagicMock()

        ready, rejected = _prepare(m, [{"WKT": FIELD_4326}])
        assert len(ready) == 1 and rejected.empty
        row = ready.iloc[0]
        assert row["Crop"] == "OTHERS"
        assert row["Sowing"] == "2026-05-01"
        assert row["Seasonfield"] == "Field 1"
        assert set(["Grower", "Farm", "Seasonfield", "Sowing", "Crop", "Geometry"]).issubset(ready.columns)

    def test_missing_geometry_column_is_a_hard_error(self, manager):
        with pytest.raises(ValueError, match="No geometry column"):
            _prepare(manager, [{"crop": "CORN", "name": "A"}])

    def test_grower_and_farm_always_come_from_the_run(self, manager):
        """A stale id in a spreadsheet must not file fields under another account."""
        ready, _ = _prepare(
            manager,
            [{"WKT": FIELD_4326, "Grower": "someone_else", "Farm": "Their Farm"}],
            grower_id="kxrx932",
            farm_name="Our Farm",
        )
        assert ready.iloc[0]["Grower"] == "kxrx932"
        assert ready.iloc[0]["Farm"] == "Our Farm"

    def test_explicit_column_mapping_wins_over_aliases(self, manager):
        ready, _ = _prepare(
            manager,
            [{"boundary": FIELD_4326, "label": "North 40"}],
            column_mapping={"label": "Seasonfield"},
        )
        assert ready.iloc[0]["Seasonfield"] == "North 40"

    def test_column_mapping_pointing_at_a_missing_column_raises(self, manager):
        with pytest.raises(ValueError, match="not in the file"):
            _prepare(manager, [{"WKT": FIELD_4326}], column_mapping={"nope": "Crop"})

    def test_aliases_are_detected_case_insensitively(self, manager):
        ready, _ = _prepare(
            manager,
            [{"THE_GEOM": FIELD_4326, "Culture": "CORN", "Parcelle": "P1", "Date_Semis": "2026-04-01"}],
        )
        row = ready.iloc[0]
        assert (row["Crop"], row["Seasonfield"], row["Sowing"]) == ("CORN", "P1", "2026-04-01")

    def test_blank_crop_falls_back_to_the_default(self, manager):
        ready, _ = _prepare(manager, [{"WKT": FIELD_4326, "crop": "  "}], default_crop="OTHERS")
        assert ready.iloc[0]["Crop"] == "OTHERS"

    def test_unknown_crop_falls_back_and_says_so(self, manager):
        """Silently sending an unknown code would 400 at the API instead."""
        ready, _ = _prepare(manager, [{"WKT": FIELD_4326, "crop": "QUINOA"}])
        assert ready.iloc[0]["Crop"] == "OTHERS"
        assert "QUINOA" in ready.iloc[0]["Geometry_notes"]

    def test_known_crop_is_kept_and_uppercased(self, manager):
        ready, _ = _prepare(manager, [{"WKT": FIELD_4326, "crop": "corn"}])
        assert ready.iloc[0]["Crop"] == "CORN"

    def test_no_sowing_column_and_no_default_is_a_hard_error(self, manager):
        with pytest.raises(ValueError, match="no sowing-date column"):
            _prepare(manager, [{"WKT": FIELD_4326}], default_sowing_date=None)

    def test_rejected_rows_are_returned_with_a_reason_not_dropped(self, manager):
        ready, rejected = _prepare(
            manager,
            [
                {"WKT": FIELD_4326, "name": "good"},
                {"WKT": "POINT(-93.7 42.0)", "name": "a point"},
                {"WKT": "", "name": "blank"},
            ],
        )
        assert len(ready) == 1
        assert len(rejected) == 2
        assert set(rejected["name"]) == {"a point", "blank"}
        assert rejected["Reason"].str.len().gt(0).all()

    def test_area_is_reported_per_row(self, manager):
        ready, _ = _prepare(manager, [{"WKT": FIELD_4326}])
        assert 60 < ready.iloc[0]["Area_ha"] < 70

    def test_min_area_filter_is_applied_per_row(self, manager):
        sliver = "POLYGON((-93.70 42.00, -93.6996 42.00, -93.6996 42.0002, -93.70 42.0002, -93.70 42.00))"
        ready, rejected = _prepare(manager, [{"WKT": FIELD_4326}, {"WKT": sliver}], min_area_ha=1.0)
        assert len(ready) == 1
        assert "below the 1.0 ha minimum" in rejected.iloc[0]["Reason"]

    def test_reprojection_applies_to_the_whole_frame(self, manager):
        ready, rejected = _prepare(manager, [{"WKT": FIELD_UTM15N}], source_epsg=32615)
        assert rejected.empty
        assert "reprojected" in ready.iloc[0]["Geometry_notes"]

    def test_generated_names_are_unique_per_row(self, manager):
        ready, _ = _prepare(manager, [{"WKT": FIELD_4326}, {"WKT": FIELD_4326}])
        assert ready["Seasonfield"].tolist() == ["Field 1", "Field 2"]

    def test_output_is_accepted_by_the_creation_contract(self, manager):
        """The six columns create_entities_from_dataframe reads, by name."""
        ready, _ = _prepare(manager, [{"WKT": FIELD_4326}])
        for column in ("Grower", "Farm", "Seasonfield", "Sowing", "Crop", "Geometry"):
            assert column in ready.columns
            assert ready.iloc[0][column] not in ("", None)
