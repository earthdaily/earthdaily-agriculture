"""Allowed values, bulk results and constraints: one source in code, rendered into the AI context.

`core.choices` declares allowed parameter values as `Literal` types and validates from them;
`core.results.BulkResult` names the dict every bulk method returns; `generate_ai_context`
reads both, plus the class docstring's `Constraints:` section, into CLAUDE.md / agents.md.
CoverageExtractor is the pilot (docs/proposals/agent-friendly-docstrings.md, Phase 1).
"""

from __future__ import annotations

from typing import Literal, Optional, get_args

import pytest

from earthdaily.agriculture.core.choices import (
    MergeMode,
    check_choice,
    choice_values,
    is_case_insensitive,
    register_case_insensitive,
)
from earthdaily.agriculture.core.results import (
    CORE_KEYS,
    KEY_DOCS,
    RESULT_TYPES,
    BulkResult,
    SpatialBulkResult,
    result_keys,
)
from earthdaily.agriculture.extractors.coverage_function import (
    CoverageExtractor,
    CoverageFilter,
    CoverageIndex,
    CoverageMask,
)

pytestmark = pytest.mark.public

Colors = Literal["red", "green"]


@pytest.fixture(scope="module")
def gen():
    from earthdaily.agriculture.ai_enablement import generate_ai_context

    return generate_ai_context


class TestCheckChoice:
    def test_valid_value_is_returned(self):
        assert check_choice("filter", "duplicate", CoverageFilter) == "duplicate"

    def test_invalid_value_names_the_parameter_and_the_values(self):
        with pytest.raises(ValueError, match=r"Invalid filter 'bogus'\. Must be one of \['none', 'duplicate'"):
            check_choice("filter", "bogus", CoverageFilter)

    def test_case_sensitive_by_default(self):
        with pytest.raises(ValueError, match="vegetation_index"):
            check_choice("vegetation_index", "ndvi", CoverageIndex)

    def test_registered_alias_returns_the_canonical_casing(self):
        assert check_choice("mask", "mlcirrus", CoverageMask) == "MLCirrus"
        assert check_choice("mask", "Auto", CoverageMask) == "Auto"

    def test_case_insensitive_error_says_so(self):
        with pytest.raises(ValueError, match=r"case-insensitive"):
            check_choice("mask", "cloudless", CoverageMask)

    def test_non_string_is_rejected_not_crashed(self):
        with pytest.raises(ValueError, match="mask"):
            check_choice("mask", 3, CoverageMask)

    def test_register_is_idempotent(self):
        register_case_insensitive(Colors)
        register_case_insensitive(Colors)
        assert is_case_insensitive(Colors)
        assert check_choice("color", "RED", Colors) == "red"


class TestChoiceValues:
    def test_literal(self):
        assert choice_values(MergeMode) == ("auto", "preserve", "mark")

    @pytest.mark.parametrize("annotation", [Optional[MergeMode], MergeMode | None])
    def test_optional_literal_lists_the_literal_values_only(self, annotation):
        assert choice_values(annotation) == ("auto", "preserve", "mark")

    @pytest.mark.parametrize("annotation", [str, int | None, None])
    def test_not_a_literal(self, annotation):
        assert choice_values(annotation) is None


class TestBulkResult:
    def test_key_docs_match_the_typed_dicts(self):
        # KEY_DOCS is what the generated docs show: exactly the keys of all result types.
        assert set(KEY_DOCS) == {k for t in RESULT_TYPES for k in t.__annotations__}

    def test_every_result_type_has_the_core_keys(self):
        for t in RESULT_TYPES:
            assert set(CORE_KEYS) <= set(t.__annotations__), t.__name__

    def test_subtypes_add_only_their_own_keys(self):
        # The card for a non-spatial extractor must not list the spatial-grouping keys.
        assert "spatial_groups" not in BulkResult.__annotations__
        assert "spatial_groups" in SpatialBulkResult.__annotations__

    def test_result_keys_puts_core_first(self):
        keys = result_keys(SpatialBulkResult)
        assert keys[: len(CORE_KEYS)] == list(CORE_KEYS)
        assert keys[-1] == "grouped_from"


class TestCoverageUsesTheLiterals:
    """The validator and the signature read the same Literal — no second list to drift."""

    def test_setup_signature_declares_the_literals(self, gen):
        params = {p["name"]: p for p in gen.get_setup_method(CoverageExtractor)["parameters"]}
        assert params["vegetation_index"]["choices"] == list(get_args(CoverageIndex))
        assert params["filter"]["choices"] == list(get_args(CoverageFilter))
        assert params["mask"]["choices"] == list(get_args(CoverageMask))
        assert params["mask"]["case_insensitive"] is True
        assert params["filter"]["case_insensitive"] is False
        assert params["start_date"]["choices"] is None

    def test_lowercase_mask_is_normalized(self, coverage_extractor):
        coverage_extractor.setup_coverage_parameters(mask="all")
        assert coverage_extractor.coverage_params["mask"] == "All"

    def test_default_mask_is_api_casing(self, coverage_extractor):
        coverage_extractor.setup_coverage_parameters()
        assert coverage_extractor.coverage_params["mask"] == "Auto"

    def test_invalid_merge_existing_lists_the_merge_modes(
        self, configured_coverage_extractor, sample_coverage_entity_list
    ):
        with pytest.raises(
            ValueError, match=r"Invalid merge_existing 'nope'\. Must be one of \['auto', 'preserve', 'mark'\]"
        ):
            configured_coverage_extractor.process_entity_coverage_bulk_parallel(
                entity_list=sample_coverage_entity_list, merge_existing="nope", skip_export=True
            )


class TestRunMethod:
    def test_coverage_entry_point_and_result_type(self, gen):
        run = gen.get_run_method(CoverageExtractor, "setup_coverage_parameters")
        assert run == {
            "method": "process_entity_coverage_bulk_parallel",
            "variants": ["process_entity_coverage_bulk_parallel_recalibration"],
            "returns": "BulkResult",
            "return_keys": result_keys(BulkResult),
        }

    def test_single_bulk_method_is_the_entry_point(self, gen):
        class OneMethod:
            def setup_thing_parameters(self): ...
            def process_thing_bulk_extraction_parallel(self): ...

        run = gen.get_run_method(OneMethod, "setup_thing_parameters")
        assert run["method"] == "process_thing_bulk_extraction_parallel"
        assert run["variants"] == [] and run["returns"] is None

    def test_steps_have_no_single_entry_point(self, gen):
        # Shaped like Tillage: two bulk methods, neither is process_entity_<type>_bulk_parallel.
        class TwoSteps:
            def setup_steps_parameters(self): ...
            def process_steps_batch_bulk_extraction_parallel(self): ...
            def process_steps_result_bulk_extraction_parallel(self): ...

        run = gen.get_run_method(TwoSteps, "setup_steps_parameters")
        assert run["method"] is None
        assert len(run["variants"]) == 2

    def test_no_bulk_method(self, gen):
        class NoBulk:
            def setup_x_parameters(self): ...

        assert gen.get_run_method(NoBulk, "setup_x_parameters") is None


class TestConstraintsSection:
    def test_bullets_and_continuations_are_parsed(self, gen):
        doc = (
            "Thing.\n\n"
            "Output columns:\n    a, b\n\n"
            "Constraints:\n"
            "    - x requires y.\n"
            "    - z cannot be combined\n"
            "      with w.\n"
        )
        assert gen.parse_docstring(doc)["constraints"] == ["x requires y.", "z cannot be combined with w."]

    def test_coverage_constraints_reach_the_card(self, gen):
        entry = gen.introspect_class(
            "earthdaily.agriculture.extractors.coverage_function", "CoverageExtractor", "Foundational"
        )
        assert len(entry["constraints"]) == 3
        assert any("mask='All'" in c and "filter='duplicate'" in c for c in entry["constraints"])
        card = gen.generate_claude_md([entry])
        assert "- **Run:** `process_entity_coverage_bulk_parallel(entity_list, ...)` → `BulkResult`" in card
        assert "one of: Auto | Native | ACM | ML | MLCirrus | All, case-insensitive" in card
        assert "- **Constraints:** filter='crop_coverage' requires historical_seasons." in card

    def test_agents_card_lists_allowed_values_and_result_keys(self, gen):
        entry = gen.introspect_class(
            "earthdaily.agriculture.extractors.coverage_function", "CoverageExtractor", "Foundational"
        )
        card = gen.generate_agents_md([entry])
        assert "| Parameter | Default | Allowed values |" in card
        assert "| `filter` | 'none' | `none`, `duplicate`, `crop_coverage` |" in card
        assert "`All` (case-insensitive) |" in card
        # Every table row keeps exactly 3 cells: no stray pipe from a value list.
        setup_rows = [line for line in card.splitlines() if line.startswith("| `")]
        assert setup_rows and all(line.count("|") == 4 for line in setup_rows)
        assert "Returns `BulkResult`" in card
        for key in CORE_KEYS:
            assert f"- `{key}` — " in card
        assert "### Constraints" in card


class TestGddOffsetSetup:
    """GDDOffset had no tests; these cover the rules its card now documents."""

    @pytest.fixture
    def ext(self):
        from earthdaily.agriculture.extractors.gdd_offset_functions import GDDOffsetExtractor

        ext = GDDOffsetExtractor.__new__(GDDOffsetExtractor)  # __init__ needs credentials
        ext.ensure_token_valid = lambda: None  # setup is @requires_token; no token here
        return ext

    def test_lower_threshold_must_be_below_upper(self, ext):
        with pytest.raises(ValueError, match="lower_threshold"):
            ext.setup_gdd_offset_parameters(lower_threshold=30, upper_threshold=10)

    def test_provider_is_validated_from_its_literal(self, ext):
        with pytest.raises(ValueError, match=r"Invalid provider 'NOPE'\. Must be one of \['GLOBAL1'\]"):
            ext.setup_gdd_offset_parameters(provider="NOPE")

    def test_constraint_reaches_the_card(self, gen):
        entry = gen.introspect_class(
            "earthdaily.agriculture.extractors.gdd_offset_functions", "GDDOffsetExtractor", "Crop Development"
        )
        assert entry["constraints"] == ["lower_threshold must be below upper_threshold."]
        assert entry["run_method"]["returns"] == "SpatialBulkResult"


class TestListParameters:
    """List parameters (weather_parameters, sensors) take Literal items, checked one by one."""

    def test_choice_values_reads_lists_and_unions(self):
        from earthdaily.agriculture.extractors.weather_functions import WeatherParameter

        values = choice_values(WeatherParameter | list[WeatherParameter] | Literal["none"])
        assert values[: len(get_args(WeatherParameter))] == get_args(WeatherParameter)
        assert values[-1] == "none"
        assert choice_values(list[MergeMode]) == ("auto", "preserve", "mark")

    def test_check_choices_names_every_bad_item(self):
        from earthdaily.agriculture.core.choices import check_choices

        assert check_choices("modes", ["auto", "mark"], MergeMode) == ["auto", "mark"]
        with pytest.raises(ValueError, match=r"Invalid modes \['x', 'y'\]\. Each must be one of"):
            check_choices("modes", ["auto", "x", "y"], MergeMode)

    def test_case_insensitive_list_items_come_back_canonical(self):
        from earthdaily.agriculture.core.choices import check_choices
        from earthdaily.agriculture.extractors.coverage_function import CoverageSensor

        assert check_choices("sensors", ["sentinel_2", "Landsat_9"], CoverageSensor) == ["SENTINEL_2", "LANDSAT_9"]

    def test_coverage_sensors_still_normalized_and_deduplicated(self, coverage_extractor):
        coverage_extractor.setup_coverage_parameters(sensors=[" sentinel_2", "SENTINEL_2", "landsat_9"])
        assert coverage_extractor.coverage_params["sensors"] == ["LANDSAT_9", "SENTINEL_2"]

    def test_weather_card_lists_the_parameters(self, gen):
        entry = gen.introspect_class(
            "earthdaily.agriculture.extractors.weather_functions", "WeatherExtractor", "Foundational"
        )
        params = {p["name"]: p for p in entry["setup_method"]["parameters"]}
        assert "precipitation.cumulative" in params["weather_parameters"]["choices"]
        assert "none" in params["weather_parameters"]["choices"]
        assert params["weather_type"]["choices"] == ["HISTORICAL_DAILY", "FORECAST_DAILY", "FORECAST_HOURLY"]


class TestMrtsIndices:
    @pytest.mark.parametrize("index", ["CVIN", "BI", "BIOMASS"])
    def test_added_indices_are_accepted(self, index):
        # Added 2026-10-06: CVIN and BI returned data live; BIOMASS is zone-limited (confirmed).
        from earthdaily.agriculture.extractors.VTS_functions import MrtsVegetationIndex

        assert check_choice("vegetation_index", index, MrtsVegetationIndex) == index

    @pytest.mark.parametrize("index", ["CAB", "RVI"])
    def test_unverified_api_indices_are_still_rejected(self, index):
        from earthdaily.agriculture.extractors.VTS_functions import MrtsVegetationIndex

        with pytest.raises(ValueError, match="vegetation_index"):
            check_choice("vegetation_index", index, MrtsVegetationIndex)
