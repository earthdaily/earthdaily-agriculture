"""
Specific-dates extraction is deprecated (removed in 3.0).

Every entry point must warn — as a DeprecationWarning for code and test runners, and
in the log, because Python hides DeprecationWarning outside __main__ and a notebook or a
scheduled run would otherwise never see it. The modes that stay must not warn.
"""

import warnings
from unittest.mock import patch

import pandas as pd
import pytest

from earthdaily.agriculture.services.workflow_manager import WorkflowManager

pytestmark = pytest.mark.public

DATES = ["2025-06-01", "2025-07-01"]


def _specific_dates_warnings(record):
    return [w for w in record if issubclass(w.category, DeprecationWarning) and "specific_dates" in str(w.message)]


class TestVegetationTs:
    def test_setup_with_specific_dates_warns(self, vts_extractor):
        with pytest.warns(DeprecationWarning, match="removed in 3.0"):
            vts_extractor.setup_vegetation_ts_parameters(extraction_mode="specific_dates", target_dates=DATES)
        # still works until 3.0
        assert vts_extractor.vegetation_ts_params["target_dates"] == DATES

    @pytest.mark.parametrize("mode", ["period", "windows"])
    def test_the_modes_that_stay_do_not_warn(self, vts_extractor, mode):
        with warnings.catch_warnings(record=True) as record:
            warnings.simplefilter("always")
            vts_extractor.setup_vegetation_ts_parameters(
                extraction_mode=mode, start_date="2025-05-01", end_date="2025-09-30"
            )
        assert not _specific_dates_warnings(record)

    def test_bulk_method_warns(self, vts_extractor):
        vts_extractor.setup_vegetation_ts_parameters(
            extraction_mode="windows", start_date="2025-05-01", end_date="2025-09-30"
        )
        with pytest.warns(DeprecationWarning, match="process_specific_dates_bulk_parallel"):
            with pytest.raises(ValueError, match="No target_dates"):
                vts_extractor.process_specific_dates_bulk_parallel(pd.DataFrame({"id": ["a"]}), skip_export=True)

    def test_single_entity_method_warns(self, vts_extractor):
        vts_extractor.setup_vegetation_ts_parameters(extraction_mode="specific_dates", target_dates=DATES)
        with pytest.warns(DeprecationWarning, match="process_single_entity_specific_dates"):
            with patch.object(vts_extractor, "get_vegetation_api", return_value={"value": []}):
                vts_extractor.process_single_entity_specific_dates({"id": "a"})

    def test_the_warning_names_the_replacement(self, vts_extractor):
        with pytest.warns(DeprecationWarning) as record:
            vts_extractor.setup_vegetation_ts_parameters(extraction_mode="specific_dates", target_dates=DATES)
        assert "extraction_mode='windows'" in str(record[0].message)


class TestWorkflowManager:
    @pytest.fixture
    def manager(self):
        with patch.object(WorkflowManager, "__init__", lambda self, *a, **kw: None):
            wm = WorkflowManager()
        wm.target_dates = None
        return wm

    def test_load_target_dates_warns(self, manager, tmp_path):
        csv = tmp_path / "dates.csv"
        pd.DataFrame({"target_date": DATES}).to_csv(csv, index=False)
        with pytest.warns(DeprecationWarning, match="load_target_dates"):
            manager.load_target_dates(str(csv))

    def test_a_target_dates_column_warns(self, manager):
        manager.sfd_list = pd.DataFrame({"id": ["a"], "target_dates": ["2025-06-01|2025-07-01"]})
        with pytest.warns(DeprecationWarning, match="target_dates' column"):
            manager._parse_target_dates_from_column()
        assert manager.target_dates == DATES
