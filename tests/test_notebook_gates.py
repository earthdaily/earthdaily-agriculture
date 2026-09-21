"""
Tests for the notebook gate helpers in ``earthdaily.agriculture.notebook_setup``.

These cover the operator-facing contract, which is the whole point of the
helpers: a blocked run must report **every** unmet prerequisite in one pass (so
clearing one gate does not merely uncover the next), must name the literal line
to change, and must not advertise a fix for something that is already fine.
"""

import pytest

from earthdaily.agriculture.notebook_setup import RunBlocked, _install_blocked_renderer, preflight, stop

pytestmark = pytest.mark.public


# ---------------------------------------------------------------------------
# preflight — the all-clear path
# ---------------------------------------------------------------------------


def test_preflight_returns_when_every_check_passes(capsys):
    """All met → no exception, and each check is confirmed on screen."""
    result = preflight(
        "Customer 'ACME' does not exist",
        [
            ("creation enabled", True, "CREATE = True"),
            ("creation confirmed", True, 'CONFIRM = "ACME"'),
        ],
        notes=["a note that should stay hidden"],
    )
    out = capsys.readouterr().out

    assert result is None
    assert "✅ creation enabled" in out
    assert "✅ creation confirmed" in out
    # Nothing is blocked, so no banner, no remedies, no notes.
    assert "⛔" not in out
    assert "CREATE = True" not in out
    assert "a note that should stay hidden" not in out


# ---------------------------------------------------------------------------
# preflight — the blocked path
# ---------------------------------------------------------------------------


def test_preflight_reports_all_unmet_prerequisites_at_once(capsys):
    """Two unmet gates are reported together, not one run at a time."""
    with pytest.raises(RunBlocked) as excinfo:
        preflight(
            "Customer 'ACME' does not exist in the 'preprod' back office",
            [
                ("creation enabled", False, "CREATE_CUSTOMER_IF_MISSING = True"),
                ("creation confirmed", False, 'CONFIRM_CREATE_CUSTOMER = "ACME"'),
                ("writes enabled", False, "DRY_RUN = False"),
                ("back office reachable", True, "connect to the corporate VPN"),
            ],
        )
    out = capsys.readouterr().out

    # Every remedy, in one screenful.
    assert "CREATE_CUSTOMER_IF_MISSING = True" in out
    assert 'CONFIRM_CREATE_CUSTOMER = "ACME"' in out
    assert "DRY_RUN = False" in out

    assert "⛔ Customer 'ACME' does not exist in the 'preprod' back office" in out
    assert "Nothing has been written." in out
    assert "3 of 4 unmet" in out
    # Short summary: some frontends show only this when the traceback is empty.
    assert "3 of 4 prerequisite(s) unmet" in str(excinfo.value)


def test_preflight_does_not_offer_a_fix_for_a_check_that_passed(capsys):
    """A met check is a tick, never an instruction — that would be misleading."""
    with pytest.raises(RunBlocked):
        preflight(
            "blocked",
            [
                ("back office reachable", True, "connect to the corporate VPN"),
                ("writes enabled", False, "DRY_RUN = False"),
            ],
        )
    out = capsys.readouterr().out

    assert "✅ back office reachable" in out
    assert "connect to the corporate VPN" not in out
    assert "❌ writes enabled" in out
    assert "DRY_RUN = False" in out


def test_preflight_prints_notes_only_when_blocked(capsys):
    """Notes carry the plan and the irreversibility warning — needed before deciding."""
    with pytest.raises(RunBlocked):
        preflight(
            "blocked",
            [("creation enabled", False, "CREATE = True")],
            notes=["POST /customers   id=ACME", "", "Only removable from the back office."],
        )
    out = capsys.readouterr().out

    assert "POST /customers   id=ACME" in out
    assert "Only removable from the back office." in out
    # Blank spacer lines carry no trailing indent.
    assert "   \n" not in out


def test_preflight_rerun_instruction_is_customisable(capsys):
    with pytest.raises(RunBlocked):
        preflight(
            "blocked",
            [("creation enabled", False, "CREATE = True")],
            rerun="re-run this cell (the values live in Step 0b)",
        )
    assert "re-run this cell (the values live in Step 0b)." in capsys.readouterr().out


def test_preflight_accepts_truthy_and_falsy_values_not_just_bools(capsys):
    """Callers pass API payloads straight in — ``[]`` means unmet, not a crash."""
    with pytest.raises(RunBlocked):
        preflight(
            "blocked",
            [
                ("roles resolved", ["APP_ACCESS"], "add roles to the profile"),
                ("crops resolved", [], "add crops to the profile"),
            ],
        )
    out = capsys.readouterr().out

    assert "✅ roles resolved" in out
    assert "❌ crops resolved" in out
    assert "1 of 2 unmet" in out


# ---------------------------------------------------------------------------
# stop
# ---------------------------------------------------------------------------


def test_stop_prints_detail_and_raises_with_the_summary(capsys):
    with pytest.raises(RunBlocked, match="fix config/user_profiles.yml"):
        stop("2 invalid profile definition(s) — fix config/user_profiles.yml", "❌ first\n❌ second")
    out = capsys.readouterr().out

    assert "❌ first" in out
    assert "❌ second" in out


def test_stop_without_detail_prints_nothing(capsys):
    with pytest.raises(RunBlocked):
        stop("blocked on something")
    assert capsys.readouterr().out == ""


def test_run_blocked_is_not_a_value_or_runtime_error():
    """Distinct type, so ``except Exception`` handlers around real failures do
    not swallow a deliberate stop — and vice versa."""
    assert issubclass(RunBlocked, Exception)
    assert not issubclass(RunBlocked, (ValueError, RuntimeError))


# ---------------------------------------------------------------------------
# The IPython renderer
# ---------------------------------------------------------------------------


def test_renderer_installation_is_a_no_op_outside_ipython():
    """Under pytest there is no shell; init() must not fail because of it."""
    assert _install_blocked_renderer() is False
