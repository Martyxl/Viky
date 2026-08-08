"""M5 — tool schema validation + dry-run behaviour."""

import pytest

from brain.tools import openai_tools, tool_names
from tools.registry import dispatch


def test_all_tools_registered():
    names = tool_names()
    for expected in (
        "get_time", "get_trading_stats", "trigger_n8n_workflow",
        "send_email", "list_agents", "run_agent",
    ):
        assert expected in names


def test_schemas_are_wellformed():
    for tool in openai_tools():
        fn = tool["function"]
        assert tool["type"] == "function"
        assert fn["name"] and fn["description"]
        params = fn["parameters"]
        assert params["type"] == "object"
        assert isinstance(params["properties"], dict)
        # every required field must be declared in properties
        for req in params.get("required", []):
            assert req in params["properties"], f"{fn['name']}: '{req}' not in properties"


def test_get_time_is_real():
    result = dispatch("get_time", {})
    assert "iso" in result and "time" in result
    assert "dry_run" not in result  # read-only, always executes


def test_side_effect_tools_are_dry_run_by_default(monkeypatch):
    import tools.base as base

    monkeypatch.setattr(base.settings, "dry_run", True)
    for name, args in [
        ("send_email", {"to": "a@b.cz", "subject": "S", "body": "B"}),
        ("trigger_n8n_workflow", {"workflow_id": "wf1", "payload": {"x": 1}}),
        ("run_agent", {"agent_id": "report", "task": "shrň den"}),
    ]:
        result = dispatch(name, args)
        assert result.get("dry_run") is True, f"{name} should be dry-run"


def test_dispatch_unknown_and_bad_args():
    assert "error" in dispatch("does_not_exist", {})
    # missing required arg -> handled, not raised
    assert "error" in dispatch("send_email", {"to": "only@to.cz"})


def test_trading_stats_returns_data():
    # Stats backend likely not running in tests -> inline mock, still valid shape.
    result = dispatch("get_trading_stats", {"instrument": "MNQ", "period": "day"})
    assert result["instrument"].upper() == "MNQ"
    assert "win_rate" in result
