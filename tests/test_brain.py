"""M5 — brain tool-calling loop, with LiteLLM mocked (no API key needed)."""

from types import SimpleNamespace

import pytest

from brain.llm import Brain


def _msg(content=None, tool_calls=None):
    return SimpleNamespace(content=content, tool_calls=tool_calls)


def _resp(message):
    return SimpleNamespace(choices=[SimpleNamespace(message=message)])


def _toolcall(call_id, name, arguments):
    return SimpleNamespace(id=call_id, function=SimpleNamespace(name=name, arguments=arguments))


def test_chat_runs_tool_then_answers(monkeypatch):
    """Model asks for get_time, then produces a spoken answer using the result."""
    calls = {"n": 0}
    seen_tool_result = {}

    def fake_completion(self, messages):
        calls["n"] += 1
        if calls["n"] == 1:
            return _resp(_msg(tool_calls=[_toolcall("c1", "get_time", "{}")]))
        # Second call: the tool result must be present in the message history.
        tool_msgs = [m for m in messages if m.get("role") == "tool"]
        assert tool_msgs, "tool result not fed back to the model"
        seen_tool_result["content"] = tool_msgs[-1]["content"]
        return _resp(_msg(content="Je sobota, devět hodin."))

    monkeypatch.setattr(Brain, "_completion", fake_completion)

    rep = Brain().chat("Kolik je hodin?")
    assert rep.reply == "Je sobota, devět hodin."
    assert len(rep.tool_calls) == 1
    assert rep.tool_calls[0]["name"] == "get_time"
    assert "iso" in seen_tool_result["content"]  # get_time result serialized


def test_chat_no_tools_direct_answer(monkeypatch):
    monkeypatch.setattr(Brain, "_completion",
                        lambda self, messages: _resp(_msg(content="Ahoj Marty!")))
    rep = Brain().chat("Ahoj")
    assert rep.reply == "Ahoj Marty!"
    assert rep.tool_calls == []


def test_chat_loop_cap(monkeypatch):
    """A model that always calls a tool must not loop forever."""
    def always_tool(self, messages):
        return _resp(_msg(tool_calls=[_toolcall("c", "get_time", "{}")]))

    monkeypatch.setattr(Brain, "_completion", always_tool)
    rep = Brain(max_tool_iters=3).chat("test")
    assert rep.reply  # returns a fallback message, doesn't hang
    assert len(rep.tool_calls) >= 3
