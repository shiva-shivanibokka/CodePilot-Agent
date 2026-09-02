"""
The wiring check has to work on a machine with no key.

This exists because it did not. `codepilot doctor` treated a missing
`ANTHROPIC_API_KEY` as a failure, which is correct on a laptop and wrong on
every CI runner — so the one place the wiring check runs unattended was the one
place it could not pass. Only CI caught it, on the first push.
"""

from __future__ import annotations

import pytest

from codepilot.doctor import Absent, Doctor, _api_key, run


async def test_a_missing_key_is_a_skip_not_a_failure(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(Absent):
        await _api_key()


async def test_a_missing_key_is_a_failure_when_live_checks_were_asked_for(monkeypatch):
    """--live cannot do anything without one, so silence would be misleading."""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="--live needs"):
        await _api_key(live=True)


async def test_a_present_key_is_never_printed_in_full(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-" + "S" * 90)
    detail = await _api_key()
    assert "S" * 20 not in detail
    assert "97 chars" in detail


async def test_an_absent_check_does_not_count_against_the_run():
    doctor = Doctor()

    async def missing() -> str:
        raise Absent("not configured here")

    await doctor.check("something optional", missing)
    assert doctor.failures == []


async def test_the_whole_doctor_passes_without_a_key(monkeypatch):
    """The CI invocation, exactly."""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    assert await run() == 0
