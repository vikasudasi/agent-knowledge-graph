"""Pytest fixtures and shared test configuration."""

from __future__ import annotations

import pytest


@pytest.fixture()
def placeholder_fixture_name() -> str:
    """Return a static fixture to validate test wiring."""
    return "agent-knowledge-graph"


def not_implemented_fixture_strategy() -> None:
    """Placeholder for future fixture orchestration."""
    raise NotImplementedError("not_implemented_fixture_strategy is not yet implemented")
