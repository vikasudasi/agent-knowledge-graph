"""CLI smoke tests for the Typer application."""

from __future__ import annotations

from typer.testing import CliRunner

from cli.main import app


runner = CliRunner()


def test_help_command_displays_usage() -> None:
    """Ensure the top-level help command runs successfully."""
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "Persistent knowledge for AI agents" in result.stdout


def not_implemented_cli_snapshot() -> None:
    """Placeholder for golden-output testing."""
    raise NotImplementedError("not_implemented_cli_snapshot is not yet implemented")
