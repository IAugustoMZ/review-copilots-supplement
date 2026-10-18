"""Phase 1 smoke test: the package imports and the CLI is wired up.

Later phases add real unit tests per module (see docs/TECHNICAL_IMPLEMENTATION_PLAN.md
section 24); this file is intentionally minimal until there is real logic to test.
"""

from typer.testing import CliRunner

from review_pipeline.src.cli.app import app


def test_cli_status_runs() -> None:
    runner = CliRunner()
    result = runner.invoke(app, ["status"])
    assert result.exit_code == 0
    assert "review-cli" in result.stdout
