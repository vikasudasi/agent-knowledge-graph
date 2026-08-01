# agent-knowledge-graph — Repo Scaffolding & CI

## Project Overview

Build the foundational repository structure for `agent-knowledge-graph` — an open-source CLI tool + Python library that gives AI agents persistent structured memory via a Neo4j knowledge graph with vector (semantic) search.

## Tech Stack

- Python 3.11+
- `uv` for dependency management (no pip/poetry)
- `typer` for CLI (with `rich` for console output)
- `neo4j` driver for graph DB
- `sentence-transformers` for local embeddings
- `httpx` for LLM API calls (OpenRouter)
- `pydantic` for config validation
- `pytest` + `pytest-cov` + `pytest-asyncio` for testing
- `ruff` for linting, `mypy` for type checking
- GitHub Actions for CI

## File Structure to Create

```
/root/agent-knowledge-graph/
├── .gitignore
├── .pre-commit-config.yaml
├── LICENSE                    # MIT
├── README.md                  # 14-section skeleton (fill with real content)
├── pyproject.toml             # Build config, deps, entry points
├── core/
│   ├── __init__.py
│   ├── config.py              # placeholder
│   ├── graph.py               # placeholder (Neo4j client)
│   ├── llm.py                 # placeholder (LLM abstraction)
│   ├── embedding.py           # placeholder (embedding provider)
│   └── models.py              # placeholder (Pydantic models)
├── pipelines/
│   ├── __init__.py
│   ├── base.py               # placeholder (pipeline framework)
│   └── session.py            # placeholder (session-ingest pipeline)
├── adapters/
│   ├── __init__.py
│   ├── hermes/
│   │   ├── __init__.py
│   │   └── plugin.py         # placeholder
│   ├── mcp/
│   │   ├── __init__.py
│   │   └── server.py         # placeholder
│   └── langchain/
│       ├── __init__.py
│       └── tool.py           # placeholder
├── cli/
│   ├── __init__.py
│   ├── main.py               # Typer app entry point
│   ├── init.py               # kg init command (placeholder)
│   ├── build.py              # kg build command (placeholder)
│   ├── query.py              # kg query command (placeholder)
│   └── status.py             # kg status command (placeholder)
├── docker/
│   └── docker-compose.yml    # placeholder
├── tests/
│   ├── __init__.py
│   ├── conftest.py
│   ├── test_cli.py
│   └── test_config.py        # placeholder
├── docs/
│   ├── ARCHITECTURE.md       # placeholder
│   ├── PIPELINES.md          # placeholder
│   └── AGENTS.md             # placeholder
└── .github/
    ├── dependabot.yml
    └── workflows/
        ├── test.yml          # pytest + coverage
        ├── lint.yml          # ruff
        ├── typecheck.yml     # mypy
        └── build.yml         # wheel build
```

## README Requirements

The README.md must be comprehensive and professional:
- Title + one-liner value proposition ("Persistent knowledge graph memory for AI agents")
- Badges: CI status, coverage, Python version, license
- "Why" section explaining the problem (agents have no persistent structured memory)
- Installation instructions (pip install, source install, dev install)
- Quick start: 5 steps from install to first query
- Detailed usage examples with real commands and output
- CLI options reference table
- Architecture overview (layer diagram in Mermaid)
- Configuration reference
- Pipeline system explanation
- Agent adapters overview (Hermes, MCP, LangChain)
- Contributing guide
- FAQ (Docker required? Do I need an API key?)
- License section (MIT)
- Target: 200+ lines, 14+ sections, no placeholder content

## pyproject.toml Requirements

```toml
[build-system]
requires = ["setuptools", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "agent-knowledge-graph"
version = "0.1.0"
description = "Persistent knowledge graph memory for AI agents — Neo4j + vector search + NL→Cypher"
requires-python = ">=3.11"
dependencies = [
    "typer>=0.12",
    "rich>=13.0",
    "neo4j>=5.20",
    "httpx>=0.27",
    "pydantic>=2.0",
    "pydantic-settings>=2.0",
    "pyyaml>=6.0",
    "sentence-transformers>=3.0",
    "pygments>=2.17",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0",
    "pytest-cov>=5.0",
    "pytest-asyncio>=0.24",
    "ruff>=0.5",
    "mypy>=1.10",
    "pre-commit>=3.0",
    "uv>=0.4",
]
hermes = [
    "hermes-agent>=1.0",
]
mcp = [
    "mcp>=1.0",
]

[project.scripts]
kg = "cli.main:app"

[tool.ruff]
line-length = 100
target-version = "py311"
[tool.ruff.lint]
select = ["E", "F", "I", "N", "W", "UP"]
[tool.ruff.format]
quote-style = "double"

[tool.mypy]
python_version = "3.11"
strict = true
ignore_missing_imports = true
warn_unused_ignores = true

[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-v --cov=core --cov=cli --cov=pipelines --cov-report=term-missing"
asyncio_mode = "auto"
```

## .gitignore Requirements

Must include patterns for:
- Python: `__pycache__/`, `*.pyc`, `*.pyo`, `*.egg-info/`, `dist/`, `build/`
- Virtual envs: `.venv/`, `venv/`, `.env`
- Neo4j: `data/`, `logs/`, `import/` (Docker volumes)
- IDE: `.vscode/`, `.idea/`, `*.swp`
- OS: `.DS_Store`, `Thumbs.db`
- Coverage: `.coverage`, `coverage/`, `htmlcov/`
- Cache: `.ruff_cache/`, `.mypy_cache/`, `.pytest_cache/`

## GitHub Actions Workflows

### test.yml (on push/PR to main)
- Setup: checkout, install uv, python 3.11, `uv sync`
- Test: `uv run pytest --cov --cov-report=term-missing`
- Matrix: python 3.11, 3.12

### lint.yml (on push/PR)
- Setup: checkout, install uv, python 3.11
- Run: `uv run ruff check .`
- Run: `uv run ruff format --check .`

### typecheck.yml (on push/PR)
- Setup: checkout, install uv, python 3.11, `uv sync`
- Run: `uv run mypy core/ cli/ pipelines/`

### build.yml (on tag push v*)
- Setup: checkout, install uv, python 3.11
- Build: `uv build`
- Upload: store wheel as artifact

### dependabot.yml
- `pip` ecosystem, weekly checks
- Assignee: vikasudasi

## Pre-commit Config

```yaml
repos:
  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.5.0
    hooks:
      - id: ruff
        args: [--fix]
      - id: ruff-format
  - repo: https://github.com/pre-commit/mirrors-mypy
    rev: v1.10.0
    hooks:
      - id: mypy
        additional_dependencies: [pydantic]
  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v4.6.0
    hooks:
      - id: trailing-whitespace
      - id: end-of-file-fixer
      - id: check-yaml
      - id: check-added-large-files
```

## CLI Entry Point (kg command)

The `cli/main.py` must create a Typer app with placeholder subcommands:

```python
import typer

app = typer.Typer(
    name="kg",
    help="agent-knowledge-graph — Persistent knowledge for AI agents",
    rich_markup_mode="rich",
)

@app.command()
def init(
    reset: bool = typer.Option(False, "--reset", help="Reset Neo4j and re-initialize"),
):
    """Initialize Neo4j and create schema."""
    typer.echo("[kg] init not yet implemented")

@app.command()
def build(
    pipeline: str = typer.Argument(..., help="Pipeline name (sessions, files)"),
    full: bool = typer.Option(False, "--full", help="Full rebuild from scratch"),
    limit: int = typer.Option(None, "--limit", help="Max items to process"),
):
    """Run an ingestion pipeline."""
    typer.echo(f"[kg] build {pipeline} not yet implemented")

@app.command()
def query(
    question: str = typer.Argument(..., help="Natural language question"),
    cypher: bool = typer.Option(False, "--cypher", help="Raw Cypher mode"),
    json_output: bool = typer.Option(False, "--json", help="JSON output format"),
):
    """Query the knowledge graph."""
    typer.echo(f"[kg] query not yet implemented")

@app.command()
def status():
    """Show knowledge graph stats and health."""
    typer.echo("[kg] status not yet implemented")

if __name__ == "__main__":
    app()
```

## Placeholder Files

All placeholder Python files (`core/*.py`, `pipelines/*.py`, `adapters/*/*.py`, `tests/*.py`) must:
- Have proper `__init__.py` files where needed
- Have a module docstring
- Have a class/function stub that raises `NotImplementedError`
- Use `from __future__ import annotations`
- Have proper type hints

Example placeholder:
```python
"""Configuration management for agent-knowledge-graph."""

from __future__ import annotations


class ConfigManager:
    """Loads and manages configuration from XDG config paths."""

    def __init__(self) -> None:
        raise NotImplementedError("ConfigManager not yet implemented")
```

## Docker Placeholder

`docker/docker-compose.yml` must be a valid Docker Compose file for Neo4j 5.x:

```yaml
version: "3.8"
services:
  neo4j:
    image: neo4j:5-community
    container_name: kg-neo4j
    ports:
      - "7687:7687"  # Bolt
      - "7474:7474"  # HTTP
    environment:
      NEO4J_AUTH: neo4j/password
      NEO4J_PLUGINS: '["apoc"]'
      NEO4J_dbms_memory_pagecache_size: 2G
      NEO4J_dbms_memory_heap_initial__size: 2G
      NEO4J_dbms_memory_heap_max__size: 4G
    volumes:
      - neo4j_data:/data
      - neo4j_logs:/logs
    healthcheck:
      test: ["CMD", "cypher-shell", "-u", "neo4j", "-p", "password", "RETURN 1"]
      interval: 30s
      timeout: 10s
      retries: 5
volumes:
  neo4j_data:
  neo4j_logs:
```

## Instructions for Cursor

1. Create ALL files listed in the file structure above
2. Every placeholder module must be importable without errors
3. Run `uv sync` at the end to verify dependencies resolve
4. Run `uv run python -c "from cli.main import app; print('CLI OK')"` to verify CLI import
5. Run `uv run kg --help` to verify CLI entry point works
6. Do NOT skip the README — write it properly with real content, not placeholders
7. After completion, the repo must be in a state where `uv run pytest` runs (even if tests are trivial/skipped)