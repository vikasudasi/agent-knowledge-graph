# Contributing

## Development Setup

```bash
git clone https://github.com/vikasudasi/agent-knowledge-graph.git
cd agent-knowledge-graph
uv sync --extra dev
```

## Branching and Commits

- Create focused branches per change.
- Keep commits atomic and descriptive.
- Include tests with behavior changes.

Suggested commit style:

- `feat(cli): add query explain formatting`
- `test(adapters): cover mcp handler failure paths`

## Quality Gates

Run before opening a PR:

```bash
uv run python -m pytest
uv run python -m pytest --cov --cov-report=term-skip-covered
uv run ruff check .
uv run ruff format --check .
uv run mypy core cli pipelines adapters
```

## Pull Request Checklist

- Problem and scope described clearly
- Tests added or updated
- No unrelated refactors mixed in
- Docs updated when command and config behavior changes
- CI green

## Reporting Issues

When filing a bug, include:

- command used
- observed output and expected output
- Python version
- relevant `KG_*` configuration (without secrets)
