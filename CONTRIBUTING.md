# Contributing

Thanks for your interest in TagAtlas. This is a small, focused library, so
contributions that keep the public API explicit and well tested are welcome.

## Development Setup

TagAtlas uses [uv](https://github.com/astral-sh/uv) for environment management.

```bash
git clone https://github.com/waliwuao/tagatlas.git
cd tagatlas
uv sync --dev
```

Python 3.11 or 3.12 is required because of the GTSAM wheels.

## Before Opening A Pull Request

Run the full local check suite and make sure it passes:

```bash
uv run pytest            # includes the >=90% branch-coverage gate
uv run ruff check .
uv run ruff format --check .
uv run mypy src
uv lock --check
```

CI runs the same checks on Python 3.11 and 3.12, builds the wheel, and imports
it in a clean virtual environment.

## Guidelines

- Keep the public API typed and documented; `mypy` runs in strict mode on
  `src/`.
- Add or update tests for every behavior change. Pure algorithm modules
  (`models`, `camera`, `geometry`, `pose_graph`) are easy to test without the
  native detector.
- Prefer explicit configuration objects over hidden global state.
- Do not add runtime dependencies without a clear justification; the core
  library should stay small.
- Update `CHANGELOG.md` under `[Unreleased]` for user-visible changes.
