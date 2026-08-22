# lgbm-panel-data

Guide and utilities for using **LightGBM** for panel / time-series forecasting,
especially monthly series with forecast horizons > 12 months.

## Environment

This project uses [`uv`](https://docs.astral.sh/uv/) as the **sole** Python
environment manager. All dependency and environment management goes through `uv`.

```bash
uv sync            # create venv + install all deps
uv run pytest      # run the test suite
uv run ruff check .   # lint
```

## Workflow

See the bundled skill for the full `uv`-only workflow:

- Project: `.github/skills/uv-only/SKILL.md`
- Personal: `~/.copilot/skills/uv-only/SKILL.md`
