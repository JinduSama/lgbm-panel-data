---
name: uv-only
description: "Work exclusively with uv for all Python tasks in this project. Use when: creating or editing pyproject.toml, installing packages, creating venvs, running scripts/tests/linters, or any Python dev task. uv is the single source of truth for the environment. Do NOT use pip, conda, venv, or virtualenv directly."
argument-hint: "Manage Python with uv only"
user-invocable: true
disable-model-invocation: false
---

# uv-Only Python Workflow

This project uses [`uv`](https://github.com/astral-sh/uv) as the **sole** tool for
managing the Python environment. `uv` is an extremely fast package + environment
manager (Rust-based) that replaces `pip`, `venv`, `conda`, and `virtualenv`.

## Golden Rules

1. **Never** use `pip install`, `python -m venv`, `conda`, or `virtualenv` directly.
   All dependency and environment management goes through `uv`.
2. Dependencies are declared in `pyproject.toml` (PEP 621) and resolved by `uv`.
3. The active environment is created and managed by `uv` automatically.
4. Prefer `uv run` over calling the interpreter directly — it uses the project's
   resolved environment and auto-installs missing dependencies.

## Common Commands

| Task | Command |
|------|---------|
| Run a script | `uv run <script>.py` |
| Run a command in the project env | `uv run <command>` |
| Install a dependency | `uv add <package>` |
| Install a dev dependency | `uv add --dev <package>` |
| Add a dependency group | `uv add --group <group> <package>` |
| Sync / install all deps | `uv sync` |
| Run tests | `uv run pytest` |
| Lint | `uv run ruff check .` |
| Format | `uv run ruff format .` |
| Show the environment | `uv pip list` |
| Show resolved lockfile | `uv lock` |

## Environment Setup

The first time you work in this project, ensure the environment is ready:

```bash
uv sync            # create venv + install all deps from pyproject.toml
```

If `uv` is not on your PATH, install it:

```bash
uv --version
# If missing:
curl -LsSf https://astral.sh/uv/install.sh | sh
```

## Conventions

- Declare runtime deps directly under `[project] dependencies`.
- Declare dev/test deps under `[dependency-groups] dev` (or `--group dev`).
- Keep `pyproject.toml` as the single source of truth; do not maintain a separate
  `requirements.txt`.
- When the model needs to inspect the environment, use `uv pip list` rather than
  probing the filesystem for a `.venv`.

## References

- [uv command reference](https://docs.astral.sh/uv/)
- [PEP 621 — pyproject.toml metadata](https://peps.python.org/pep-0621/)
