## Python environment

Use uv for this project's Python commands:

- Run a script: `uv run python script.py`
- Run tests: `uv run pytest`
- Run a Python module: `uv run python -m module_name`
- Add dependencies: `uv add <package>`

Do not activate .venv between tool calls. uv selects it for each command.