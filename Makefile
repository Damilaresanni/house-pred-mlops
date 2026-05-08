.PHONY:	install	lint test train build all

PYTHON=python3
VENV = .venv
ACTIVATE = source $(VENV)/bin/activate


venv:
	uv venv $(VENV)