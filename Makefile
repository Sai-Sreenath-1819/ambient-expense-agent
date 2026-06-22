# Makefile for managing local ADK development environment

.PHONY: install playground run generate-traces grade

install:
	uv sync --all-extras

playground:
	uv run agents-cli playground

run:
	uv run python -m uvicorn expense_agent.fast_api_app:app --host 0.0.0.0 --port 8080

generate-traces:
	uv run python tests/eval/generate_traces.py

grade:
	/Users/sreenath/.local/share/uv/tools/google-agents-cli/bin/python tests/eval/run_grade.py

