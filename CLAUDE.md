# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project status

Study project for RAG and agentic AI, following the IBM certification track. **The repository currently contains no source code** — only `.gitignore` and a local virtual environment. Expect to create files rather than navigate existing ones, and re-read this file once real structure exists (its architecture section is empty by definition today).

Notes and prose are written in French; keep that language for user-facing content unless asked otherwise.

## Environment

Windows + PowerShell. Python 3.12.2 in `.venv/` (git-ignored).

```bash
.\.venv\Scripts\Activate.ps1
```

Activation does not persist between separate tool invocations. In a non-interactive one-shot command, either chain activation with the work in a single call, or invoke the interpreter directly:

```bash
.\.venv\Scripts\python.exe -m pip install <pkg>
```

## Dependencies

Installed so far: `langchain` 1.3.14, `langchain-core` 1.5.2, `langgraph` 1.2.10 (plus transitive deps). These are LangChain **v1** — the `create_agent` / LangGraph-backed API, not the legacy `LLMChain` / `initialize_agent` patterns that dominate older tutorials and pre-v1 training data. When writing LangChain code, check the current API rather than reproducing chain-era idioms.

There is no `requirements.txt` / `pyproject.toml` yet; packages have been added ad hoc with pip. If the project grows past a few scripts, pin them into one of those files.

No test runner, linter, or build tooling is configured — there are no build/lint/test commands to run yet.

## Conventions

`.gitignore` deliberately excludes vector stores, embeddings, model weights, and `data/raw/` + `data/processed/`. Anything indexable or downloadable is meant to be reproducible from code, not committed. If a small fixture corpus genuinely needs to be versioned, loosen the specific rule rather than force-adding files.
