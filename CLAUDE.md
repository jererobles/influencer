# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build/Run Commands
- Run application: `python mux.py`
- Install dependencies: use `uv` package manager, project requirements live in pyproject.toml

## Code Style Guidelines
- Follow PEP 8 for Python code style
- Use type hints for all function parameters and return values
- Import organization: standard library, then third-party, then local
- Use dataclasses for structured data
- Use async/await for asynchronous code
- Error handling: use try/except blocks with specific exceptions
- Logging: use the provided logger (log) for all messages
- Variables: use snake_case for variable names
- Classes: use PascalCase for class names
- Functions: use snake_case for function names
- Constants: use UPPER_CASE for constants
- Object-oriented: prefer composition over inheritance

## Documentation
- Add docstrings to all functions and classes
- Use inline comments for complex logic