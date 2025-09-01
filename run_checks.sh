#!/bin/bash
set -e

# Import sorting
isort .

# Linting (including docstring style)
ruff check .

# Type checking
mypy .
