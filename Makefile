.PHONY: install dev-install test clean

install:
	pip install -e .

dev-install:
	pip install -e ".[dev]"

test:
	pytest

clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
