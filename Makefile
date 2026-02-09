.PHONY: test lint coverage clean all docs

all: lint test

test:
	python -m pytest tests/ --tb=short -q

lint:
	flake8 src/ tests/

coverage:
	python -m pytest tests/ --cov=ising_toolkit --cov-report=term-missing

docs:
	mkdocs build

docs-serve:
	mkdocs serve

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type d -name .pytest_cache -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	rm -rf .coverage htmlcov/
