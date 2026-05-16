.PHONY: docs-html docs-strict

docs-html:
	python -m sphinx -b html docs/source docs/_build/html

docs-strict:
	python -m sphinx -W --keep-going -b html docs/source docs/_build/html
