.PHONY: docs-html docs-strict docs-clean docs-serve

docs-html:
	python -m sphinx -b html docs/source docs/_build/html

docs-strict:
	python -m sphinx -W --keep-going -b html docs/source docs/_build/html

docs-clean:
	rm -rf docs/_build

docs-serve: docs-html
	python -m http.server --directory docs/_build/html 8000
