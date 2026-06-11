.PHONY: ruff test mypy ci

ruff:
	pixi run -e dev ruff check .

mypy:
	pixi run -e dev mypy .

test:
	pixi run -e dev pytest test/

ci: mypy test ruff