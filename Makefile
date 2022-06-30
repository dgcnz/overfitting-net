init-venv:
	python -m pip install --upgrade pip setuptools wheel
	python -m pip install --upgrade pip-tools
	make pipcompile-dev
	make pipsync-dev

# For updating new dependencies

pipcompile-dev:
	pip-compile pip-files/requirements-dev.in --output-file=pip-files/requirements-dev.txt

pipcompile-test:
	pip-compile pip-files/requirements-test.in --output-file=pip-files/requirements-test.txt

pipcompile-prod:
	pip-compile pip-files/requirements-prod.in --output-file=pip-files/requirements-prod.txt


# For synchronizing the dependencies with the current virtual environment

pipsync-dev:
	pip-sync pip-files/requirements-dev.txt

pipsync-test:
	pip-sync pip-files/requirements-test.txt

pipsync-prod:
	pip-sync pip-files/requirements-prod.txt
