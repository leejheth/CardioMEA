.ONESHELL:
.PHONY: setup

VENV := venv/bin

setup:
	rm -rf venv
	python3.9 -m venv venv
	$(VENV)/pip install -r src/requirements.txt

setup_dev:
	rm -rf venv
	python3.9 -m venv venv
	$(VENV)/pip install -r src/requirements.in

freeze:
	$(VENV)/pip freeze > src/requirements.txt