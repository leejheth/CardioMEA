.ONESHELL:
.PHONY: setup

VENV := venv/bin

################################
# Environment setup
################################
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

################################
# Data processing
################################
create_list:
	$(VENV)/kedro run -p list_rec_files

extracellular:
	$(VENV)/python scripts/create_local_config.py
	$(VENV)/python scripts/parallel_run.py --pipeline extracellular

intracellular:
	$(VENV)/python scripts/create_local_config.py
	$(VENV)/python scripts/parallel_run.py --pipeline intracellular

################################
# Data visualization and analysis
################################
vis:
	$(VENV)/python scripts/create_local_config.py
	$(VENV)/kedro run -p dashboard