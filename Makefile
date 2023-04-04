.ONESHELL:
.PHONY: setup

CONDA_BASE    := $(shell conda info --base)
DIR           := $(shell basename `pwd`)
ENV           := cardio-env

setup:
	conda env remove -y -n $(ENV) && \
	conda create -y -n $(ENV) python=3.9 && \
	. $(CONDA_BASE)/etc/profile.d/conda.sh && \
	conda activate $(ENV) && \
	$(CONDA_BASE)/envs/$(ENV)/bin/pip install -r src/requirements.txt && \
	echo 'PYTHONPATH=$(PWD)/src:$$PYTHONPATH' > .env && \
	echo 'CardioMEA project setup successful.' && \
	echo 'To activate your conda environment, run `conda activate $(ENV)`.'
