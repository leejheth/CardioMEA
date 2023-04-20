.ONESHELL:
.PHONY: setup

DIR           := $(shell basename `pwd`)
ENV           := cardio-env

setup:
	. $(shell conda info --base)/etc/profile.d/conda.sh && \
	conda env remove -y -n $(ENV) && \
	conda create -y -n $(ENV) python=3.9 && \
	conda activate $(ENV) && \
	python -m pip install -r src/requirements.txt && \
	echo 'PYTHONPATH=$(PWD)/src:$$PYTHONPATH' > .env && \
	echo 'CardioMEA project setup successful.' && \
	echo 'To activate your conda environment, run `conda activate $(ENV)`.'
