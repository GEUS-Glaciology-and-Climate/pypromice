SHELL = bash
.DEFAULT_GOAL := help
.PHONY: help

help: ## This help
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST)

env: ## Instructions for setting up python environment
	@echo "	Once: conda create -y -f environment.txt"
	@echo "	After: conda activate PROMICE_dev"
	@echo "	Note: See https://direnv.net/"

test: ## Run python testing framework

build: ## Build Python for pip installation
	python3 -m build


install: ## Install locally
	python3 -m pip install --upgrade .


dist: ## Distribute (to GitHub) for remote pip installation

test_py: ## Run Python code on test data
	PYTHONPATH=./src python ./bin/promiceAWS --config_file=./test_data/metadata/KPC_L.toml -i ./test_data/input -o ./test_data/output_py

test_GDL: ## Run GDL on test data
	cd test_data
	gdl -e awsdataprocessing_gdl_v3 -args infolder=./input/ outfolder=./output_GDL/ metadata=./metadata/KPC_L_metadata_TX.csv station=KPC_L

test_IDL: ## Run IDL on test data
	cd test_data
	idl -e awsdataprocessing_gdl_v3 -args infolder=./input/ outfolder=./output_IDL/ metadata=./metadata/KPC_L_metadata_TX.csv station=KPC_L

FORCE: # dummy target

clean: ## Clean everything
	rm -fR test_data/output_{IDL,GDL,py}
