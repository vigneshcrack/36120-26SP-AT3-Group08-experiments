#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_NAME = 36120-26sp-at3-group08-experiments
PYTHON_VERSION = 3.12
PYTHON_INTERPRETER = python

#################################################################################
# COMMANDS                                                                      #
#################################################################################


## Install Python dependencies
.PHONY: requirements
requirements:
	uv sync
	



## Delete all compiled Python files
.PHONY: clean
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete


## Lint using ruff (use `make format` to do formatting)
.PHONY: lint
lint:
	ruff format --check
	ruff check

## Format source code with ruff
.PHONY: format
format:
	ruff check --fix
	ruff format



## Run tests
.PHONY: test
test:
	python -m pytest tests


## Set up Python interpreter environment
.PHONY: create_environment
create_environment:
	uv venv --python $(PYTHON_VERSION)
	@echo ">>> New uv virtual environment created. Activate with:"
	@echo ">>> Windows: .\\\\.venv\\\\Scripts\\\\activate"
	@echo ">>> Unix/macOS: source ./.venv/bin/activate"
	



#################################################################################
# PROJECT RULES                                                                 #
#################################################################################


## Download a new Kraken snapshot of all 8 tokens into data/raw/
.PHONY: data
data: requirements
	$(PYTHON_INTERPRETER) -m at3_crypto.dataset


## Build the labelled Market Risk dataset into data/interim/
.PHONY: features
features:
	$(PYTHON_INTERPRETER) -m at3_crypto.features


## Re-fit the selected Student B configuration and score the test period once
.PHONY: evaluate
evaluate:
	$(PYTHON_INTERPRETER) -m at3_crypto.modeling.train --evaluate


## Re-fit the Student B models on all labelled data
.PHONY: train
train:
	$(PYTHON_INTERPRETER) -m at3_crypto.modeling.train


## Forecast next week's Market Risk Index for every token
.PHONY: predict
predict:
	$(PYTHON_INTERPRETER) -m at3_crypto.modeling.predict


## Re-run Student D's TOR pipeline end to end (reads the committed snapshot)
.PHONY: tor
tor:
	$(PYTHON_INTERPRETER) -m at3_crypto.modeling.student_d_tor --trials 40


#################################################################################
# Self Documenting Commands                                                     #
#################################################################################

.DEFAULT_GOAL := help

define PRINT_HELP_PYSCRIPT
import re, sys; \
lines = '\n'.join([line for line in sys.stdin]); \
matches = re.findall(r'\n## (.*)\n[\s\S]+?\n([a-zA-Z_-]+):', lines); \
print('Available rules:\n'); \
print('\n'.join(['{:25}{}'.format(*reversed(match)) for match in matches]))
endef
export PRINT_HELP_PYSCRIPT

help:
	@$(PYTHON_INTERPRETER) -c "${PRINT_HELP_PYSCRIPT}" < $(MAKEFILE_LIST)
