# Makefile for the Cybersecurity Dataset Analysis Project

# --- Variables ---
PYTHON = uv run python
# Add src to PYTHONPATH so python -m can find the package
HTE_CLI = PYTHONPATH=src:$(PYTHONPATH) $(PYTHON) -m human_ttc_eval.cli

SPHINXBUILD = sphinx-build
DOCSDIR = docs
DOCS_SOURCE = $(DOCSDIR)/source
DOCS_BUILD = $(DOCSDIR)/build

# Dataset specific variables (currently only KYPO)
DATASET_NAME_KYPO = kypo
RAW_DATA_DIR_KYPO = data/cybersecurity_dataset_v4
PARSED_DATA_FILE_KYPO = data/cybersecurity_human_runs.jsonl
SUMMARIES_DIR_KYPO = results/dataset-summaries/$(DATASET_NAME_KYPO)

# Source file dependencies for KYPO
CLI_PY = src/human_ttc_eval/cli.py
KYPO_PARSER_PY = src/human_ttc_eval/datasets/kypo/parser.py
KYPO_SUMMARISER_PY = src/human_ttc_eval/datasets/kypo/summariser.py

# Default dataset for single-dataset operations (can be overridden)
DATASET ?= $(DATASET_NAME_KYPO)

# Phony targets are targets that don't produce an output file with the same name
.PHONY: all help datasets docs bench clean clean_datasets clean_docs kypo-parse kypo-summarise

# --- Targets ---

all: help

help:
	@echo "Makefile for Cybersecurity Dataset Analysis Project"
	@echo ""
	@echo "Usage:"
	@echo "  make datasets          - Run dataset parsing and summarization for all configured datasets (currently KYPO)."
	@echo "  make kypo-parse        - Parse the KYPO dataset."
	@echo "  make kypo-summarise    - Summarise the KYPO dataset (depends on parsing)."
	@echo "  make docs              - Build the Sphinx documentation (depends on datasets)."
	@echo "  make bench [DATASET=name] - (Placeholder) Run benchmarks for a dataset (default: $(DATASET_NAME_KYPO))."
	@echo "  make clean             - Remove all generated files (datasets, summaries, docs build)."
	@echo "  make clean_datasets    - Remove generated dataset files and summaries."
	@echo "  make clean_docs        - Remove Sphinx build directory."
	@echo ""

# Main target to process all datasets (currently just KYPO)
datasets: kypo-summarise

kypo-parse: $(PARSED_DATA_FILE_KYPO)

$(PARSED_DATA_FILE_KYPO): $(CLI_PY) $(KYPO_PARSER_PY) # Add $(RAW_DATA_DIR_KYPO) if it's a file/can be tracked
	@echo ">>> Parsing KYPO dataset using CLI..."
	$(HTE_CLI) parse $(DATASET_NAME_KYPO) --input-dir $(RAW_DATA_DIR_KYPO) --output-file $(PARSED_DATA_FILE_KYPO)

kypo-summarise: $(PARSED_DATA_FILE_KYPO) $(CLI_PY) $(KYPO_SUMMARISER_PY)
	@echo ">>> Summarising KYPO dataset using CLI..."
	@mkdir -p $(SUMMARIES_DIR_KYPO) # Ensure directory exists before CLI tries to write into it
	$(HTE_CLI) summarise $(DATASET_NAME_KYPO) --jsonl-file $(PARSED_DATA_FILE_KYPO) --output-dir $(SUMMARIES_DIR_KYPO)

docs: datasets
	@echo ">>> Building Sphinx documentation..."
	@cd $(DOCSDIR) && $(SPHINXBUILD) -b html source build/html
	@echo "Documentation built in $(DOCS_BUILD)/html/index.html"

bench:
	@echo ">>> Benchmarking for dataset: $(DATASET) (Placeholder)..."
	@echo "(This is where you would add commands to run InspectAI or other benchmark tools for $(DATASET))"

clean_datasets:
	@echo ">>> Cleaning generated dataset files and summaries..."
	rm -f $(PARSED_DATA_FILE_KYPO)
	rm -rf results/dataset-summaries # Clean all dataset summaries for now
	@echo "Dataset files and summaries cleaned."

clean_docs:
	@echo ">>> Cleaning Sphinx build directory..."
	rm -rf $(DOCS_BUILD)
	@echo "Sphinx build directory cleaned."

clean: clean_datasets clean_docs
	@echo "All generated files cleaned." 