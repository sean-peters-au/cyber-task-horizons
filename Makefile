# Makefile for the Cybersecurity Dataset Analysis Project

# --- Variables ---
PYTHON = uv run python
HTE_CLI = PYTHONPATH=src:$(PYTHONPATH) $(PYTHON) -m human_ttc_eval.cli

# General Project Paths
THIRD_PARTY_DIR = third-party
DATA_DIR = data
RESULTS_DIR = results

# Repositories to clone
THIRD_PARTY_REPOS = \
    https://github.com/METR/eval-analysis-public.git \
    https://github.com/princeton-nlp/intercode.git \
    https://github.com/andyzorigin/cybench.git \
    https://github.com/NYU-LLM-CTF/NYU_CTF_Bench.git \
    https://github.com/TellinaTool/nl2bash.git

# Dataset specific variables
DATASET_NAME_KYPO = kypo
RAW_DATA_DIR_KYPO = $(DATA_DIR)/raw/$(DATASET_NAME_KYPO) # Base dir for raw kypo data
PREPARED_DATA_FILE_KYPO = $(DATA_DIR)/processed/$(DATASET_NAME_KYPO)/$(DATASET_NAME_KYPO)_prepared.jsonl
SUMMARIES_DIR_KYPO = $(RESULTS_DIR)/dataset-summaries/$(DATASET_NAME_KYPO)

# CyBench specific variables
DATASET_NAME_CYBENCH = cybench
RAW_DATA_DIR_CYBENCH = $(DATA_DIR)/raw/$(DATASET_NAME_CYBENCH) # retrieve run output_dir
PREPARED_DATA_FILE_CYBENCH = $(DATA_DIR)/processed/$(DATASET_NAME_CYBENCH)/$(DATASET_NAME_CYBENCH)_prepared.jsonl
SUMMARIES_DIR_CYBENCH = $(RESULTS_DIR)/dataset-summaries/$(DATASET_NAME_CYBENCH)

# NL2Bash specific variables
DATASET_NAME_NL2BASH = nl2bash
RAW_DATA_DIR_NL2BASH = $(DATA_DIR)/raw/$(DATASET_NAME_NL2BASH)
PREPARED_DATA_FILE_NL2BASH = $(DATA_DIR)/processed/$(DATASET_NAME_NL2BASH)/$(DATASET_NAME_NL2BASH)_prepared.jsonl
SUMMARIES_DIR_NL2BASH = $(RESULTS_DIR)/dataset-summaries/$(DATASET_NAME_NL2BASH)

# InterCode specific variables
DATASET_NAME_INTERCODE_CTF = intercode-ctf
RAW_DATA_DIR_INTERCODE_CTF = $(DATA_DIR)/raw/$(DATASET_NAME_INTERCODE_CTF)
PREPARED_DATA_FILE_INTERCODE_CTF = $(DATA_DIR)/processed/$(DATASET_NAME_INTERCODE_CTF)/$(DATASET_NAME_INTERCODE_CTF)_prepared.jsonl
SUMMARIES_DIR_INTERCODE_CTF = $(RESULTS_DIR)/dataset-summaries/$(DATASET_NAME_INTERCODE_CTF)

# Benchmark specific variables
BENCHMARK_BASE_DIR = $(RESULTS_DIR)/benchmarks # Base for all benchmark outputs
CYBENCH_BENCHMARK_OUTPUT_DIR = $(BENCHMARK_BASE_DIR)/$(DATASET_NAME_CYBENCH)
NL2BASH_BENCHMARK_OUTPUT_DIR = $(BENCHMARK_BASE_DIR)/$(DATASET_NAME_NL2BASH)
INTERCODE_CTF_BENCHMARK_OUTPUT_DIR = $(BENCHMARK_BASE_DIR)/$(DATASET_NAME_INTERCODE_CTF)
MODEL ?= openai/gpt-4o-2024-05-13  # Default model for benchmarks
NUM_RUNS ?= 1 # Changed default to 1 as per user request in previous interactions for CLI

# Plot specific variables
PLOTS_OUTPUT_DIR = $(RESULTS_DIR)/plots # Default output for plots
SUCCESS_RATE ?= 0.5  # Default success rate threshold for horizon plots

# Test specific variables
TEST_DIR = src/tests

# Phony targets
.PHONY: all help datasets benchmark docs bench clean clean_datasets clean_benchmarks clean_docs test \
        kypo-prepare kypo-describe \
        cybench-retrieve cybench-prepare cybench-describe cybench-benchmark \
        nl2bash-retrieve nl2bash-prepare nl2bash-describe nl2bash-benchmark \
        intercode-ctf-retrieve intercode-ctf-prepare intercode-ctf-describe intercode-ctf-benchmark \
        plot plot-cybench plot-nl2bash plot-all third-party

# --- Third-Party Repository Setup ---
third-party:
	@echo ">>> Checking and cloning third-party repositories into $(THIRD_PARTY_DIR)/"
	@mkdir -p $(THIRD_PARTY_DIR)
	@for repo_url in $(THIRD_PARTY_REPOS); do \
		repo_name=$$(basename $$repo_url .git); \
		target_dir=$(THIRD_PARTY_DIR)/$$repo_name; \
		if [ "$$repo_name" = "cybench" ]; then \
			: ; \
		fi; \
		if [ "$$(basename $$repo_url)" = "cybench.git" ]; then \
			target_dir=$(THIRD_PARTY_DIR)/cybench; \
		fi; \
		if [ -d "$$target_dir" ]; then \
			echo ">>> $$target_dir already exists. Skipping clone for $$repo_url."; \
		else \
			echo ">>> Cloning $$repo_url into $$target_dir..."; \
			git clone --depth=1 $$repo_url $$target_dir; \
		fi; \
	done
	@echo ">>> Third-party repositories setup complete."

# --- Targets ---

all: help

help:
	@echo "Makefile for Human-TTC-Eval Project (Refactored CLI)"
	@echo ""
	@echo "Usage: make <target>"
	@echo ""
	@echo "Core Commands:"
	@echo "  test                    Run unit tests"
	@echo "  clean                   Remove all generated files (datasets, benchmarks, plots, docs)"
	@echo "  third-party             Setup third-party repositories"
	@echo "  docs                    Generate Sphinx documentation"
	@echo ""
	@echo "Dataset Workflow (example for 'cybench'):"
	@echo "  make cybench-retrieve   Retrieve raw data for CyBench"
	@echo "  make cybench-prepare    Prepare raw CyBench data to METR JSONL format"
	@echo "  make cybench-describe   Generate summaries/analysis for CyBench human data"
	@echo "  make cybench-benchmark  Run AI model benchmarks on CyBench (uses MODEL, NUM_RUNS)"
	@echo ""
	@echo "Available dataset targets (replace 'cybench' with 'kypo', 'nl2bash', 'intercode-ctf'):"
	@echo "  <dataset>-retrieve, <dataset>-prepare, <dataset>-describe, <dataset>-benchmark"
	@echo ""
	@echo "General Plotting:"
	@echo "  make plot                 Generate horizon plots for all datasets found in results/benchmarks"
	@echo "  make plot-cybench         Generate plots specifically for CyBench results"
	@echo "  make plot-nl2bash         Generate plots specifically for NL2Bash results"
	@echo "  make plot-all             Generate plots at multiple success rates for all datasets"
	@echo ""
	@echo "Variables for Benchmarking & Plotting:"
	@echo "  MODEL                   Model for benchmarks (default: $(MODEL))"
	@echo "  NUM_RUNS                Number of benchmark runs (default: $(NUM_RUNS))"
	@echo "  SUCCESS_RATE            Success rate for horizon plots (default: $(SUCCESS_RATE))"
	@echo ""

# Test target
test:
	@echo ">>> Running unit tests..."
	PYTHONPATH=src:$(PYTHONPATH) $(PYTHON) -m pytest $(TEST_DIR) -v

# Main target to process all datasets
datasets: kypo-describe cybench-describe nl2bash-describe intercode-ctf-describe

# Main benchmark target
benchmark: cybench-benchmark nl2bash-benchmark intercode-ctf-benchmark # Add other datasets as they become ready

# Alias for benchmark
bench: benchmark

# --- KYPO Targets ---
kypo-prepare: $(PREPARED_DATA_FILE_KYPO)

$(PREPARED_DATA_FILE_KYPO):
	@echo ">>> Preparing KYPO dataset using CLI..."
	# Assumes raw data is already in data/raw/kypo as per previous project structure.
	# The Prepare class will look for data/raw/kypo internally.
	$(HTE_CLI) prepare $(DATASET_NAME_KYPO)

kypo-describe: $(PREPARED_DATA_FILE_KYPO)
	@echo ">>> Describing KYPO dataset using CLI..."
	@mkdir -p $(SUMMARIES_DIR_KYPO)
	$(HTE_CLI) describe $(DATASET_NAME_KYPO)

# --- CyBench Targets ---
cybench-retrieve: $(RAW_DATA_DIR_CYBENCH)

$(RAW_DATA_DIR_CYBENCH): third-party
	@echo ">>> Retrieving CyBench raw data using CLI..."
	# CLI command `retrieve run cybench` will create and populate data/raw/cybench
	$(HTE_CLI) retrieve run $(DATASET_NAME_CYBENCH)

cybench-prepare: $(PREPARED_DATA_FILE_CYBENCH)

$(PREPARED_DATA_FILE_CYBENCH): cybench-retrieve
	@echo ">>> Preparing CyBench raw data into METR format using CLI..."
	$(HTE_CLI) prepare $(DATASET_NAME_CYBENCH)

cybench-describe: $(PREPARED_DATA_FILE_CYBENCH)
	@echo ">>> Describing CyBench dataset using CLI..."
	@mkdir -p $(SUMMARIES_DIR_CYBENCH)
	$(HTE_CLI) describe $(DATASET_NAME_CYBENCH)

# --- NL2Bash Targets ---
nl2bash-retrieve: $(RAW_DATA_DIR_NL2BASH)

$(RAW_DATA_DIR_NL2BASH): third-party
	@echo ">>> Retrieving NL2Bash raw data using CLI..."
	$(HTE_CLI) retrieve run $(DATASET_NAME_NL2BASH)

nl2bash-prepare: $(PREPARED_DATA_FILE_NL2BASH)

$(PREPARED_DATA_FILE_NL2BASH): nl2bash-retrieve
	@echo ">>> Preparing NL2Bash raw data into METR format using CLI..."
	$(HTE_CLI) prepare $(DATASET_NAME_NL2BASH)

nl2bash-describe: $(PREPARED_DATA_FILE_NL2BASH)
	@echo ">>> Describing NL2Bash dataset using CLI..."
	@mkdir -p $(SUMMARIES_DIR_NL2BASH)
	$(HTE_CLI) describe $(DATASET_NAME_NL2BASH)

# --- InterCode-CTF Targets ---
intercode-ctf-retrieve: $(RAW_DATA_DIR_INTERCODE_CTF)

$(RAW_DATA_DIR_INTERCODE_CTF): third-party
	@echo ">>> Retrieving InterCode-CTF raw data using CLI..."
	$(HTE_CLI) retrieve run $(DATASET_NAME_INTERCODE_CTF)

intercode-ctf-prepare: $(PREPARED_DATA_FILE_INTERCODE_CTF)

$(PREPARED_DATA_FILE_INTERCODE_CTF): intercode-ctf-retrieve
	@echo ">>> Preparing InterCode-CTF raw data into METR format using CLI..."
	$(HTE_CLI) prepare $(DATASET_NAME_INTERCODE_CTF)

intercode-ctf-describe: $(PREPARED_DATA_FILE_INTERCODE_CTF)
	@echo ">>> Describing InterCode-CTF dataset using CLI..."
	@mkdir -p $(SUMMARIES_DIR_INTERCODE_CTF)
	$(HTE_CLI) describe $(DATASET_NAME_INTERCODE_CTF)

# --- Benchmark Targets ---
# cybench-setup-env is still relevant if cybench_bench.py uses it.
cybench-setup-env:
	@echo "Ensuring .env exists for CyBench specific setup..."
	@if [ ! -f .env ]; then \
		echo "Warning: Project root .env file not found. CyBench setup might rely on it."; \
	fi
	# The actual syncing logic, if still needed by the cybench_bench.py, should be there.
	# This target is now more of a prerequisite check or placeholder.

cybench-benchmark: cybench-setup-env $(PREPARED_DATA_FILE_CYBENCH)
	@echo ">>> Running CyBench benchmark evaluation..."
	@echo "Model: $(MODEL), Runs: $(NUM_RUNS)"
	@mkdir -p $(CYBENCH_BENCHMARK_OUTPUT_DIR)
	$(HTE_CLI) benchmark $(DATASET_NAME_CYBENCH) --model "$(MODEL)" --num-runs $(NUM_RUNS)

nl2bash-benchmark: $(PREPARED_DATA_FILE_NL2BASH)
	@echo ">>> Running NL2Bash benchmark evaluation..."
	@echo "Model: $(MODEL), Runs: $(NUM_RUNS)"
	@mkdir -p $(NL2BASH_BENCHMARK_OUTPUT_DIR)
	$(HTE_CLI) benchmark $(DATASET_NAME_NL2BASH) --model "$(MODEL)" --num-runs $(NUM_RUNS)

intercode-ctf-benchmark: $(PREPARED_DATA_FILE_INTERCODE_CTF)
	@echo ">>> Running InterCode-CTF benchmark evaluation..."
	@echo "Model: $(MODEL), Runs: $(NUM_RUNS)"
	@mkdir -p $(INTERCODE_CTF_BENCHMARK_OUTPUT_DIR)
	$(HTE_CLI) benchmark $(DATASET_NAME_INTERCODE_CTF) --model "$(MODEL)" --num-runs $(NUM_RUNS)

# --- Plotting Targets ---
plot:
	@echo ">>> Generating horizon plots for all datasets..."
	@echo "Success rate threshold: $(SUCCESS_RATE)"
	@mkdir -p $(PLOTS_OUTPUT_DIR)
	$(HTE_CLI) plot \
		--results-dir $(BENCHMARK_BASE_DIR) \
		--output-dir $(PLOTS_OUTPUT_DIR) \
		--success-rate $(SUCCESS_RATE)

plot-cybench:
	@echo ">>> Generating horizon plots for CyBench dataset..."
	@echo "Success rate threshold: $(SUCCESS_RATE)"
	@mkdir -p $(PLOTS_OUTPUT_DIR)
	$(HTE_CLI) plot \
		--results-dir $(CYBENCH_BENCHMARK_OUTPUT_DIR) \
		--output-dir $(PLOTS_OUTPUT_DIR) \
		--dataset $(DATASET_NAME_CYBENCH) \
		--success-rate $(SUCCESS_RATE)

plot-nl2bash:
	@echo ">>> Generating horizon plots for NL2Bash dataset..."
	@echo "Success rate threshold: $(SUCCESS_RATE)"
	@mkdir -p $(PLOTS_OUTPUT_DIR)
	$(HTE_CLI) plot \
		--results-dir $(NL2BASH_BENCHMARK_OUTPUT_DIR) \
		--output-dir $(PLOTS_OUTPUT_DIR) \
		--dataset $(DATASET_NAME_NL2BASH) \
		--success-rate $(SUCCESS_RATE)

plot-all:
	@echo ">>> Generating horizon plots at multiple success rates..."
	@mkdir -p $(PLOTS_OUTPUT_DIR)
	@for rate in 0.3 0.5 0.7 0.9; do \
		echo ">>> Generating plots at $${rate} success rate..."; \
		$(HTE_CLI) plot \
			--results-dir $(BENCHMARK_BASE_DIR) \
			--output-dir $(PLOTS_OUTPUT_DIR) \
			--success-rate $$rate; \
	done

# --- Documentation Target ---
docs: # Removed 'datasets' dependency for now, can be added if docs consume processed data.
	@echo ">>> Building Sphinx documentation..."
	@cd docs && sphinx-build -b html source build/html
	@echo "Documentation built in docs/build/html/index.html"

# --- Clean Targets ---
clean_datasets:
	@echo ">>> Cleaning generated dataset files and summaries..."
	rm -rf $(DATA_DIR)/raw/*/
	rm -rf $(DATA_DIR)/processed/*/
	rm -rf $(RESULTS_DIR)/dataset-summaries/*/
	@echo "Dataset files and summaries cleaned."

clean_benchmarks:
	@echo ">>> Cleaning benchmark results..."
	rm -rf $(BENCHMARK_BASE_DIR)/*/
	@echo "Benchmark results cleaned."

clean_plots:
	@echo ">>> Cleaning generated plots..."
	rm -rf $(PLOTS_OUTPUT_DIR)/*
	@echo "Generated plots cleaned."

clean_docs:
	@echo ">>> Cleaning Sphinx build directory..."
	rm -rf docs/build
	@echo "Sphinx build directory cleaned."

clean: clean_datasets clean_benchmarks clean_plots clean_docs
	@echo "All generated files cleaned." 