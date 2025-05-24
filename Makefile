# Makefile for the Cybersecurity Dataset Analysis Project

# --- Variables ---
PYTHON = uv run python
HTE_CLI = PYTHONPATH=src:$(PYTHONPATH) $(PYTHON) -m human_ttc_eval.cli

# General Project Paths
THIRD_PARTY_DIR = third-party

# Repositories to clone
THIRD_PARTY_REPOS = \
    https://github.com/METR/eval-analysis-public.git \
    https://github.com/princeton-nlp/intercode.git \
    https://github.com/andyzorigin/cybench.git \
    https://github.com/NYU-LLM-CTF/NYU_CTF_Bench.git \
    https://github.com/TellinaTool/nl2bash.git

# Dataset specific variables
DATASET_NAME_KYPO = kypo
RAW_DATA_DIR_KYPO = data/cybersecurity_dataset_v4
PARSED_DATA_FILE_KYPO = data/cybersecurity_human_runs.jsonl
SUMMARIES_DIR_KYPO = results/dataset-summaries/$(DATASET_NAME_KYPO)

# CyBench specific variables
DATASET_NAME_CYBENCH = cybench
CYBENCH_METADATA_DIR = data/raw/cybench
CYBENCH_PARSED_FILE = data/cybench_human_runs.jsonl
CYBENCH_SUMMARIES_DIR = results/dataset-summaries/$(DATASET_NAME_CYBENCH)

# NL2Bash specific variables
DATASET_NAME_NL2BASH = nl2bash
NL2BASH_METADATA_DIR = data/raw/nl2bash
NL2BASH_PARSED_FILE = data/processed/nl2bash/all_tasks.jsonl
NL2BASH_SUMMARIES_DIR = results/dataset-summaries/$(DATASET_NAME_NL2BASH)

# InterCode specific variables

# NYU-CTF specific variables

# METR specific variables

# Benchmark specific variables
BENCHMARK_RESULTS_DIR = results/benchmarks
CYBENCH_BENCHMARK_DIR = $(BENCHMARK_RESULTS_DIR)/cybench
NL2BASH_BENCHMARK_DIR = $(BENCHMARK_RESULTS_DIR)/nl2bash
MODEL ?= openai/gpt-4o-2024-05-13  # Default model for benchmarks
NUM_RUNS ?= 2

# Plot specific variables
PLOTS_DIR = results/plots
SUCCESS_RATE ?= 0.5  # Default success rate threshold for horizon plots

# Phony targets are targets that don't produce an output file with the same name
.PHONY: all help datasets benchmark docs bench clean clean_datasets clean_benchmarks clean_docs kypo-parse kypo-summarise cybench-retrieve cybench-parse cybench-summarise cybench-benchmark cybench-benchmark-unguided cybench-setup-env nl2bash-retrieve nl2bash-parse nl2bash-summarise nl2bash-benchmark plot plot-cybench plot-nl2bash plot-all third-party

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
	@echo "Makefile for Cybersecurity Dataset Analysis Project"
	@echo ""
	@echo "Usage:"
	@echo "  make <target>"
	@echo ""
	@echo "Available targets:"
	@echo "  help                Display this help message"
	@echo "  datasets            Process all supported datasets (KYPO, CyBench, NL2Bash)"
	@echo "  benchmark           Run benchmarks on all datasets"
	@echo "  plot                Generate horizon plots from benchmark results"
	@echo "  docs                Generate documentation"
	@echo "  bench               Run benchmarks (alias for benchmark)"
	@echo "  clean               Remove all generated files"
	@echo ""
	@echo "Dataset-specific targets:"
	@echo "  kypo-parse          Parse raw KYPO logs into JSONL format"
	@echo "  kypo-summarise      Generate summaries and plots from parsed KYPO data"
	@echo "  cybench-retrieve    Extract metadata from local CyBench repository clone"
	@echo "  cybench-parse       Parse CyBench metadata into METR all_runs.jsonl format"
	@echo "  cybench-summarise   Generate summaries and plots from CyBench data"
	@echo "  nl2bash-retrieve    Download NL2Bash dataset and extract metadata"
	@echo "  nl2bash-parse       Parse NL2Bash dataset into METR all_runs.jsonl format"
	@echo "  nl2bash-summarise   Generate summaries and statistics from NL2Bash data"
	@echo "  nl2bash-benchmark   Run NL2Bash benchmark evaluation"
	@echo ""
	@echo "Benchmark targets:"
	@echo "  cybench-setup-env         Sync API keys to CyBench directory"
	@echo "  cybench-benchmark         Run CyBench evaluation with subtasks (default model)"
	@echo "  cybench-benchmark-unguided Run CyBench evaluation in unguided mode"
	@echo ""
	@echo "Plotting targets:"
	@echo "  plot                Generate horizon plots for all datasets"
	@echo "  plot-cybench        Generate horizon plots for CyBench only"
	@echo "  plot-nl2bash        Generate horizon plots for NL2Bash only"
	@echo "  plot-all            Generate horizon plots at multiple success rates"
	@echo ""
	@echo "Variables:"
	@echo "  CYBENCH_REPO_PATH   Path to CyBench repository clone (default: third-party/cybench)"
	@echo "  MODEL               Model for benchmarks (default: openai/gpt-4o-2024-05-13)"
	@echo "  NUM_RUNS            Number of runs for statistical analysis (default: 1)"
	@echo "  SUCCESS_RATE        Success rate threshold for horizon plots (default: 0.5)"
	@echo ""
	@echo "Examples:"
	@echo "  make kypo-parse"
	@echo "  make cybench-retrieve"
	@echo "  make nl2bash-retrieve"
	@echo "  make cybench-benchmark MODEL=anthropic/claude-3-5-sonnet-20240620"
	@echo "  make cybench-benchmark-unguided
	@echo "  make plot SUCCESS_RATE=0.3"
	@echo "  make plot-cybench SUCCESS_RATE=0.7"
	@echo ""

# Main target to process all datasets (KYPO, CyBench, and NL2Bash)
datasets: kypo-summarise cybench-summarise nl2bash-summarise

# Main benchmark target
benchmark: cybench-benchmark

# Alias for benchmark
bench: benchmark

kypo-parse: $(PARSED_DATA_FILE_KYPO)

$(PARSED_DATA_FILE_KYPO):
	@echo ">>> Parsing KYPO dataset using CLI..."
	$(HTE_CLI) parse $(DATASET_NAME_KYPO) --input-dir $(RAW_DATA_DIR_KYPO) --output-file $(PARSED_DATA_FILE_KYPO)

kypo-summarise: $(PARSED_DATA_FILE_KYPO)
	@echo ">>> Summarising KYPO dataset using CLI..."
	@mkdir -p $(SUMMARIES_DIR_KYPO)
	$(HTE_CLI) summarise $(DATASET_NAME_KYPO) --jsonl-file $(PARSED_DATA_FILE_KYPO) --output-dir $(SUMMARIES_DIR_KYPO)

cybench-retrieve: third-party
	@echo ">>> Retrieving CyBench metadata from local repository..."
	@mkdir -p $(CYBENCH_METADATA_DIR)
	$(HTE_CLI) retrieve metadata $(DATASET_NAME_CYBENCH) --output-dir $(CYBENCH_METADATA_DIR)

cybench-parse: cybench-retrieve
	@echo ">>> Parsing CyBench metadata into METR format..."
	$(HTE_CLI) parse $(DATASET_NAME_CYBENCH) --input-dir $(CYBENCH_METADATA_DIR) --output-file $(CYBENCH_PARSED_FILE)

cybench-summarise: cybench-parse
	@echo ">>> Summarising CyBench dataset using CLI..."
	@mkdir -p $(CYBENCH_SUMMARIES_DIR)
	$(HTE_CLI) summarise $(DATASET_NAME_CYBENCH) --jsonl-file $(CYBENCH_PARSED_FILE) --output-dir $(CYBENCH_SUMMARIES_DIR)

nl2bash-retrieve: third-party
	@echo ">>> Retrieving NL2Bash dataset and metadata..."
	@mkdir -p $(NL2BASH_METADATA_DIR)
	$(HTE_CLI) retrieve metadata $(DATASET_NAME_NL2BASH) --output-dir $(NL2BASH_METADATA_DIR)

nl2bash-parse: nl2bash-retrieve
	@echo ">>> Parsing NL2Bash dataset into METR format..."
	@mkdir -p $(dir $(NL2BASH_PARSED_FILE))
	$(HTE_CLI) parse $(DATASET_NAME_NL2BASH) --input-dir $(NL2BASH_METADATA_DIR) --output-file $(NL2BASH_PARSED_FILE)

nl2bash-summarise: nl2bash-parse
	@echo ">>> Summarising NL2Bash dataset using CLI..."
	@mkdir -p $(NL2BASH_SUMMARIES_DIR)
	$(HTE_CLI) summarise $(DATASET_NAME_NL2BASH) --jsonl-file $(NL2BASH_PARSED_FILE) --output-dir $(NL2BASH_SUMMARIES_DIR)

cybench-setup-env: .env
	@echo ">>> Syncing API keys to CyBench directory..."
	@if [ ! -f .env ]; then \
		echo "❌ No .env file found in project root. Please create one with your API keys."; \
		echo "Required keys: OPENAI_API_KEY, ANTHROPIC_API_KEY, GOOGLE_API_KEY"; \
		exit 1; \
	fi
	@if [ ! -d "third-party/cybench" ]; then \
		echo "❌ CyBench repository not found at third-party/cybench"; \
		exit 1; \
	fi
	@echo "# Auto-generated from project root .env by Makefile" > "third-party/cybench/.env"
	@echo "# Only includes keys for models you have access to" >> "third-party/cybench/.env"
	@echo "" >> "third-party/cybench/.env"
	@grep -E "^(OPENAI_API_KEY|ANTHROPIC_API_KEY|GOOGLE_API_KEY)" .env >> "third-party/cybench/.env" || true
	@echo "" >> "third-party/cybench/.env"
	@echo "# Optional keys (leave empty if not needed)" >> "third-party/cybench/.env"
	@echo "AZURE_OPENAI_API_KEY=" >> "third-party/cybench/.env"
	@echo "AZURE_OPENAI_ENDPOINT=" >> "third-party/cybench/.env"
	@echo "TOGETHER_API_KEY=" >> "third-party/cybench/.env"
	@echo "HELM_API_KEY=" >> "third-party/cybench/.env"
	@echo "✅ API keys synced to third-party/cybench/.env"

cybench-benchmark: cybench-setup-env
	@echo ">>> Running CyBench benchmark evaluation..."
	@echo ">>> Model: $(MODEL)"
	@echo ">>> Number of runs: $(NUM_RUNS)"
	@echo ">>> Mode: Subtask (guided)"
	@mkdir -p $(CYBENCH_BENCHMARK_DIR)
	$(HTE_CLI) benchmark \
		--dataset cybench \
		--model $(MODEL) \
		--output-dir $(CYBENCH_BENCHMARK_DIR) \
		--num-runs $(NUM_RUNS)

cybench-benchmark-unguided: cybench-setup-env
	@echo ">>> Running CyBench benchmark evaluation (unguided mode)..."
	@echo ">>> Model: $(MODEL)"
	@echo ">>> Number of runs: $(NUM_RUNS)"
	@echo ">>> Mode: Unguided (single objective)"
	@mkdir -p $(CYBENCH_BENCHMARK_DIR)
	$(HTE_CLI) benchmark \
		--dataset cybench \
		--model $(MODEL) \
		--output-dir $(CYBENCH_BENCHMARK_DIR) \
		--num-runs $(NUM_RUNS)

nl2bash-benchmark:
	@echo ">>> Running NL2Bash benchmark evaluation..."
	@echo ">>> Model: $(MODEL)"
	@echo ">>> Number of runs: $(NUM_RUNS)"
	@mkdir -p $(NL2BASH_BENCHMARK_DIR)
	$(HTE_CLI) benchmark \
		--dataset nl2bash \
		--model $(MODEL) \
		--output-dir $(NL2BASH_BENCHMARK_DIR) \
		--num-runs $(NUM_RUNS)

plot:
	@echo ">>> Generating horizon plots for all datasets..."
	@echo ">>> Success rate threshold: $(SUCCESS_RATE)"
	@mkdir -p $(PLOTS_DIR)
	$(HTE_CLI) plot \
		--results-dir $(BENCHMARK_RESULTS_DIR) \
		--output-dir $(PLOTS_DIR) \
		--success-rate $(SUCCESS_RATE)

plot-cybench:
	@echo ">>> Generating horizon plots for CyBench dataset..."
	@echo ">>> Success rate threshold: $(SUCCESS_RATE)"
	@mkdir -p $(PLOTS_DIR)
	$(HTE_CLI) plot \
		--results-dir $(BENCHMARK_RESULTS_DIR) \
		--output-dir $(PLOTS_DIR) \
		--dataset cybench \
		--success-rate $(SUCCESS_RATE)

plot-nl2bash:
	@echo ">>> Generating horizon plots for NL2Bash dataset..."
	@echo ">>> Success rate threshold: $(SUCCESS_RATE)"
	@mkdir -p $(PLOTS_DIR)
	$(HTE_CLI) plot \
		--results-dir $(BENCHMARK_RESULTS_DIR) \
		--output-dir $(PLOTS_DIR) \
		--dataset nl2bash \
		--success-rate $(SUCCESS_RATE)

plot-all:
	@echo ">>> Generating horizon plots at multiple success rates..."
	@mkdir -p $(PLOTS_DIR)
	@for rate in 0.3 0.5 0.7 0.9; do \
		echo ">>> Generating plots at $${rate} success rate..."; \
		$(HTE_CLI) plot \
			--results-dir $(BENCHMARK_RESULTS_DIR) \
			--output-dir $(PLOTS_DIR) \
			--success-rate $$rate; \
	done

docs: datasets
	@echo ">>> Building Sphinx documentation..."
	@cd docs && sphinx-build -b html source build/html
	@echo "Documentation built in docs/build/html/index.html"

clean_datasets:
	@echo ">>> Cleaning generated dataset files and summaries..."
	rm -f $(PARSED_DATA_FILE_KYPO)
	rm -f $(CYBENCH_PARSED_FILE)
	rm -f $(NL2BASH_PARSED_FILE)
	rm -rf $(CYBENCH_METADATA_DIR)
	rm -rf $(NL2BASH_METADATA_DIR)
	rm -rf data/processed/nl2bash
	rm -rf results/dataset-summaries
	@echo "Dataset files and summaries cleaned."

clean_benchmarks:
	@echo ">>> Cleaning benchmark results..."
	rm -rf $(BENCHMARK_RESULTS_DIR)
	@echo "Benchmark results cleaned."

clean_plots:
	@echo ">>> Cleaning generated plots..."
	rm -rf $(PLOTS_DIR)
	@echo "Generated plots cleaned."

clean_docs:
	@echo ">>> Cleaning Sphinx build directory..."
	rm -rf docs/build
	@echo "Sphinx build directory cleaned."

clean: clean_datasets clean_benchmarks clean_plots clean_docs
	@echo "All generated files cleaned." 