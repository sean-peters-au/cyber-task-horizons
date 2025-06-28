# Makefile for the Cybersecurity Dataset Analysis Project

# --- Variables ---
PYTHON = uv run python
HTE_CLI = PYTHONPATH=src:$(PYTHONPATH) $(PYTHON) -m human_ttc_eval.cli

# Core variables
MODEL ?= openai/gpt2-xl
NUM_RUNS ?= 1
SUCCESS_RATE ?= 50
DATASET ?= cybench
TIER ?= 1

# Dataset list
DATASETS = cybench nl2bash intercode-ctf cybashbench nyuctf

# Repositories to clone
THIRD_PARTY_REPOS = \
    https://github.com/METR/eval-analysis-public.git \
    https://github.com/princeton-nlp/intercode.git \
    https://github.com/andyzorigin/cybench.git \
    https://github.com/NYU-LLM-CTF/NYU_CTF_Bench.git \
    https://github.com/TellinaTool/nl2bash.git

# Custom Docker image
CUSTOM_AGENT_IMAGE = human-ttc-eval-agent:latest

# --- Model tiers for benchmarking ---
# Tier 1: Fastest and cheapest models, for quick checks.
MODELS_1 = \
	openai/gpt2-xl \
	openai/gpt-3.5-turbo \
	anthropic/claude-3-5-haiku-20241022 \
	openai/o4-mini-2025-04-16

# Tier 2: Mid-range models, a balance of cost and capability.
MODELS_2 = \
	openai/gpt2-xl \
	openai/gpt-3.5-turbo \
	openai/o4-mini-2025-04-16 \
	openai/davinci-002 \
	openai/gpt-4-0314 \
	openai/gpt-4-1106-preview \
	anthropic/claude-3-5-sonnet-20241022 \
	anthropic/claude-3-7-sonnet-20250219 \
	google/gemini-2.5-flash-preview-20250520

# Tier 3: Most powerful (and expensive) models, for deep analysis.
MODELS_3 = \
	openai/gpt2-xl \
	openai/gpt-3.5-turbo \
	openai/o4-mini-2025-04-16 \
	anthropic/claude-3-5-sonnet-20241022 \
	anthropic/claude-opus-4-20250514 \
	openai/o3-2025-04-16 \
	openai/gpt-4-0613 \
	openai/gpt-4-32k-0613

# Tier 4: Publication models from the blog post
MODELS_PUBLICATION = \
	openai/gpt2-xl \
	openai/davinci-002 \
	openai/gpt-3.5-turbo \
	anthropic/claude-3-5-sonnet-20240620 \
	anthropic/claude-3-5-haiku-20241022 \
	anthropic/claude-3-5-sonnet-20241022 \
	openai/o4-mini-2025-04-16 \
	openai/o3-2025-04-16 \
	google/gemini-2.5-pro-preview-06-05

# Phony targets
.PHONY: all help datasets docs clean clean_datasets clean_benchmarks clean_docs test \
        retrieve prepare describe bench retrieve-all prepare-all describe-all \
        plot third-party repro progress run-gpt2xl-local \
        run-davinci-local start-local-model-servers stop-local-model-servers \
        build-agent-image

# --- Docker Image Build ---
build-agent-image:
	@echo ">>> Building custom agent Docker image..."
	@docker build --platform linux/amd64 -t $(CUSTOM_AGENT_IMAGE) -f src/human_ttc_eval/docker/Dockerfile .
	@echo ">>> Custom agent image '$(CUSTOM_AGENT_IMAGE)' built successfully."

# --- Third-Party Repository Setup ---
third-party:
	@echo ">>> Checking and cloning third-party repositories into third-party/"
	@mkdir -p third-party
	@for repo_url in $(THIRD_PARTY_REPOS); do \
		repo_name=$$(basename $$repo_url .git); \
		target_dir=third-party/$$repo_name; \
		if [ "$$(basename $$repo_url)" = "cybench.git" ]; then \
			target_dir=third-party/cybench; \
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
	@echo "Human TTC Eval - Dataset preparation and evaluation pipeline"
	@echo ""
	@echo "Usage: make <target> [DATASET=<dataset>] [MODEL=<model>] [TIER=<tier>]"
	@echo ""
	@echo "Quick Start:"
	@echo "  make repro TIER=1       - Full reproduction pipeline (retrieve, prepare, describe, bench all)"
	@echo ""
	@echo "Data pipeline commands:"
	@echo "  make retrieve-all       - Retrieve all raw datasets"
	@echo "  make prepare-all        - Prepare all datasets for evaluation"
	@echo "  make describe-all       - Generate descriptions for all datasets"
	@echo "  make retrieve DATASET=cybench  - Retrieve specific dataset"
	@echo "  make prepare DATASET=nl2bash   - Prepare specific dataset"
	@echo "  make describe DATASET=intercode-ctf - Describe specific dataset"
	@echo ""
	@echo "Benchmarking commands:"
	@echo "  make bench DATASET=cybench MODEL=openai/gpt-4 - Run single benchmark"
	@echo "  make progress             - Check progress against publication models"
	@echo "  make repro TIER=1       - Run all datasets on MODELS_1 (fast)"
	@echo "  make repro TIER=2       - Run all datasets on MODELS_2 (medium)"
	@echo "  make repro TIER=3       - Run all datasets on MODELS_3 (full)"
	@echo "  make repro TIER=publication - Run all datasets on MODELS_PUBLICATION (publication models)"
	@echo ""
	@echo "Available datasets: cybench, nl2bash, intercode-ctf, cybashbench, nyuctf"
	@echo ""
	@echo "Model tiers:"
	@echo "  TIER=1: gpt2-xl, gpt-3.5-turbo, claude-3.5-haiku, o4-mini"
	@echo "  TIER=2: davinci-002, classic gpt-4s, sonnet variants, gemini-flash"
	@echo "  TIER=3: sonnet, frontier models (gemini-pro, opus-4, o3), other gpt-4s"
	@echo "  TIER=publication: Models listed in the blog post for final results"
	@echo ""
	@echo "Local model servers:"
	@echo "  make start-local-model-servers - Start all local servers in the background"
	@echo "  make stop-local-model-servers  - Stop all local servers"
	@echo "  make run-gpt2xl-local          - Run only gpt2-xl server in foreground"
	@echo "  make run-davinci-local         - Run only davinci proxy in foreground"
	@echo ""
	@echo "Examples:"
	@echo "  make repro TIER=1"
	@echo "  make bench DATASET=cybench MODEL=openai/gpt-4"
	@echo "  make prepare DATASET=nl2bash"

# Test target
test:
	@echo ">>> Running unit tests..."
	PYTHONPATH=src:$(PYTHONPATH) $(PYTHON) -m pytest src/tests -v

# Main target to process all datasets
datasets: describe-all

# --- Generic Dataset Targets ---
retrieve: third-party
	@echo ">>> Retrieving $(DATASET) raw data using CLI..."
	$(HTE_CLI) retrieve $(DATASET)

prepare: retrieve
	@echo ">>> Preparing $(DATASET) raw data into METR format using CLI..."
	$(HTE_CLI) prepare $(DATASET)

describe: prepare
	@echo ">>> Describing $(DATASET) dataset using CLI..."
	@mkdir -p results/dataset-summaries/$(DATASET)
	$(HTE_CLI) describe $(DATASET)

# --- Convenience targets for all datasets ---
retrieve-all:
	@for dataset in $(DATASETS); do \
		echo ">>> Processing $$dataset..."; \
		$(MAKE) retrieve DATASET=$$dataset; \
	done

prepare-all:
	@for dataset in $(DATASETS); do \
		echo ">>> Processing $$dataset..."; \
		$(MAKE) prepare DATASET=$$dataset; \
	done

describe-all:
	@for dataset in $(DATASETS); do \
		echo ">>> Processing $$dataset..."; \
		$(MAKE) describe DATASET=$$dataset; \
	done

# --- Benchmark Targets ---
bench: prepare
	@echo ">>> Running $(DATASET) benchmark evaluation..."
	@echo "Model: $(MODEL), Runs: $(NUM_RUNS)"
	@mkdir -p results/benchmarks/$(DATASET)
	$(HTE_CLI) benchmark $(DATASET) --model "$(MODEL)" --num-runs $(NUM_RUNS) 

# --- Progress Check Target ---
progress:
	@echo ">>> Checking benchmark progress for publication models..."
	$(PYTHON) scripts/check_progress.py --datasets "$(DATASETS)" --models "$(MODELS_PUBLICATION)"

# --- Plotting Targets ---
plot:
	@echo ">>> Generating horizon plots for all datasets..."
	@echo "Success rate: $(SUCCESS_RATE)%"
	@mkdir -p results/plots
	$(HTE_CLI) plot --success-rate $(SUCCESS_RATE)

# --- Task Review Target ---
review-cybashbench:
	@echo ">>> Starting CyBashBench task review tool..."
	$(PYTHON) scripts/review_cybashbench_tasks.py

# --- Local Model Server Targets ---
start-local-model-servers:
	@echo ">>> Starting all local model servers in the background..."
	@mkdir -p logs
	@echo ">>> Starting gpt2-xl server (logs at logs/gpt2xl-server.log)..."
	@HF_MODEL=gpt2-xl; \
	(PYTHONPATH=src: $(PYTHON) -m vllm.entrypoints.openai.api_server \
		--model $$HF_MODEL \
		--host 0.0.0.0 --port 8000 --max-model-len 1024 \
		--chat-template "{% for message in messages %}{% if message['role'] == 'system' %}{{ message['content'] + '\n\n' }}{% elif message['role'] == 'user' %}{{ message['content'] }}{% endif %}{% endfor %}" > logs/gpt2xl-server.log 2>&1 &)
	@echo ">>> Starting davinci-002 proxy server (logs at logs/davinci-proxy.log)..."
	@($(PYTHON) scripts/davinci_proxy_server.py > logs/davinci-proxy.log 2>&1 &)
	@echo ">>> Servers are starting. Use 'make stop-local-model-servers' to stop them."

run-gpt2xl-local:
	@echo ">>> Starting gpt2-xl server with vLLM..."
	@echo ">>> Server will be available at http://localhost:8000"
	@echo ">>> Use MODEL=openai/gpt2-xl when running benchmarks"
	@echo ">>> Press Ctrl+C to stop the server"
	@HF_MODEL=gpt2-xl; \
	echo ">>> Using HuggingFace model: $$HF_MODEL"; \
	PYTHONPATH=src: $(PYTHON) -m vllm.entrypoints.openai.api_server \
		--model $$HF_MODEL \
		--host 0.0.0.0 \
		--port 8000 \
		--max-model-len 1024 \
		--chat-template "{% for message in messages %}{% if message['role'] == 'system' %}{{ message['content'] + '\n\n' }}{% elif message['role'] == 'user' %}{{ message['content'] }}{% endif %}{% endfor %}"

run-davinci-local:
	@echo ">>> Starting davinci-002 proxy server..."
	@echo ">>> Server will be available at http://localhost:8001"
	@echo ">>> Use MODEL=openai/davinci-002 when running benchmarks"
	@echo ">>> Press Ctrl+C to stop the server"
	$(PYTHON) scripts/davinci_proxy_server.py

stop-local-model-servers:
	@echo ">>> Stopping all local model servers..."
	@pkill -f "vllm.entrypoints.openai.api_server" || echo "No vLLM server running"
	@pkill -f "davinci_proxy_server.py" || echo "No davinci-002 proxy server running"

# --- Reproduction Pipeline ---
repro: describe-all
	@echo ">>> Running full reproduction pipeline for TIER $(TIER)..."
	@if [ "$(TIER)" = "1" ]; then models="$(MODELS_1)"; \
	elif [ "$(TIER)" = "2" ]; then models="$(MODELS_2)"; \
	elif [ "$(TIER)" = "3" ]; then models="$(MODELS_3)"; \
	elif [ "$(TIER)" = "publication" ]; then models="$(MODELS_PUBLICATION)"; \
	else echo "Invalid TIER: $(TIER). Use 1, 2, 3, or publication."; exit 1; fi; \
	for model in $$models; do \
		for dataset in $(DATASETS); do \
			echo ">>> Running $$model on $$dataset..."; \
			$(MAKE) bench DATASET=$$dataset MODEL="$$model" NUM_RUNS=$(NUM_RUNS) || echo "Failed: $$model on $$dataset"; \
		done; \
	done
	@echo ">>> Reproduction pipeline complete!" 

# --- Documentation Target ---
docs: # Removed 'datasets' dependency for now, can be added if docs consume processed data.
	@echo ">>> Building Sphinx documentation..."
	@cd docs && sphinx-build -b html source build/html
	@echo "Documentation built in docs/build/html/index.html"

# --- Clean Targets ---
clean_datasets:
	@echo ">>> Cleaning generated dataset files and summaries..."
	rm -rf data/raw/* data/processed/* results/dataset-summaries/*
	@echo "Dataset files and summaries cleaned."

clean_benchmarks:
	@echo ">>> Cleaning benchmark results..."
	rm -rf results/benchmarks/*
	@echo "Benchmark results cleaned."

clean_plots:
	@echo ">>> Cleaning generated plots..."
	rm -rf results/plots/*
	@echo "Generated plots cleaned."

clean_docs:
	@echo ">>> Cleaning Sphinx build directory..."
	rm -rf docs/build
	@echo "Sphinx build directory cleaned."

clean: clean_datasets clean_benchmarks clean_plots clean_docs
	@echo "All generated files cleaned."