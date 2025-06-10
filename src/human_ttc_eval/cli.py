import click
from pathlib import Path
import logging
import sys
from typing import Optional, List

# Attempt to import dataset modules - comment out unused ones for now
# try:
# from .datasets.kypo import parser as kypo_parser_module # noqa
# from .datasets.kypo import summariser as kypo_summariser_module # noqa
# from .datasets.kypo import retriever as kypo_retriever_module    # noqa
from .datasets.cybench import cybench_retrieve # noqa
from .datasets.cybench import cybench_prepare # noqa # TODO: Rename to cybench_prepare
from .datasets.cybench import cybench_describe # noqa # TODO: Rename to cybench_describe
from .datasets.cybench import cybench_bench # noqa
from .datasets.nl2bash import nl2bash_retrieve # noqa
from .datasets.nl2bash import nl2bash_prepare # noqa
from .datasets.nl2bash import nl2bash_describe # noqa
from .datasets.nl2bash import nl2bash_bench # noqa
from .datasets.intercode_ctf import intercode_ctf_retrieve # noqa

from .core.registry import (
    get_preparer, list_preparers,
    get_describer, list_describers,
    get_retriever, list_retrievers,
    get_bench, list_benches
)
from . import config # Import the project config

# Configure basic logging for the CLI
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(name)s - %(message)s')
logger = logging.getLogger("human_ttc_eval.cli")

@click.group()
def cli():
    """Human-TTC-Eval: A CLI tool for preparing, describing, retrieving, and benchmarking datasets."""
    pass

@cli.command("list-prepare")
def cli_list_prepare():
    """Lists all available dataset preparers."""
    try:
        preparers = list_preparers()
        if preparers:
            click.echo("Available preparers:")
            for p_name in preparers:
                click.echo(f"- {p_name}")
        else:
            click.echo("No preparers registered.")
    except Exception as e:
        click.echo(f"Error listing preparers: {e}", err=True)

@cli.command("prepare")
@click.argument("dataset_name", type=str)
def cli_prepare(dataset_name: str):
    """Prepares raw dataset files into the METR JSONL format."""
    logger.info(f"CLI prepare command initiated for dataset: '{dataset_name}'")
    try:
        prepare_class = get_preparer(dataset_name)
        
        raw_data_dir = config.DATA_DIR / "raw" / dataset_name
        processed_data_dir = config.DATA_DIR / "processed" / dataset_name
        
        if not raw_data_dir.exists():
            click.echo(f"Error: Raw data directory not found: {raw_data_dir}", err=True)
            click.echo(f"Please ensure raw data for '{dataset_name}' is present.")
            return

        processed_data_dir.mkdir(parents=True, exist_ok=True)
        
        preparer_instance = prepare_class()
        
        click.echo(f"Starting preparation for dataset: {dataset_name}...")
        preparer_instance.run() 

    except ValueError as e:
        click.echo(f"Error: {e}", err=True)
        logger.error(f"ValueError during prepare setup for '{dataset_name}': {e}")
    except Exception as e:
        click.echo(f"An unexpected error occurred during preparation of '{dataset_name}': {e}", err=True)
        logger.error(f"Unexpected error during preparation of '{dataset_name}':", exc_info=True)


@cli.command("list-describe")
def cli_list_describe():
    """Lists all available dataset describers (formerly summarisers)."""
    try:
        describers = list_describers()
        if describers:
            click.echo("Available describers:")
            for d_name in describers:
                click.echo(f"- {d_name}")
        else:
            click.echo("No describers registered.")
    except Exception as e:
        click.echo(f"Error listing describers: {e}", err=True)

@cli.command("describe")
@click.argument("dataset_name", type=str)
def cli_describe(dataset_name: str):
    """Describes a dataset from its JSONL representation (human or AI runs)."""
    logger.info(f"CLI describe command initiated for dataset: '{dataset_name}'")
    try:
        describe_class = get_describer(dataset_name)
        
        input_jsonl_path = config.DATA_DIR / "processed" / dataset_name / f"{dataset_name}_human_runs.jsonl"
        if not input_jsonl_path.exists():
            click.echo(f"Error: Default input file not found: {input_jsonl_path}", err=True)
            click.echo(f"Consider running 'prepare {dataset_name}'.", err=True)
            return

        output_dir_path = config.RESULTS_DIR / "dataset-summaries" / dataset_name
        output_dir_path.mkdir(parents=True, exist_ok=True)

        # Describe class takes input_files (List[Path]) and output_dir
        describer_instance = describe_class(input_files=[input_jsonl_path], output_dir=output_dir_path)
        
        click.echo(f"Starting description for {input_jsonl_path.name}...")
        summary_data = describer_instance.run() # run() should orchestrate load, describe, save
        
        # Assuming describer_instance.run() returns some dict of results or handles all output/logging.
        if summary_data: # Or check logs from the run method
             click.echo(f"Successfully described {input_jsonl_path.name}. Outputs saved to {output_dir_path}")
        else:
            click.echo(f"Description of {input_jsonl_path.name} completed. Check outputs in {output_dir_path}. Run might return None on success.")


    except ValueError as e:
        click.echo(f"Error: {e}", err=True)
        logger.error(f"ValueError during describe setup for '{dataset_name}': {e}")
    except Exception as e:
        click.echo(f"An unexpected error occurred during description of '{dataset_name}': {e}", err=True)
        logger.error(f"Unexpected error during description of '{dataset_name}':", exc_info=True)

@cli.group("retrieve")
def cli_retrieve_group():
    """Commands for retrieving raw data for datasets."""
    pass

@cli_retrieve_group.command("list") # Becomes `retrieve list`
def cli_retrieve_list():
    """Lists all available dataset retrievers."""
    try:
        retrievers = list_retrievers()
        if retrievers:
            click.echo("Available retrievers:")
            for r_name in retrievers:
                click.echo(f"- {r_name}")
        else:
            click.echo("No retrievers registered.")
    except Exception as e:
        click.echo(f"Error listing retrievers: {e}", err=True)

@cli_retrieve_group.command("run")
@click.argument("dataset_name", type=str)
def cli_retrieve_run(dataset_name: str):
    """Retrieves raw data for a specified dataset."""
    logger.info(f"CLI retrieve run initiated for dataset: '{dataset_name}'")
    try:
        retriever_class = get_retriever(dataset_name)
        
        # Retrieve class constructor should take dataset_name.
        # It will internally set its output_dir using config.DATA_DIR / "raw" / dataset_name
        retriever_instance = retriever_class(dataset_name=dataset_name)
        
        click.echo(f"Starting data retrieval for {dataset_name}...")
        # retrieve() method should handle saving to its designated raw data directory
        retrieved_files = retriever_instance.retrieve() # retrieve() should return list of paths or confirm success

        if retrieved_files: # Assuming retrieve returns list of files or some confirmation
            click.echo(f"Successfully retrieved data for {dataset_name}.")
            # Output dir is internally managed by retriever: config.DATA_DIR / "raw" / dataset_name
            click.echo(f"Raw data saved to: {config.DATA_DIR / 'raw' / dataset_name}")
            if isinstance(retrieved_files, list) and all(isinstance(p, Path) for p in retrieved_files):
                 for p in retrieved_files: click.echo(f"  - {p.name}")
        else:
            # This could mean an error or that retrieve() logs its own success and returns None.
            click.echo(f"Data retrieval for {dataset_name} completed. Check logs and {config.DATA_DIR / 'raw' / dataset_name}.")


    except ValueError as e:
        click.echo(f"Error: {e}", err=True)
        logger.error(f"ValueError during '{dataset_name}' retrieval setup: {e}")
    except Exception as e:
        click.echo(f"An unexpected error occurred during '{dataset_name}' retrieval: {e}", err=True)
        logger.error(f"Unexpected error during '{dataset_name}' retrieval: {e}", exc_info=True)

@cli.command("list-bench")
def cli_list_bench():
    """Lists all available dataset benchmark runners."""
    try:
        benches = list_benches() # Uses list_benches from registry
        if benches:
            click.echo("Available benchmark runners:")
            for b_name in benches:
                click.echo(f"- {b_name}")
        else:
            click.echo("No benchmark runners registered.")
    except Exception as e:
        click.echo(f"Error listing benchmark runners: {e}", err=True)

@cli.command("benchmark")
@click.argument("dataset_name", type=str) # Renamed from 'dataset'
@click.option("--model", required=True, 
              help="Model identifier (e.g., 'openai/gpt-4o', 'anthropic/claude-3-opus')")
@click.option("--num-runs", default=1, type=int, show_default=True,
              help="Number of evaluation runs for statistical analysis.")
@click.option("--task-ids", 
              help="Comma-separated list of specific task IDs to run. If not provided, runs all tasks for the dataset.")
def cli_benchmark(dataset_name: str, model: str, num_runs: int, task_ids: Optional[str]):
    """Run benchmark evaluation on a dataset using a specified model."""
    logger.info(f"CLI benchmark initiated for dataset: {dataset_name}, model: {model}")
    
    try:
        bench_class = get_bench(dataset_name)
        
        # dataset_dir for Bench class is data/raw/<dataset_name> (for human baseline loading)
        # output_dir for Bench class is results/benchmarks/<dataset_name>
        bench_dataset_dir = config.DATA_DIR / "raw" / dataset_name 
        bench_output_dir = config.RESULTS_DIR / "benchmarks" / dataset_name
        
        logger.info(f"Creating output directory: {bench_output_dir}")
        bench_output_dir.mkdir(parents=True, exist_ok=True)
        # Note: human baseline for the bench instance is loaded from:
        # bench_dataset_dir.parent.parent / "processed" / dataset_name / f"{dataset_name}_prepared.jsonl"
        # which translates to: config.DATA_DIR / "processed" / dataset_name / f"{dataset_name}_prepared.jsonl"
        # This means the 'prepare' step must have been run for the dataset.

        if not (config.DATA_DIR / "processed" / dataset_name / f"{dataset_name}_prepared.jsonl").exists():
            click.echo(f"Warning: Processed human baseline for '{dataset_name}' not found.", err=True)
            click.echo(f"Path checked: {config.DATA_DIR / 'processed' / dataset_name / f'{dataset_name}_prepared.jsonl'}", err=True)
            click.echo(f"Benchmark may run but might lack human comparison data or fail if task metadata is derived from it.", err=True)
            # Depending on Bench implementation, this might be a hard requirement.

        # Bench class now takes dataset_dir and output_dir
        bench_instance = bench_class(dataset_dir=bench_dataset_dir, output_dir=bench_output_dir)
        
        task_list: Optional[List[str]] = None
        if task_ids:
            task_list = [task.strip() for task in task_ids.split(",") if task.strip()]
            click.echo(f"Running evaluation on {len(task_list)} specific task(s): {', '.join(task_list)}")
        else:
            click.echo(f"Running evaluation on all available tasks for {dataset_name}.")
        
        # Model validation can be part of the Bench class instance if needed.
        # e.g., if hasattr(bench_instance, 'validate_model_name') and not bench_instance.validate_model_name(model): ...

        click.echo(f"Starting {dataset_name} evaluation for model: {model}")
        if num_runs > 1:
            click.echo(f"Performing {num_runs} evaluation runs...")
        click.echo("This may take a while...")
        
        all_run_results = []
        for run_num in range(1, num_runs + 1):
            if num_runs > 1:
                click.echo(f"--- Run {run_num}/{num_runs} ---")
            
            # run_evaluation now returns a BenchResult object
            result = bench_instance.run_evaluation(
                model_name=model,
                model_alias=model,
                task_ids=task_list
            )
            all_run_results.append(result)
            
            if result.success:
                # save_result saves the BenchResult (including all runs.jsonl, summary, etc.)
                # into a timestamped subdirectory of bench_instance.output_dir
                saved_path = bench_instance.save_result(result)
                click.echo(f"✅ Run {run_num} completed. Results: {result.summary_stats.get('successful_tasks',0)}/{result.summary_stats.get('total_tasks',0)} successful.")
                click.echo(f"   Results saved to: {saved_path}")

            else:
                click.echo(f"❌ Run {run_num} failed: {result.error_message}")
        
        successful_runs_count = sum(1 for res in all_run_results if res.success)
        if num_runs > 1:
            click.echo(f"\n🏁 Multi-run summary: {successful_runs_count}/{num_runs} runs successful.")
        
        if successful_runs_count == 0 and num_runs > 0:
             click.echo(f"❌ All {num_runs} benchmark run(s) failed for {dataset_name} with model {model}.", err=True)
        elif num_runs == 1 and not all_run_results[0].success:
             click.echo(f"❌ Benchmark failed for {dataset_name} with model {model}.", err=True)


    except ImportError as e: # Should be caught by registry if module not found
        click.echo(f"Error: Could not import benchmark runner for '{dataset_name}': {e}", err=True)
    except ValueError as e: # Raised by get_bench if not found
        click.echo(f"Error: {e}", err=True)
        logger.error(f"ValueError during benchmark setup for '{dataset_name}': {e}")
    except Exception as e:
        click.echo(f"An unexpected error occurred during benchmark of '{dataset_name}': {e}", err=True)
        logger.error(f"Unexpected error during benchmark of '{dataset_name}': {e}", exc_info=True)

@cli.command("plot")
@click.option("--dataset", 
              help="Optional dataset filter (e.g., 'cybench', 'nl2bash'). If not specified, plots all datasets.")
@click.option("--success-rate", default=50, type=int,
              help="Success rate percentage for horizon calculation (0-100). Default: 50")
def cli_plot(dataset: Optional[str], success_rate: int):
    """Generate METR-style horizon plots from benchmark results."""
    logger.info(f"CLI plot initiated with dataset filter: {dataset}, success rate: {success_rate}%")
    
    # Validate success rate
    if not 0 <= success_rate <= 100:
        click.echo("Error: Success rate must be between 0 and 100", err=True)
        return
    
    try:
        from .analysis.plotter import create_horizon_plots
        
        # Check if we have any benchmark results
        benchmarks_dir = config.RESULTS_DIR / "benchmarks"
        if not benchmarks_dir.exists() or not any(benchmarks_dir.iterdir()):
            click.echo(f"Error: No benchmark results found in {benchmarks_dir}", err=True)
            click.echo("Run benchmarks first using 'benchmark' command.", err=True)
            return
        
        # Check if METR code is available
        metr_path = config.THIRD_PARTY_DIR / "eval-analysis-public"
        if not metr_path.exists():
            click.echo(f"Error: METR analysis code not found at {metr_path}", err=True)
            click.echo("Run 'make setup' to clone required repositories.", err=True)
            return
        
        click.echo(f"Generating METR-style horizon plots at {success_rate}% success rate...")
        if dataset:
            click.echo(f"Filtering to dataset: {dataset}")
        else:
            click.echo("Processing all available datasets")
        
        # Generate plots using METR's methodology
        plot_files = create_horizon_plots(
            success_rates=[success_rate],
            dataset_filter=dataset
        )
        
        if plot_files:
            output_dir = config.RESULTS_DIR / "plots"
            click.echo(f"✅ Horizon plots generated successfully!")
            click.echo(f"📊 Generated {len(plot_files)} plots using METR's methodology")
            click.echo(f"📁 Plots saved to: {output_dir}")
            
            for plot_file in plot_files:
                click.echo(f"   • {plot_file.name}")
        else:
            click.echo("⚠️  No plots generated. Check if benchmark results contain successful runs.", err=True)
        
    except ImportError as e:
        click.echo(f"Error: Could not import required modules: {e}", err=True)
        click.echo("Ensure all dependencies are installed.", err=True)
    except Exception as e:
        click.echo(f"An unexpected error occurred: {e}", err=True)
        logger.error(f"Unexpected error during plotting: {e}", exc_info=True)

if __name__ == "__main__":
    cli() 