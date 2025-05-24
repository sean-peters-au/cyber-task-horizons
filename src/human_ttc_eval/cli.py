import click
from pathlib import Path
import logging
import sys
from typing import Optional

try:
    from .datasets.kypo import parser as kypo_parser_module # noqa
    from .datasets.kypo import summariser as kypo_summariser_module # noqa
    from .datasets.cybench import retrieve as cybench_retrieve_module # noqa
    from .datasets.cybench import parser as cybench_parser_module # noqa
    from .datasets.cybench import summariser as cybench_summariser_module # noqa
    from .datasets.cybench import bench as cybench_bench_module # noqa
    from .datasets.nl2bash import retrieve as nl2bash_retrieve_module # noqa
    from .datasets.nl2bash import parser as nl2bash_parser_module # noqa
    from .datasets.nl2bash import summariser as nl2bash_summariser_module # noqa
    from .datasets.nl2bash import bench as nl2bash_bench_module # noqa
except ImportError as e:
    print(f"Warning: Could not import all dataset modules: {e}. Some commands might fail.", file=sys.stderr)

from .core.registry import get_parser, list_parsers, get_summariser, list_summarisers, get_retriever, list_retrievers


# Configure basic logging for the CLI
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(name)s - %(message)s')
logger = logging.getLogger("human_ttc_eval.cli")

@click.group()
def cli():
    """Human-TTC-Eval: A CLI tool for parsing, summarizing, and retrieving datasets."""
    pass

@cli.command("list-parsers")
def cli_list_parsers():
    """Lists all available dataset parsers."""
    try:
        parsers = list_parsers()
        if parsers:
            click.echo("Available parsers:")
            for p_name in parsers:
                click.echo(f"- {p_name}")
        else:
            click.echo("No parsers registered. Ensure dataset modules are imported in cli.py.")
    except Exception as e:
        click.echo(f"Error listing parsers: {e}", err=True)

@cli.command("parse")
@click.argument("dataset_name", type=str)
@click.option("--input-dir", "-i", required=True, 
              type=click.Path(exists=True, file_okay=False, dir_okay=True, readable=True, resolve_path=True),
              help="Root directory of the raw dataset.")
@click.option("--output-file", "-o", required=True, 
              type=click.Path(file_okay=True, dir_okay=False, writable=True, resolve_path=True),
              help="Path to the output JSONL file.")
def cli_parse(dataset_name: str, input_dir: str, output_file: str):
    """Parses a specified dataset and writes output to a JSONL file."""
    logger.info(f"CLI parse command initiated for dataset: '{dataset_name}'")
    try:
        parser_class = get_parser(dataset_name)
        input_path = Path(input_dir)
        output_path = Path(output_file)
        
        parser_instance = parser_class(input_dir=input_path, output_file=output_path)
        logger.info(f"Instantiated {dataset_name} parser")

        click.echo(f"Starting parsing for dataset: {dataset_name}...")
        runs_data = parser_instance.parse()

        if runs_data is not None: # Check for None, allow empty list
            logger.info(f"Parsing complete. {len(runs_data)} runs extracted. Writing to JSONL...")
            parser_instance.write_jsonl(runs_data) # write_jsonl handles mkdir and logging success
            click.echo(f"Successfully parsed {dataset_name} and wrote {len(runs_data)} records to {output_path}")
        else:
            logger.warning(f"Parsing {dataset_name} returned None. No output written.")
            click.echo(f"Parsing {dataset_name} did not produce data. No output written.", err=True)

    except ValueError as e:
        click.echo(f"Error: {e}", err=True)
        logger.error(f"ValueError during parse setup for '{dataset_name}': {e}")
    except Exception as e:
        click.echo(f"An unexpected error occurred during parsing of '{dataset_name}': {e}", err=True)
        logger.error(f"Unexpected error during parsing of '{dataset_name}':", exc_info=True)

@cli.command("list-summarisers")
def cli_list_summarisers():
    """Lists all available dataset summarisers."""
    try:
        summarisers = list_summarisers()
        if summarisers:
            click.echo("Available summarisers:")
            for s_name in summarisers:
                click.echo(f"- {s_name}")
        else:
            click.echo("No summarisers registered. Ensure dataset modules are imported in cli.py.")
    except Exception as e:
        click.echo(f"Error listing summarisers: {e}", err=True)

@cli.command("summarise")
@click.argument("dataset_name", type=str)
@click.option("--jsonl-file", "-j", required=True, 
              type=click.Path(exists=True, file_okay=True, dir_okay=False, readable=True, resolve_path=True),
              help="Path to the input JSONL file (e.g., all_runs.jsonl).")
@click.option("--output-dir", "-o", required=True, 
              type=click.Path(file_okay=False, dir_okay=True, writable=True, resolve_path=True),
              help="Directory to save summary outputs (CSVs, plots). Ensure this dir is specific, e.g., results/kypo.")
def cli_summarise(dataset_name: str, jsonl_file: str, output_dir: str):
    """Summarises a specified dataset from its JSONL output."""
    logger.info(f"CLI summarise command initiated for dataset: '{dataset_name}'")
    try:
        summariser_class = get_summariser(dataset_name)
        jsonl_path = Path(jsonl_file)
        output_dir_path = Path(output_dir)

        summariser_instance = summariser_class(jsonl_file_path=jsonl_path, output_dir=output_dir_path)
        
        click.echo(f"Starting summarisation for dataset: {dataset_name}...")
        summariser_instance.load_data()
        summariser_instance.summarise()
        summariser_instance.save_plots()
        click.echo(f"Successfully summarised {dataset_name}. Outputs saved to {output_dir_path}")

    except ValueError as e:
        click.echo(f"Error: {e}", err=True)
        logger.error(f"ValueError during summarise setup for '{dataset_name}': {e}")
    except Exception as e:
        click.echo(f"An unexpected error occurred during summarisation of '{dataset_name}': {e}", err=True)
        logger.error(f"Unexpected error during summarisation of '{dataset_name}':", exc_info=True)

@cli.group("retrieve")
def cli_retrieve():
    """Commands for retrieving raw data or metadata for datasets."""
    pass

@cli_retrieve.command("list")
def cli_retrieve_list():
    """Lists all available dataset retrievers."""
    try:
        retrievers = list_retrievers()
        if retrievers:
            click.echo("Available retrievers:")
            for r_name in retrievers:
                click.echo(f"- {r_name}")
        else:
            click.echo("No retrievers registered. Ensure dataset retriever modules are imported in cli.py.")
    except Exception as e:
        click.echo(f"Error listing retrievers: {e}", err=True)

@cli_retrieve.command("metadata")
@click.argument("dataset_name", type=str)
@click.option("--output-dir", "-o", required=True, 
              type=click.Path(file_okay=False, dir_okay=True, writable=True, resolve_path=True),
              help="Directory to save the output metadata.")
def cli_retrieve_metadata(dataset_name: str, output_dir: str):
    """Retrieve metadata for any supported dataset."""
    logger.info(f"CLI retrieve metadata initiated for dataset: '{dataset_name}'")
    try:
        retriever_class = get_retriever(dataset_name)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize retriever (all datasets use standard interface)
        retriever_instance = retriever_class(output_dir=output_path)
        result = retriever_instance.retrieve_metadata()

        click.echo(f"Successfully retrieved {dataset_name} metadata to: {result}")

    except ValueError as e:
        click.echo(f"Error: {e}", err=True)
        logger.error(f"ValueError during {dataset_name} metadata setup: {e}")
    except Exception as e:
        click.echo(f"An unexpected error occurred during {dataset_name} metadata retrieval: {e}", err=True)
        logger.error(f"Unexpected error during {dataset_name} metadata retrieval: {e}", exc_info=True)

@cli.command("benchmark")
@click.option("--dataset", required=True,
              help="Dataset to benchmark (e.g., 'cybench', 'nl2bash')")
@click.option("--model", required=True, 
              help="Model identifier (e.g., 'openai/gpt-4o-2024-05-13', 'anthropic/claude-3-sonnet')")
@click.option("--output-dir", "-o", required=True,
              type=click.Path(file_okay=False, dir_okay=True, writable=True, resolve_path=True),
              help="Directory to save benchmark results.")
@click.option("--num-runs", default=1, type=int,
              help="Number of evaluation runs for statistical analysis. Default: 1.")
@click.option("--tasks", 
              help="Comma-separated list of specific tasks to run. If not provided, runs all tasks.")
def cli_benchmark(dataset: str, model: str, output_dir: str, num_runs: int, tasks: Optional[str]):
    """Run benchmark evaluation on any supported dataset."""
    logger.info(f"CLI benchmark initiated for dataset: {dataset}, model: {model}")
    
    try:
        # Import benchmark registry
        from .core.registry import get_bench, list_benches
        
        # Get the appropriate benchmark class
        try:
            bench_class = get_bench(dataset)
        except ValueError as e:
            available = ", ".join(list_benches())
            click.echo(f"Error: {e}", err=True)
            click.echo(f"Available datasets: {available}", err=True)
            return
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize benchmark runner (all datasets use standard interface)
        bench = bench_class(output_dir=output_path)
        
        # Parse tasks if provided
        task_list = None
        if tasks:
            task_list = [task.strip() for task in tasks.split(",") if task.strip()]
            click.echo(f"Running evaluation on {len(task_list)} specific tasks")
        else:
            click.echo("Running evaluation on all available tasks")
        
        # Validate model if the benchmark supports it
        if hasattr(bench, 'validate_model_name') and not bench.validate_model_name(model):
            if hasattr(bench, 'AVAILABLE_MODELS'):
                available = ", ".join(bench.AVAILABLE_MODELS)
                click.echo(f"Error: Model '{model}' not supported by {dataset}.", err=True)
                click.echo(f"Available models: {available}", err=True)
            else:
                click.echo(f"Error: Model '{model}' not supported by {dataset}.", err=True)
            return
        
        click.echo(f"Starting {dataset} evaluation for model: {model}")
        if num_runs > 1:
            click.echo(f"Running {num_runs} evaluation runs for statistical analysis...")
        click.echo(f"This may take a while...")
        
        # Run evaluation multiple times (each saves its own timestamped file)
        successful_runs = 0
        for run_num in range(1, num_runs + 1):
            if num_runs > 1:
                click.echo(f"\n--- Run {run_num}/{num_runs} ---")
            
            result = bench.run_evaluation(model_name=model, tasks=task_list)
            
            if result.success:
                successful_runs += 1
                if num_runs == 1:
                    stats = result.summary_stats
                    click.echo(f"✅ {dataset.title()} evaluation completed successfully!")
                    click.echo(f"📊 Results: {stats.get('successful_tasks', 0)}/{stats.get('total_tasks', 0)} tasks successful")
                else:
                    click.echo(f"✅ Run {run_num} completed successfully")
            else:
                click.echo(f"❌ Run {run_num} failed: {result.error_message}")
        
        # Summary for multiple runs
        if num_runs > 1:
            click.echo(f"\n🏁 Multi-run summary: {successful_runs}/{num_runs} runs successful")
            click.echo(f"📁 Individual run results saved to: {output_path}")
            if successful_runs == 0:
                click.echo(f"❌ All {num_runs} runs failed", err=True)
        else:
            if result.success:
                click.echo(f"📁 Results saved to: {output_path}")
            else:
                click.echo(f"❌ {dataset.title()} evaluation failed: {result.error_message}", err=True)

    except ImportError as e:
        click.echo(f"Error: Could not import {dataset} benchmark: {e}", err=True)
    except Exception as e:
        click.echo(f"An unexpected error occurred: {e}", err=True)
        logger.error(f"Unexpected error during {dataset} benchmark: {e}", exc_info=True)

@cli.command("plot")
@click.option("--results-dir", "-r", required=True,
              type=click.Path(exists=True, file_okay=False, dir_okay=True, readable=True, resolve_path=True),
              help="Directory containing benchmark results.")
@click.option("--output-dir", "-o", required=True,
              type=click.Path(file_okay=False, dir_okay=True, writable=True, resolve_path=True),
              help="Directory to save generated plots.")
@click.option("--dataset", 
              help="Optional dataset filter (e.g., 'cybench', 'nl2bash'). If not specified, plots all datasets.")
@click.option("--success-rate", default=0.5, type=float,
              help="Success rate threshold for horizon calculation (0.0-1.0). Default: 0.5 (50%)")
def cli_plot(results_dir: str, output_dir: str, dataset: Optional[str], success_rate: float):
    """Generate METR-style horizon plots from benchmark results."""
    logger.info(f"CLI plot initiated for results_dir: {results_dir}")
    
    # Validate success rate
    if not 0.0 <= success_rate <= 1.0:
        click.echo("Error: Success rate must be between 0.0 and 1.0", err=True)
        return
    
    try:
        from .analysis.plotter import create_horizon_plots_from_benchmarks
        
        results_path = Path(results_dir)
        output_path = Path(output_dir)
        
        # Convert success rate to percentage for METR functions
        success_rate_pct = int(success_rate * 100)
        
        # Define required data paths
        project_root = Path(__file__).parent.parent.parent
        human_baselines_file = project_root / "data" / "cybench_human_runs.jsonl"
        models_registry_file = Path(__file__).parent / "models.json"
        
        # Check if required files exist
        if not human_baselines_file.exists():
            click.echo(f"Error: Human baselines file not found: {human_baselines_file}", err=True)
            click.echo("Run 'make cybench-parse' to generate human baseline data.", err=True)
            return
            
        if not models_registry_file.exists():
            click.echo(f"Error: Models registry not found: {models_registry_file}", err=True)
            return
        
        # Filter results directory by dataset if specified
        if dataset:
            dataset_results_dir = results_path / dataset
            if not dataset_results_dir.exists():
                click.echo(f"Error: No results found for dataset '{dataset}' in {results_path}", err=True)
                return
            results_path = dataset_results_dir
            click.echo(f"Filtering to dataset: {dataset}")
        else:
            click.echo("Processing all available datasets")
        
        click.echo(f"Generating METR-style horizon plots at {success_rate_pct}% success rate...")
        click.echo("Using METR's logistic regression methodology...")
        
        # Generate plots using METR's methodology
        plot_files = create_horizon_plots_from_benchmarks(
            benchmark_results_dir=results_path,
            human_baselines_file=human_baselines_file,
            models_registry_file=models_registry_file,
            output_dir=output_path,
            success_rates=[success_rate_pct]
        )
        
        click.echo(f"✅ Horizon plots generated successfully!")
        click.echo(f"📊 Generated {len(plot_files)} plots using METR's exact methodology")
        click.echo(f"📁 Plots saved to: {output_path}")
        
        for plot_file in plot_files:
            click.echo(f"   • {plot_file.name}")
        
    except ImportError as e:
        click.echo(f"Error: Could not import METR analysis functions: {e}", err=True)
        click.echo("Ensure third-party/eval-analysis-public is available.", err=True)
    except Exception as e:
        click.echo(f"An unexpected error occurred: {e}", err=True)
        logger.error(f"Unexpected error during plotting: {e}", exc_info=True)

if __name__ == "__main__":
    cli() 