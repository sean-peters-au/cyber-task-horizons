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
    from .datasets.nl2bash import retrieve as nl2bash_retrieve_module # noqa
    from .datasets.nl2bash import parser as nl2bash_parser_module # noqa
    from .datasets.nl2bash import summariser as nl2bash_summariser_module # noqa
except ImportError as e:
    print(f"Warning: Could not import all dataset modules: {e}. Some commands might fail.", file=sys.stderr)

from .core.registry import get_parser, list_parsers, get_summariser, list_summarisers, get_retriever, list_retrievers

# Import benchmark modules
try:
    from .datasets.cybench import bench as cybench_bench_module # noqa
except ImportError as e:
    print(f"Warning: Could not import benchmark modules: {e}. Some commands might fail.", file=sys.stderr)

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

    except ValueError as e: # Specific for get_parser not found
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

        # Ensure output directory exists; BaseSummariser also does this, but good practice here too.
        output_dir_path.mkdir(parents=True, exist_ok=True)

        summariser_instance = summariser_class(jsonl_file_path=jsonl_path, output_dir=output_dir_path)
        
        click.echo(f"Starting summarisation for dataset: {dataset_name}...")
        summariser_instance.load_data() # Loads data into summariser_instance.df
        summariser_instance.summarise()   # Generates stats (e.g., CSVs)
        summariser_instance.save_plots()  # Generates plots
        click.echo(f"Successfully summarised {dataset_name}. Outputs saved to {output_dir_path}")

    except ValueError as e: # Specific for get_summariser not found
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
        from .config import CYBENCH_REPO_PATH
        
        retriever_class = get_retriever(dataset_name)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Handle dataset-specific initialization
        if dataset_name == "cybench":
            retriever_instance = retriever_class(output_dir=output_path, cybench_repo_path=CYBENCH_REPO_PATH)
            result = retriever_instance.retrieve_metadata()
        elif dataset_name == "nl2bash":
            retriever_instance = retriever_class(output_dir=output_path)
            result = retriever_instance.retrieve_metadata()
        else:
            click.echo(f"Unsupported dataset: {dataset_name}", err=True)
            return

        click.echo(f"Successfully retrieved {dataset_name} metadata to: {result}")

    except ValueError as e:
        click.echo(f"Error: {e}", err=True)
        logger.error(f"ValueError during {dataset_name} metadata setup: {e}")
    except Exception as e:
        click.echo(f"An unexpected error occurred during {dataset_name} metadata retrieval: {e}", err=True)
        logger.error(f"Unexpected error during {dataset_name} metadata retrieval: {e}", exc_info=True)

@cli.group("benchmark") 
def cli_benchmark():
    """Commands for running AI model evaluations on datasets."""
    pass

@cli_benchmark.command("cybench")
@click.option("--model", required=True, 
              help="Model identifier (e.g., 'openai/gpt-4o-2024-05-13'). See CyBench docs for full list.")
@click.option("--output-dir", "-o", required=True,
              type=click.Path(file_okay=False, dir_okay=True, writable=True, resolve_path=True),
              help="Directory to save benchmark results.")
@click.option("--tasks", 
              help="Comma-separated list of specific tasks to run. If not provided, runs all tasks.")
@click.option("--max-iterations", default=15, type=int,
              help="Maximum iterations per task.")
@click.option("--unguided-mode", is_flag=True, default=False,
              help="Run in unguided mode (single objective, no subtasks).")
def cli_benchmark_cybench(model: str, output_dir: str, 
                         tasks: Optional[str], max_iterations: int, unguided_mode: bool):
    """Run CyBench evaluation using their native system."""
    logger.info(f"CLI benchmark cybench initiated for model: {model}")
    try:
        from .datasets.cybench.bench import CyBenchBench
        from .config import CYBENCH_REPO_PATH
        
        # Parse tasks if provided
        task_list = None
        if tasks:
            task_list = [task.strip() for task in tasks.split(",") if task.strip()]
            click.echo(f"Running evaluation on {len(task_list)} specific tasks")
        else:
            click.echo("Running evaluation on all available tasks")
        
        # Initialize benchmark runner
        bench = CyBenchBench(
            output_dir=Path(output_dir),
            cybench_repo_path=CYBENCH_REPO_PATH
        )
        
        # Validate model
        if not bench.validate_model_name(model):
            available = ", ".join(bench.AVAILABLE_MODELS)
            click.echo(f"Error: Model '{model}' not supported by CyBench.", err=True)
            click.echo(f"Available models: {available}", err=True)
            return
        
        click.echo(f"Starting CyBench evaluation for model: {model}")
        click.echo(f"This may take a while (up to 2 hours)...")
        
        # Run evaluation
        result = bench.run_evaluation(
            model_name=model,
            tasks=task_list,
            max_iterations=max_iterations,
            unguided_mode=unguided_mode
        )
        
        if result.success:
            click.echo(f"✅ Evaluation completed successfully!")
            click.echo(f"📊 Results: {result.summary_stats['successful_tasks']}/{result.summary_stats['total_tasks']} tasks successful")
            click.echo(f"📁 Detailed results saved to: {output_dir}")
        else:
            click.echo(f"❌ Evaluation failed: {result.error_message}", err=True)
            
    except ImportError as e:
        click.echo(f"Error: Could not import CyBenchBench: {e}", err=True)
    except Exception as e:
        click.echo(f"An unexpected error occurred: {e}", err=True)
        logger.error(f"Unexpected error during cybench benchmark: {e}", exc_info=True)

@cli_benchmark.command("nl2bash")
@click.option("--model", required=True, 
              help="Model to evaluate (e.g., 'openai/gpt-4', 'anthropic/claude-3-sonnet')")
@click.option("--output-dir", default="results/benchmarks/nl2bash", 
              help="Directory to save benchmark results.")
def cli_benchmark_nl2bash(model: str, output_dir: str):
    """Run NL2Bash benchmark evaluation on the specified model."""
    logger.info(f"CLI benchmark nl2bash initiated for model: {model}")
    
    try:
        from .datasets.nl2bash.bench import NL2BashBench
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize benchmark runner
        bench = NL2BashBench(output_dir=output_path)
        
        # Run evaluation
        result = bench.run_evaluation(model_name=model)
        
        if result.success:
            stats = result.summary_stats
            click.echo(f"✅ NL2Bash evaluation completed successfully!")
            click.echo(f"Tasks completed: {stats['total_tasks']}")
            click.echo(f"Successful tasks: {stats['successful_tasks']}")
            click.echo(f"Success rate: {stats['success_rate']:.1%}")
            click.echo(f"Average LLM score: {stats['average_llm_score']:.3f}")
            
            # Show complexity breakdown
            if stats.get('complexity_breakdown'):
                click.echo("\nComplexity breakdown:")
                for category, breakdown in stats['complexity_breakdown'].items():
                    click.echo(f"  {category}: {breakdown['successful']}/{breakdown['total']} "
                             f"({breakdown['success_rate']:.1%})")
            
            click.echo(f"Results saved to: {output_path}")
        else:
            click.echo(f"❌ NL2Bash evaluation failed: {result.error_message}", err=True)
            
    except ImportError as e:
        click.echo(f"Error: NL2Bash or inspect_ai module not available: {e}", err=True)
    except FileNotFoundError as e:
        click.echo(f"Error: {e}", err=True)
    except Exception as e:
        logger.error(f"Unexpected error during nl2bash benchmark: {e}", exc_info=True)
        click.echo(f"Unexpected error: {e}", err=True)

if __name__ == "__main__":
    # This makes the CLI runnable when you execute the script directly
    # e.g. python src/human_ttc_eval/cli.py parse ...
    # However, the standard way for packages is `python -m human_ttc_eval.cli parse ...`
    cli() 