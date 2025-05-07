import click
from pathlib import Path
import logging
import sys
from typing import Optional

# Ensure the src directory is in the Python path for module resolution
# This is often needed if running cli.py directly during development without full package installation
# For python -m human_ttc_eval.cli, this might not be strictly necessary
# but doesn't hurt for robustness if structure changes slightly.
# current_dir = Path(__file__).resolve().parent
# src_dir = current_dir.parent # This would be human_ttc_eval
# sys.path.insert(0, str(src_dir.parent)) # This would be src/

# Import dataset modules to ensure they register themselves
# These imports must happen before get_parser/get_summariser are called by Click
# when the CLI commands are being set up.
try:
    from .datasets.kypo import parser as kypo_parser_module # noqa
    from .datasets.kypo import summariser as kypo_summariser_module # noqa
    # Add imports for other datasets here as they are created, e.g.:
    # from .datasets.another_dataset import parser as another_parser_module # noqa
except ImportError as e:
    # This fallback is for making the CLI runnable for basic commands like --help
    # even if dataset-specific dependencies are missing, though parse/summarise will fail.
    print(f"Warning: Could not import all dataset modules: {e}. Some commands might fail.", file=sys.stderr)

from .core.registry import get_parser, list_parsers, get_summariser, list_summarisers
# from .core.utils import slugify # Not directly used in CLI logic itself

# Configure basic logging for the CLI
# Match the format from core.utils for consistency if desired, or keep simple for CLI.
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(name)s - %(message)s')
logger = logging.getLogger("human_ttc_eval.cli") # Explicit logger name

@click.group()
def cli():
    """Human-TTC-Eval: A CLI tool for parsing and summarizing datasets."""
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
@click.option("--data-root-relative", 
              type=click.Path(exists=True, file_okay=False, dir_okay=True, readable=True, resolve_path=True), 
              help="Optional: Base path for calculating relative file paths within logs (e.g., for KYPO's _raw_file_path). Defaults to input-dir.")
def cli_parse(dataset_name: str, input_dir: str, output_file: str, data_root_relative: Optional[str]):
    """Parses a specified dataset and writes output to a JSONL file."""
    logger.info(f"CLI parse command initiated for dataset: '{dataset_name}'")
    try:
        parser_class = get_parser(dataset_name)
        input_path = Path(input_dir)
        output_path = Path(output_file)
        
        data_root_for_rel_paths_path = Path(data_root_relative) if data_root_relative else input_path

        # Specific instantiation for KypoParser if needed, or general approach:
        if hasattr(parser_class, "data_root_for_relative_paths") or dataset_name == "kypo": # A bit heuristic
            parser_instance = parser_class(
                input_dir=input_path, 
                output_file=output_path,
                data_root_for_relative_paths=data_root_for_rel_paths_path
            )
            logger.info(f"Instantiated {dataset_name} parser with data_root_for_relative_paths: {data_root_for_rel_paths_path}")
        else:
            parser_instance = parser_class(input_dir=input_path, output_file=output_path)
            logger.info(f"Instantiated {dataset_name} parser with default constructor.")

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

if __name__ == "__main__":
    # This makes the CLI runnable when you execute the script directly
    # e.g. python src/human_ttc_eval/cli.py parse ...
    # However, the standard way for packages is `python -m human_ttc_eval.cli parse ...`
    cli() 