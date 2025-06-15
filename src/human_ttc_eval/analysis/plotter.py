"""
Generate horizon plots using METR's methodology.

Wrapper around METR's logistic regression and plotting functions.
"""

import sys
import logging
from pathlib import Path
from typing import List, Optional
import yaml
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np

from .transform import transform_benchmark_results
from .. import config

# Add METR code to path
METR_PATH = config.THIRD_PARTY_DIR / "eval-analysis-public"
if str(METR_PATH) not in sys.path:
    sys.path.insert(0, str(METR_PATH))

# Import METR functions
try:
    from src.wrangle.logistic import run_logistic_regressions, WrangleParams
    from src.utils.logistic import logistic_regression, get_x_for_quantile
    from src.plot.logistic import plot_horizon_graph
    from src.plot.individual_histograms import plot_logistic_regression_on_histogram
    import src.utils.plots
except ImportError as e:
    raise ImportError(f"Failed to import METR functions from {METR_PATH}: {e}")

logger = logging.getLogger(__name__)


def create_horizon_plots(
    output_dir: Optional[Path] = None,
    success_rates: List[int] = None,
    dataset_filter: Optional[str] = None
) -> List[Path]:
    """
    Create METR-style horizon plots from all benchmark results.
    
    Uses standard locations from config:
    - Benchmark results: config.RESULTS_DIR / "benchmarks"
    - Human baselines: config.DATA_DIR / "processed" / <dataset> / <dataset>_tasks.jsonl
    - Models registry: src/human_ttc_eval/models.json
    
    Args:
        output_dir: Directory to save plots (default: config.RESULTS_DIR / "plots")
        success_rates: List of success rate percentages (default: [50])
        dataset_filter: Optional dataset name to filter results
        
    Returns:
        List of paths to generated plot files
    """
    if output_dir is None:
        output_dir = config.RESULTS_DIR / "plots"
    
    if success_rates is None:
        success_rates = [50]
    
    # Create output directories
    output_dir.mkdir(parents=True, exist_ok=True)
    metr_data_dir = output_dir / "metr_data"
    metr_data_dir.mkdir(exist_ok=True)
    
    logger.info("Transforming benchmark results to METR format...")
    
    # Transform our data to METR format
    files = transform_benchmark_results(metr_data_dir)
    
    all_runs_file = files['runs_file']
    release_dates_file = files['release_dates_file']
    
    logger.info("Generating horizon plots...")
    
    # Load the transformed data
    runs_df = pd.read_json(all_runs_file, lines=True, orient="records")
    
    # Apply dataset filter if specified
    if dataset_filter:
        runs_df = runs_df[runs_df['task_source'] == dataset_filter]
        logger.info(f"Filtered to {len(runs_df)} runs from dataset: {dataset_filter}")
    
    if runs_df.empty:
        logger.warning("No runs to plot after filtering")
        return []
    
    # Configure METR parameters
    wrangle_params = WrangleParams(
        runs_file=all_runs_file,
        weighting="weight", 
        categories=[],
        regularization=0.1,
        exclude=[],
        success_percents=success_rates,
        confidence_level=0.8
    )
    
    # Run METR's logistic regression
    logger.info(f"Running logistic regressions for {len(runs_df)} runs...")
    
    # Drop 'alias' column for METR regression to avoid conflicts
    runs_df_for_regression = runs_df.drop(columns=['alias'])
    
    regressions = run_logistic_regressions(
        runs=runs_df_for_regression,
        release_dates_file=release_dates_file,
        wrangle_params=wrangle_params,
        bootstrap_file=None,
        include_empirical_rates=True
    )
    
    # Save regression results  
    logistic_fits_file = metr_data_dir / "logistic_fits.csv"
    regressions.to_csv(logistic_fits_file, index=False)
    logger.info(f"Saved logistic fits to {logistic_fits_file}")
    
    # Generate individual histogram plots
    hist_output_dir = output_dir / "individual_histograms"
    _generate_individual_histograms(
        all_runs_file=all_runs_file,
        logistic_fits_file=logistic_fits_file,
        output_dir=hist_output_dir
    )
    logger.info(f"Individual histogram plots generated in {hist_output_dir}")
    
    # Generate task length distribution histogram
    dist_plot_file = _plot_task_length_distribution(
        all_runs_file=all_runs_file,
        output_dir=output_dir,
        dataset_filter=dataset_filter
    )
    plot_files = [dist_plot_file]

    # Convert release_date to proper datetime format
    regressions['release_date'] = pd.to_datetime(regressions['release_date'])
    
    # Load release dates
    with open(release_dates_file, 'r') as f:
        release_dates_dict = yaml.safe_load(f)
    
    # Create plots
    for success_rate in success_rates:
        plot_file = _create_single_horizon_plot(
            regressions=regressions,
            runs_df=runs_df,
            release_dates_dict=release_dates_dict,
            success_rate=success_rate,
            output_dir=output_dir,
            dataset_filter=dataset_filter
        )
        plot_files.append(plot_file)
        logger.info(f"Saved horizon plot to {plot_file}")
    
    logger.info(f"Generated {len(plot_files)} total plots")
    return plot_files


def _create_single_horizon_plot(
    regressions: pd.DataFrame,
    runs_df: pd.DataFrame,
    release_dates_dict: dict,
    success_rate: int,
    output_dir: Path,
    dataset_filter: Optional[str] = None
) -> Path:
    """Create a single horizon plot."""
    # Create plot configuration
    plot_params = _create_plot_params(regressions['agent'].unique())
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Determine title
    title = f'AI Agent Horizon at {success_rate}% Success Rate'
    if dataset_filter:
        title += f' ({dataset_filter})'
    
    # Call METR's plotting function
    plot_horizon_graph(
        plot_params=plot_params,
        all_agent_summaries=regressions,
        runs_df=runs_df,
        release_dates=release_dates_dict,
        lower_y_lim=0.01,  # ~30 seconds
        upper_y_lim=10000,  # ~7 days
        x_lim_start='2020-01-01',
        x_lim_end='2025-12-31',
        subtitle='',
        title=title,
        weight_key='weight',
        exclude_agents=[],
        success_percent=success_rate,
        confidence_level=0.8,
        fig=fig
    )
    
    # Add lines of fit
    for agent in regressions['agent'].unique():
        agent_df = regressions[regressions['agent'] == agent].sort_values('release_date')
        if not agent_df.empty and 'a' in agent_df.columns and 'b' in agent_df.columns:
            style = plot_params['agent_styling'].get(agent, plot_params['agent_styling']['default'])
            
            # Calculate horizon values on the fly
            horizon_values = agent_df.apply(
                lambda row: get_x_for_quantile(row['a'], row['b'], success_rate / 100),
                axis=1
            )
            
            ax.plot(
                agent_df['release_date'],
                horizon_values,
                color=style['lab_color'],
                linestyle='-',
                linewidth=2,
                zorder=5  # Below scatter points but above grid
            )

    # Re-draw legend to include lines
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, loc='upper left', bbox_to_anchor=(1, 1))

    # Save plot
    plot_file = output_dir / f"horizon_plot_p{success_rate}.png"
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    return plot_file


def _create_plot_params(unique_agents: list) -> dict:
    """Create plot parameters for METR functions."""
    # Base styling
    plot_params = {
        'scatter_styling': {
            'error_bar': {
                'color': 'grey', 'fmt': 'none', 'capsize': 2, 
                'alpha': 1, 'zorder': 9, 'linewidth': 1.5, 'capthick': 1.5
            },
            'grid': {'which': 'major', 'linestyle': '-', 'alpha': 0.2, 'color': 'grey'},
            'scatter': {'s': 150, 'edgecolor': 'black', 'linewidth': 0.5, 'zorder': 10}
        },
        'agent_styling': {
            'default': {'lab_color': 'blue', 'marker': 'o'}
        },
        'legend_order': list(unique_agents),
        'ax_label_fontsize': 14,
        'title_fontsize': 16,
        'annotation_fontsize': 12,
        'xlabelpad': 10,
        'ylabelpad': 10,
        'suptitle_fontsize': 14
    }
    
    # Agent-specific styling
    colors = ['#3e805f', '#e26e2f', '#4285f4', '#9333ea', '#f59e0b']
    markers = ['o', 's', '^', 'D', 'v']
    
    for i, agent in enumerate(unique_agents):
        plot_params['agent_styling'][agent] = {
            'lab_color': colors[i % len(colors)],
            'marker': markers[i % len(markers)]
        }
    
    return plot_params


def _generate_individual_histograms(
    all_runs_file: Path,
    logistic_fits_file: Path,
    output_dir: Path
) -> Path:
    """Generate individual histogram plots with logistic regression overlays."""
    logger.info("Generating individual histograms...")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "individual_histograms.png"
    
    # Load data
    all_runs_df = pd.read_json(all_runs_file, lines=True)
    agent_summaries_df = pd.read_csv(logistic_fits_file)
    
    # Ensure alias column exists
    if 'alias' not in all_runs_df.columns:
        logger.error("'alias' column not found in all_runs_df")
        return output_file
    
    focus_agents = agent_summaries_df["agent"].unique().tolist()
    if not focus_agents:
        logger.warning("No agents found in logistic_fits_file")
        return output_file
    
    # Create plot parameters
    plot_params = _create_histogram_plot_params(focus_agents)
    
    # Script parameters for METR function
    script_params = {
        'title': 'Individual Model Success Rates vs. Task Length',
        'n_subplot_cols': 3,
        'horizontal_lines': [{
            'p_success': 0.5,
            'styling': {
                'color': '#b30c00',
                'linestyle': 'dashed',
                'linewidth': 1.5,
                'alpha': 0.8
            }
        }],
        'annotate_p50': True,
        'exclude': [],
        'include_agents': focus_agents,
        'weighting': 'weight'
    }
    
    # Call METR's plotting function
    try:
        plot_logistic_regression_on_histogram(
            plot_params=plot_params,
            agent_summaries=agent_summaries_df,
            all_runs=all_runs_df,
            focus_agents=focus_agents,
            output_file=output_file,
            script_params=script_params
        )
        logger.info(f"Successfully generated individual histogram plot to {output_file}")
    except Exception as e:
        logger.error(f"Error generating individual histogram plot: {e}", exc_info=True)
    
    return output_file


def _create_histogram_plot_params(focus_agents: list) -> dict:
    """Create plot parameters for histogram plots."""
    base_colors = ['#3e805f', '#e26e2f', '#4285f4', '#9333ea', '#f59e0b']
    base_markers = ['o', 's', '^', 'D', 'v']
    
    agent_styling = {}
    for i, agent_name in enumerate(focus_agents):
        color = base_colors[i % len(base_colors)]
        agent_styling[agent_name] = {
            'color': color,
            'marker': base_markers[i % len(base_markers)],
            'unique_color': color
        }
    
    return {
        'scatter_styling': {
            'error_bar': {
                'color': 'grey', 'fmt': 'none', 'capsize': 2,
                'alpha': 1, 'zorder': 9, 'linewidth': 1.5, 'capthick': 1.5
            },
            'grid': {'which': 'major', 'linestyle': '-', 'alpha': 0.2, 'color': 'grey'},
            'scatter': {'s': 150, 'edgecolor': 'black', 'linewidth': 0.5, 'zorder': 10}
        },
        'agent_styling': {
            'default': {'color': 'blue', 'marker': 'o', 'unique_color': 'blue'},
            **agent_styling
        },
        'legend_order': focus_agents,
        'ax_label_fontsize': 12,
        'title_fontsize': 14,
        'annotation_fontsize': 10,
        'xlabelpad': 8,
        'ylabelpad': 8,
        'suptitle_fontsize': 16,
        'colors': base_colors,
        'markers': base_markers
    }


def _plot_task_length_distribution(
    all_runs_file: Path,
    output_dir: Path,
    dataset_filter: Optional[str] = None
) -> Path:
    """Plots a stacked histogram of task lengths by dataset."""
    logger.info("Generating task length distribution histogram...")
    output_file = output_dir / "task_length_distribution.png"
    
    # Load data and get unique tasks
    runs_df = pd.read_json(all_runs_file, lines=True)
    if dataset_filter:
        runs_df = runs_df[runs_df['task_source'] == dataset_filter]
    tasks_df = runs_df.drop_duplicates(subset=['task_id'])
    
    if tasks_df.empty:
        logger.warning("No tasks found for distribution plot.")
        return output_file
        
    # Prepare data for stacked histogram
    datasets = sorted(tasks_df['task_source'].unique())
    data_to_plot = [tasks_df[tasks_df['task_source'] == ds]['human_minutes'] for ds in datasets]
    colors = ['#4c72b0', '#dd8452', '#55a868', '#c44e52', '#8c564b', '#9467bd', '#e377c2']
    
    # Create plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Use logarithmic bins
    min_time = tasks_df['human_minutes'].min()
    max_time = tasks_df['human_minutes'].max()
    if min_time <= 0: min_time = 1/60  # Start at 1 second
    
    bins = np.logspace(np.log10(min_time), np.log10(max_time), 30)
    
    # Plot stacked histogram
    ax.hist(data_to_plot, bins=bins, stacked=True, label=datasets, color=colors[:len(datasets)], edgecolor='white')
    
    # Formatting
    ax.set_xscale('log')
    ax.set_xlabel('Human Time-to-Complete')
    ax.set_ylabel('Number of Tasks')
    ax.set_title('Distribution of Task Lengths by Dataset')
    
    # Use custom formatter for x-axis ticks
    def time_formatter(y, pos):
        if y < 1:
            return f'{(y * 60):.0f} sec'
        elif y < 60:
            return f'{y:.0f} min'
        else:
            return f'{y/60:.1f} hr'

    ax.xaxis.set_major_formatter(FuncFormatter(time_formatter))
    
    ax.legend(title="Dataset")
    ax.grid(True, which='major', axis='y', linestyle='--', alpha=0.6)
    
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close(fig)
    logger.info(f"Saved task length distribution plot to {output_file}")
    
    return output_file 