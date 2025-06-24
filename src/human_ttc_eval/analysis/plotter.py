"""
Generate horizon plots using METR's methodology.

Wrapper around METR's logistic regression and plotting functions.
"""

import sys
import logging
from pathlib import Path
from typing import List, Optional, Dict, Any
import yaml
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np
import warnings
from matplotlib.axes import Axes
from matplotlib.dates import date2num
from sklearn.metrics import r2_score
from sklearn.linear_model import LogisticRegression, LinearRegression

from .transform import transform_benchmark_results
from .token_analysis import extract_tokens_from_eval_logs, create_token_plots
from ..core.run import Run
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
    dataset_filter: Optional[str] = None,
    n_bootstraps: int = 500,
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
        n_bootstraps: Number of bootstrap iterations for confidence intervals
        
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
    
    # Create a mapping of agent (full name) to alias for later use
    agent_to_alias_map = runs_df[['agent', 'alias']].drop_duplicates().set_index('agent')['alias'].to_dict()
    logger.info(f"Agent to alias mapping: {agent_to_alias_map}")
    
    # Check for any missing aliases and warn
    unique_agents = runs_df['agent'].unique()
    missing_alias_agents = [agent for agent in unique_agents if agent not in agent_to_alias_map]
    if missing_alias_agents:
        logger.debug(f"Agents without alias mapping: {missing_alias_agents}")
    
    # Replace agent names with aliases in runs_df for METR functions
    # Use a more defensive mapping that ensures we never lose agents
    runs_df_for_metr = runs_df.copy()
    def safe_alias_map(agent_name):
        alias = agent_to_alias_map.get(agent_name)
        if alias is None or alias == agent_name:
            # If no alias or alias is same as agent name, use agent name
            return agent_name
        return alias
    
    runs_df_for_metr['agent'] = runs_df_for_metr['agent'].map(safe_alias_map)
    
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
    runs_df_for_regression = runs_df_for_metr.drop(columns=['alias'])
    
    regressions = run_logistic_regressions(
        runs=runs_df_for_regression,
        release_dates_file=release_dates_file,
        wrangle_params=wrangle_params,
        bootstrap_file=None,
        include_empirical_rates=True
    )
    
    # The regressions dataframe now has aliases as agent names
    # We need to keep track of this for consistency
    
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

    # Load release dates first
    with open(release_dates_file, 'r') as f:
        release_dates_dict = yaml.safe_load(f)
    
    # Add release dates to regressions using alias mapping
    for i, row in regressions.iterrows():
        agent_alias = row['agent']
        # Find the full model name that corresponds to this alias
        full_name = None
        for full_model_name, alias in agent_to_alias_map.items():
            if alias == agent_alias:
                full_name = full_model_name
                break
        
        if full_name and full_name in release_dates_dict.get('date', {}):
            regressions.at[i, 'release_date'] = release_dates_dict['date'][full_name]
        else:
            logger.warning(f"No release date found for agent {agent_alias} (full name: {full_name})")
    
    # Convert release_date to proper datetime format
    regressions['release_date'] = pd.to_datetime(regressions['release_date'])
    
    # Also create a version of release_dates_dict with aliases as keys for METR
    release_dates_dict_aliases = {'date': {}}
    if 'date' in release_dates_dict:
        for agent, date in release_dates_dict['date'].items():
            alias = agent_to_alias_map.get(agent, agent)
            release_dates_dict_aliases['date'][alias] = date
    
    # Perform bootstrapping to generate data for confidence intervals
    bootstrap_results_file = _perform_bootstrapping(
        all_runs_file=all_runs_file,
        output_dir=output_dir,
        n_bootstraps=n_bootstraps,
        agent_to_alias_map=agent_to_alias_map,
    )

    # Create plots
    for success_rate in success_rates:
        # Create combined plot
        plot_file = _create_single_horizon_plot(
            regressions=regressions,
            runs_df=runs_df_for_metr,  # Use the version with aliases
            release_dates_dict=release_dates_dict_aliases,  # Use aliased version
            success_rate=success_rate,
            output_dir=output_dir,
            bootstrap_results_file=bootstrap_results_file,
            agent_to_alias_map=agent_to_alias_map,
        )
        plot_files.append(plot_file)
        logger.info(f"Saved horizon plot to {plot_file}")
        
        # Create plots for each dataset filter
        if dataset_filter:
            filtered_regressions = regressions[regressions['task_source'] == dataset_filter]
            filtered_runs_df = runs_df_for_metr[runs_df_for_metr['task_source'] == dataset_filter]
            if not filtered_regressions.empty:
                plot_files.append(
                    _create_single_horizon_plot(
                        regressions=filtered_regressions,
                        runs_df=filtered_runs_df,
                        release_dates_dict=release_dates_dict_aliases,  # Use aliased version
                        success_rate=success_rate,
                        output_dir=output_dir,
                        dataset_filter=dataset_filter,
                        bootstrap_results_file=bootstrap_results_file,
                        agent_to_alias_map=agent_to_alias_map,
                    )
                )
                logger.info(f"Saved horizon plot to {plot_files[-1]}")

    logger.info(f"Generated {len(plot_files)} total plots")
    
    # Generate token analysis plots
    token_plot_files = _generate_token_plots(runs_df, output_dir)
    plot_files.extend(token_plot_files)
    
    logger.info(f"Generated {len(plot_files)} total plots (including token analysis)")
    return plot_files


def _create_single_horizon_plot(
    regressions: pd.DataFrame,
    runs_df: pd.DataFrame,
    release_dates_dict: dict,
    success_rate: int,
    output_dir: Path,
    dataset_filter: Optional[str] = None,
    bootstrap_results_file: Optional[Path] = None,
    agent_to_alias_map: Optional[Dict[str, str]] = None,
) -> Path:
    """Create a single horizon plot."""
    # Determine y-axis limits based on actual data to ensure all models are included
    p_col = f'p{success_rate}'
    p50_values = regressions[p_col].dropna()
    if not p50_values.empty:
        min_p50 = p50_values.min()
        max_p50 = p50_values.max()
        logger.info(f"P50 range: {min_p50:.8f} to {max_p50:.8f}")
        
        # Set fixed y-axis limits as requested: 0 to 4 hours (240 minutes)
        # Lower limit must include GPT-2's value of 6.2e-05 (0.000062)
        lower_y_lim = 1e-2   # 0.000001 minutes (0.06 seconds) to include GPT-2
        upper_y_lim = 240    # 4 hours in minutes
        
        logger.info(f"Calculated y-axis limits: {lower_y_lim:.8f} to {upper_y_lim:.8f}")
    else:
        # Fallback values
        lower_y_lim = 1e-2
        upper_y_lim = 1000
    
    # Get all agents from regressions - let METR handle which ones to display
    all_agents = regressions['agent'].unique().tolist()
    logger.info(f"All agents in regression data: {all_agents}")
    
    # Create plot configuration with all agents
    plot_params = _create_plot_params(all_agents)
    
    # Verify all agents in regressions have styling
    missing_styling = []
    for agent in regressions['agent'].unique():
        if agent not in plot_params['agent_styling']:
            missing_styling.append(agent)
    
    if missing_styling:
        logger.warning(f"Agents missing styling: {missing_styling}")
    else:
        logger.info("All agents have styling configured")
    
    # Debug logging
    logger.debug(f"Agents in regressions: {sorted(regressions['agent'].unique())}")
    logger.debug(f"Agents in runs_df: {sorted(runs_df['agent'].unique())}")  
    logger.debug(f"Agents in plot_params['agent_styling']: {sorted(plot_params['agent_styling'].keys())}")
    logger.debug(f"Agents in plot_params['legend_order']: {plot_params['legend_order']}")
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Determine title
    title = f'Model Offensive Cybersecurity Horizon at {success_rate}% Success Rate'
    if dataset_filter:
        title += f' ({dataset_filter})'
    
    logger.info(f"Using y-axis limits: {lower_y_lim:.4f} to {upper_y_lim:.1f} minutes")
    
    # Call METR's plotting function
    plot_horizon_graph(
        plot_params=plot_params,
        all_agent_summaries=regressions,
        runs_df=runs_df,
        release_dates=release_dates_dict,
        lower_y_lim=lower_y_lim,
        upper_y_lim=upper_y_lim,
        x_lim_start='2019-01-01',
        x_lim_end='2026-01-01',
        subtitle='',
        title=title,
        weight_key='weight',
        exclude_agents=[],
        success_percent=success_rate,
        confidence_level=0.8,
        fig=fig
    )

    # Update SOTA models list to use aliases
    sota_models_full_names = [
        'google/gemini-2.5-pro-preview-06-05',
        'openai/davinci-002',
        'anthropic/claude-3-5-sonnet-20240620',
        'openai/gpt-3.5-turbo',
    ]
    
    if agent_to_alias_map:
        sota_models_for_fit = [agent_to_alias_map.get(m, m) for m in sota_models_full_names]
    else:
        sota_models_for_fit = sota_models_full_names
    
    # Get axis limits from the existing plot for trendline and CI
    x_lim_dates = ax.get_xlim()
    ax_min_date_num, ax_max_date_num = x_lim_dates
    ax_min_date = pd.to_datetime(ax_min_date_num, unit='D')
    ax_max_date = pd.to_datetime(ax_max_date_num, unit='D')

    # Add a line of best fit for selected models using METR's functions
    regressions_for_fit = regressions[regressions['agent'].isin(sota_models_for_fit)].copy()
    
    if len(regressions_for_fit) > 1:
        reg, score = fit_trendline(
            regressions_for_fit[f'p{success_rate}'],
            pd.to_datetime(regressions_for_fit['release_date']),
            log_scale=True
        )
        if reg:
            # Get the date range for the trendline
            fit_min_date = pd.to_datetime(regressions_for_fit['release_date']).min()
            fit_max_date = pd.to_datetime(regressions_for_fit['release_date']).max()
            
            # Debug: Show main trendline details
            logger.info(f"Main trendline calculated from {len(regressions_for_fit)} SOTA models")
            logger.info(f"SOTA models used: {regressions_for_fit['agent'].tolist()}")
            logger.info(f"Main p50 values: {regressions_for_fit[f'p{success_rate}'].tolist()}")
            logger.info(f"Main trendline slope: {reg.coef_[0]:.6f}")
            logger.info(f"Main trendline doubling time: {np.log(2) / reg.coef_[0]:.1f} days")
            
            # Add bootstrap confidence interval for the trendline
            if bootstrap_results_file and bootstrap_results_file.exists():
                # Use the new proper bootstrap method
                _add_bootstrap_confidence_region_for_trendline(
                    ax=ax,
                    sota_p50s=regressions_for_fit[f'p{success_rate}'].tolist(),
                    sota_dates=regressions_for_fit['release_date'].tolist(),
                    confidence_level=0.95,
                    n_bootstraps=1000
                )

            # Plot the main trendline on top of the CI
            plot_trendline(
                ax=ax,
                reg=reg,
                score=score,
                line_start_date=fit_min_date.strftime('%Y-%m-%d'),
                line_end_date=fit_max_date.strftime('%Y-%m-%d'),
                log_scale=True
            )

    # Legend is now handled by METR's fixed legend creation logic

    # Add title and save
    ax.set_title(title, fontsize=16)
    fig.tight_layout()
    output_file = output_dir / f"horizon_plot_p{success_rate}{f'_{dataset_filter}' if dataset_filter else ''}.png"
    fig.savefig(output_file)
    logger.info(f"Saved horizon plot to {output_file}")
    plt.close(fig)
    return output_file


def _create_plot_params(agents: List[str]) -> Dict:
    """
    Create plot parameters compatible with METR's plotting functions,
    but with expanded styling to support more agents.
    """
    # Convert to list and ensure it's not empty
    agents_list = list(agents) if agents is not None else []
    
    # Sort agents chronologically for legend order
    chronological_order = [
        'GPT 2',  # 2019-11-05
        'GPT 3',  # 2020-07-11
        'GPT 3.5',  # 2022-03-15
        'Claude 3.5 Sonnet (June 2024)',  # 2024-06-20
        'Claude 3.5 Haiku',  # 2024-10-22
        'Claude 3.5 Sonnet (Oct 2024)',  # 2024-10-22
        'O3',  # 2025-04-16
        'O4 Mini',  # 2025-04-16
        'Gemini 2.5 Pro (June 2025)',  # 2025-06-05
    ]
    
    # Create chronologically sorted legend order
    sorted_agents = []
    for agent in chronological_order:
        if agent in agents_list:
            sorted_agents.append(agent)
    
    # Add any agents not in the predefined order at the end
    for agent in agents_list:
        if agent not in sorted_agents:
            sorted_agents.append(agent)
    
    # Base styling from METR's defaults
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
        'legend_order': sorted_agents,
        'ax_label_fontsize': 14,
        'title_fontsize': 16,
        'annotation_fontsize': 12,
        'xlabelpad': 10,
        'ylabelpad': 10,
        'suptitle_fontsize': 14
    }

    # Expanded colors and markers to handle many agents
    colors = [
        '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
        '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
        '#aec7e8', '#ffbb78', '#98df8a', '#ff9896', '#c5b0d5',
        '#c49c94', '#f7b6d2', '#c7c7c7', '#dbdb8d', '#9edae5'
    ]
    markers = [
        'o', 'v', '^', '<', '>', 's', 'p', '*', 'h', 'H',
        'D', 'd', 'P', 'X', 'o', 'v', '^', '<', '>', 's'
    ]
    
    # Assign a unique color and marker to each agent (in chronological order)
    for i, agent in enumerate(sorted_agents):
        plot_params['agent_styling'][agent] = {
            'lab_color': colors[i % len(colors)],
            'marker': markers[i % len(markers)]
        }
        logger.debug(f"Agent {agent}: color={colors[i % len(colors)]}, marker={markers[i % len(markers)]}")
        
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
    
    # Create agent to alias mapping
    if 'alias' in all_runs_df.columns:
        agent_to_alias = all_runs_df[['agent', 'alias']].drop_duplicates().set_index('agent')['alias'].to_dict()
        
        # Replace agent names with aliases in both dataframes for consistency
        all_runs_df['agent'] = all_runs_df['agent'].map(lambda x: agent_to_alias.get(x, x))
        # Note: agent_summaries_df already has aliases as 'agent' from the regression step
    else:
        logger.warning("'alias' column not found in all_runs_df, using agent names as-is")
    
    # Sort agents by release date for consistent plot ordering
    # Define the chronological order manually based on known release dates
    chronological_order = [
        'GPT 2',  # 2019-11-05
        'GPT 3',  # 2020-07-11
        'GPT 3.5',  # 2022-03-15
        'Claude 3.5 Sonnet (June 2024)',  # 2024-06-20
        'Claude 3.5 Haiku',  # 2024-10-22
        'Claude 3.5 Sonnet (Oct 2024)',  # 2024-10-22
        'O3',  # 2025-04-16
        'O4 Mini',  # 2025-04-16
        'Gemini 2.5 Pro (June 2025)',  # 2025-06-05
    ]
    
    # Get available agents
    available_agents = agent_summaries_df["agent"].unique().tolist()
    
    # Order agents chronologically, putting any new agents at the end
    focus_agents = []
    for agent in chronological_order:
        if agent in available_agents:
            focus_agents.append(agent)
    
    # Add any agents not in the predefined order at the end
    for agent in available_agents:
        if agent not in focus_agents:
            focus_agents.append(agent)
    
    logger.info(f"Ordered agents chronologically: {focus_agents}")
        
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


def _generate_token_plots(runs_df: pd.DataFrame, output_dir: Path) -> List[Path]:
    """Generate token analysis plots from inspect_ai logs."""
    logger.info("Starting token analysis...")
    
    # Create token analysis output directory
    token_output_dir = output_dir / "token_analysis"
    token_output_dir.mkdir(exist_ok=True)
    
    # Convert runs_df to Run objects for token analysis
    runs_data = []
    for _, row in runs_df.iterrows():
        run = Run(
            task_id=row['task_id'],
            task_family=row.get('task_family', row.get('task_source', 'unknown')),
            run_id=row.get('run_id', f"{row.get('agent', 'unknown')}_{row['task_id']}"),
            alias=row.get('alias', row.get('agent', 'unknown')),
            model=row.get('model', row.get('agent', 'unknown')),
            score_binarized=int(row['score_binarized']),
            human_minutes=float(row['human_minutes']),
            score_cont=row.get('score_cont', float(row['score_binarized'])),
            human_source=row.get('human_source', 'baseline'),
            task_source=row.get('task_source', 'unknown')
        )
        runs_data.append(run)
    
    # Find inspect_ai log directories automatically
    all_token_data = {}
    benchmarks_dir = config.RESULTS_DIR / "benchmarks"
    
    if benchmarks_dir.exists():
        for dataset_dir in benchmarks_dir.iterdir():
            if dataset_dir.is_dir():
                inspect_logs_dir = dataset_dir / "inspect_logs"
                if inspect_logs_dir.exists():
                    logger.info(f"Extracting tokens from {inspect_logs_dir}")
                    dataset_tokens = extract_tokens_from_eval_logs(inspect_logs_dir)
                    all_token_data.update(dataset_tokens)
    
    if not all_token_data:
        logger.info("No token data found - skipping token analysis plots")
        return []
    
    # Generate token analysis plots
    create_token_plots(runs_data, all_token_data, token_output_dir)
    
    # Return list of generated token plot files
    token_plot_files = []
    for plot_file in token_output_dir.glob("*.png"):
        token_plot_files.append(plot_file)
    
    logger.info(f"Generated {len(token_plot_files)} token analysis plots")
    return token_plot_files


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


def _perform_bootstrapping(
    all_runs_file: Path,
    output_dir: Path,
    n_bootstraps: int,
    agent_to_alias_map: Optional[Dict[str, str]] = None
) -> Path:
    """
    Performs bootstrapping on task results to generate a distribution of P50 horizons.
    """
    logger.info(f"Performing {n_bootstraps} bootstrap iterations for CI...")
    output_dir.mkdir(parents=True, exist_ok=True)
    bootstrap_file = output_dir / "bootstrap_results.csv"

    if bootstrap_file.exists():
        logger.info("Bootstrap results already exist, skipping.")
        return bootstrap_file

    all_runs_df = pd.read_json(all_runs_file, lines=True)
    
    # Replace agent names with aliases if mapping provided
    if agent_to_alias_map:
        all_runs_df['agent'] = all_runs_df['agent'].map(lambda x: agent_to_alias_map.get(x, x))
    
    # Get unique tasks to resample from
    unique_tasks = all_runs_df['task_id'].unique()
    n_tasks = len(unique_tasks)
    
    agents = sorted(all_runs_df['agent'].unique())
    bootstrap_p50s = []

    for i in range(n_bootstraps):
        if (i + 1) % 50 == 0:
            logger.info(f"  Bootstrap iteration {i+1}/{n_bootstraps}...")
            
        # Resample tasks with replacement
        resampled_task_ids = np.random.choice(unique_tasks, size=n_tasks, replace=True)
        
        # Create a new dataframe with only the resampled tasks
        resampled_df = pd.DataFrame({'task_id': resampled_task_ids})
        bootstrap_sample_df = pd.merge(resampled_df, all_runs_df, on='task_id', how='left')

        # Calculate logistic fits for this bootstrap sample
        p50_results_for_iter = {}
        for agent in agents:
            agent_df = bootstrap_sample_df[bootstrap_sample_df['agent'] == agent]
            
            if len(agent_df) < 2 or agent_df['score_binarized'].nunique() < 2:
                p50_results_for_iter[f"{agent}_p50"] = np.nan
                continue

            try:
                X = np.log2(agent_df['human_minutes']).values.reshape(-1, 1)
                y = agent_df['score_binarized']
                weights = agent_df.get('weight', None)

                model = LogisticRegression(C=1.0, class_weight='balanced')
                model.fit(X, y, sample_weight=weights)

                coef = model.coef_[0][0]
                intercept = model.intercept_[0]
                
                if coef != 0:
                    p50_log2 = -intercept / coef
                    p50 = np.exp2(p50_log2)
                    p50_results_for_iter[f"{agent}_p50"] = p50
                else:
                    p50_results_for_iter[f"{agent}_p50"] = np.nan

            except Exception:
                p50_results_for_iter[f"{agent}_p50"] = np.nan
                
        bootstrap_p50s.append(p50_results_for_iter)

    bootstrap_df = pd.DataFrame(bootstrap_p50s)
    bootstrap_df.to_csv(bootstrap_file, index=False)
    logger.info(f"Bootstrap results saved to {bootstrap_file}")
    
    return bootstrap_file


def _add_bootstrap_confidence_region_for_trendline(
    ax: plt.Axes,
    sota_p50s: list,
    sota_dates: list,
    confidence_level: float,
    n_bootstraps: int = 1000
):
    """
    Calculate and plot a bootstrap confidence region for the trendline
    by resampling the SOTA models with replacement.
    """
    logger.info("Adding bootstrap confidence region for trendline...")
    
    # Convert to arrays
    p50s = np.array(sota_p50s)
    dates = pd.to_datetime(sota_dates)
    n_models = len(p50s)
    
    logger.info(f"Bootstrapping trendline from {n_models} SOTA models")
    logger.info(f"SOTA p50s: {p50s}")
    logger.info(f"SOTA dates: {[d.strftime('%Y-%m-%d') for d in dates]}")
    logger.info(f"Date range: {dates.min()} to {dates.max()}")
    
    # Create time points for prediction
    time_points = pd.date_range(
        start=dates.min(),
        end=dates.max(),
        periods=100,
    )
    
    predictions = np.zeros((n_bootstraps, len(time_points)))
    slopes = []
    
    for i in range(n_bootstraps):
        # Resample models with replacement
        indices = np.random.choice(n_models, size=n_models, replace=True)
        resampled_p50s = p50s[indices]
        resampled_dates = dates[indices]
        
        if i < 5:  # Log first few resamplings
            logger.info(f"  Bootstrap {i}: indices={indices}, p50s={resampled_p50s}")
        
        try:
            # Fit trendline to resampled data
            log_y = np.log2(resampled_p50s)
            X = date2num(resampled_dates).reshape(-1, 1)
            
            model = LinearRegression()
            model.fit(X, log_y)
            slopes.append(model.coef_[0])
            
            # Predict over time range
            time_x = date2num(time_points).reshape(-1, 1)
            pred_log = model.predict(time_x)
            predictions[i, :] = np.exp2(pred_log)
            
        except Exception as e:
            predictions[i, :] = np.nan
    
    logger.info(f"Slope range: {np.min(slopes):.6f} to {np.max(slopes):.6f}")
    logger.info(f"Slope std dev: {np.std(slopes):.6f}")
    
    # Calculate confidence bounds
    low_q = (1 - confidence_level) / 2
    high_q = 1 - low_q
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        lower_bound = np.nanpercentile(predictions, low_q * 100, axis=0)
        upper_bound = np.nanpercentile(predictions, high_q * 100, axis=0)
    
    # Plot confidence region
    ax.fill_between(
        time_points,
        lower_bound,
        upper_bound,
        color="gray",
        alpha=0.3,
        zorder=5
    )
    
    logger.info("Trendline confidence band plotting complete")


# =====================================================================================
# METR eval-analysis-public code adapted for our plotter
# Source: third-party/eval-analysis-public/src/plot/logistic.py
# =====================================================================================

def fit_trendline(
    p50s: pd.Series, release_dates: pd.Series, log_scale: bool
) -> tuple[Any, float]:
    """Fits an exponential trendline to p50s over time."""
    valid_data = pd.DataFrame({"p50": p50s, "release_date": release_dates}).dropna()
    if len(valid_data) < 2:
        return None, 0

    y = np.log2(valid_data["p50"]) if log_scale else valid_data["p50"]
    X = date2num(valid_data["release_date"]).reshape(-1, 1)

    model = LinearRegression()
    model.fit(X, y)

    predictions = model.predict(X)
    score = r2_score(y, predictions)
    return model, score


def plot_trendline(
    ax: Axes,
    reg: Any,
    score: float,
    line_start_date: str,
    line_end_date: str,
    log_scale: bool,
):
    """Plots a trendline with its doubling time and R^2 score."""
    start_num = date2num(pd.to_datetime(line_start_date))
    end_num = date2num(pd.to_datetime(line_end_date))
    
    trend_x = np.linspace(start_num, end_num, 200)
    trend_y_log = reg.predict(trend_x.reshape(-1, 1))
    trend_y = np.exp2(trend_y_log) if log_scale else trend_y_log
    
    ax.plot(trend_x, trend_y, "--", color="red", zorder=3)

    # Annotation
    slope = reg.coef_[0]
    doubling_time_days = (np.log(2) / slope) if slope > 0 else 0
    doubling_time_months = doubling_time_days / 30.44  # Average days in a month
    
    annotation = f"Doubling time: {doubling_time_months:.1f} months\nR²: {score:.2f}"
    
    ax.text(
        0.95, 0.05, annotation,
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment='bottom',
        horizontalalignment='right'
    ) 