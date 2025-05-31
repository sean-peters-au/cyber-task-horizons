"""
Generate horizon plots using METR's methodology.

Wrapper around METR's logistic regression and plotting functions.
"""

import sys
import logging
from pathlib import Path
from typing import List, Any
import yaml
import pandas as pd
import json

logger = logging.getLogger(__name__)

def _import_metr_functions():
    """Import METR functions with careful dependency handling."""
    try:
        # Add METR code to path
        metr_path = Path(__file__).parent.parent.parent.parent / "third-party" / "eval-analysis-public"
        if not metr_path.exists():
            raise ImportError(f"METR code not found at {metr_path}")
        
        # Add to Python path
        if str(metr_path) not in sys.path:
            sys.path.insert(0, str(metr_path))
        
        # Import METR functions
        try:
            from src.wrangle.logistic import run_logistic_regressions, WrangleParams
            from src.utils.logistic import logistic_regression, get_x_for_quantile
            from src.plot.logistic import plot_horizon_graph
            from src.plot.individual_histograms import plot_logistic_regression_on_histogram
            import src.utils.plots
            
            return {
                'run_logistic_regressions': run_logistic_regressions,
                'WrangleParams': WrangleParams,
                'logistic_regression': logistic_regression,
                'get_x_for_quantile': get_x_for_quantile,
                'plot_horizon_graph': plot_horizon_graph,
                'plot_logistic_regression_on_histogram': plot_logistic_regression_on_histogram,
                'src_utils_plots': src.utils.plots
            }
            
        except ImportError as e:
            raise ImportError(f"Failed to import METR functions: {e}")
                
    except Exception as e:
        raise ImportError(f"Failed to import METR functions from {metr_path}: {e}")

def create_horizon_plots_from_benchmarks(
    benchmark_results_dir: Path,
    human_baselines_file: Path, 
    models_registry_file: Path,
    output_dir: Path,
    success_rates: List[int] = [50]
) -> List[Path]:
    """
    Create METR-style horizon plots from benchmark results using exact METR methodology.
    
    Args:
        benchmark_results_dir: Directory containing benchmark JSON files
        human_baselines_file: Path to human baseline timings JSONL file  
        models_registry_file: Path to models registry JSON file
        output_dir: Directory to save plots
        success_rates: List of success rate percentages (e.g., [50, 80])
        
    Returns:
        List of paths to generated plot files
    """
    from .transform import transform_benchmark_results
    
    # Import METR functions
    try:
        metr_funcs = _import_metr_functions()
    except ImportError as e:
        logger.error(f"Could not import METR functions: {e}")
        raise
    
    # Create output directories
    output_dir.mkdir(parents=True, exist_ok=True)
    metr_data_dir = output_dir / "metr_data"
    metr_data_dir.mkdir(exist_ok=True)
    
    logger.info("Transforming benchmark results to METR format...")
    
    # Transform our data to METR format
    files = transform_benchmark_results(
        results_dir=benchmark_results_dir,
        human_baselines_file=human_baselines_file,
        models_registry_file=models_registry_file,
        output_dir=metr_data_dir
    )
    
    all_runs_file = files['runs_file']
    release_dates_file = files['release_dates_file']
    
    logger.info("Generating horizon plots...")
    
    # Load the transformed data
    runs_df = pd.read_json(all_runs_file, lines=True, orient="records")
    
    # Configure METR parameters
    wrangle_params = metr_funcs['WrangleParams'](
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

    # Prepare a version of the DataFrame specifically for run_logistic_regressions
    # by dropping the 'alias' column to avoid issues with METR's internal rename.
    runs_df_for_metr_regression = runs_df.drop(columns=['alias'])

    regressions = metr_funcs['run_logistic_regressions'](
        runs=runs_df_for_metr_regression,
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
    generate_individual_histograms(
        all_runs_file=all_runs_file,
        logistic_fits_file=logistic_fits_file,
        models_registry_file=models_registry_file,
        output_dir=hist_output_dir,
        metr_plot_utils_module=metr_funcs['src_utils_plots']
    )
    logger.info(f"Individual histogram plots generated in {hist_output_dir}")
    
    # Convert release_date to proper datetime format for METR function
    regressions['release_date'] = pd.to_datetime(regressions['release_date'])
    
    # Load release dates for METR function
    with open(release_dates_file, 'r') as f:
        release_dates_dict = yaml.safe_load(f)
    
    # Create basic plot configuration matching METR's style
    plot_params = {
        'scatter_styling': {
            'error_bar': {'color': 'grey', 'fmt': 'none', 'capsize': 2, 'alpha': 1, 'zorder': 9, 'linewidth': 1.5, 'capthick': 1.5},
            'grid': {'which': 'major', 'linestyle': '-', 'alpha': 0.2, 'color': 'grey'},
            'scatter': {'s': 150, 'edgecolor': 'black', 'linewidth': 0.5, 'zorder': 10}
        },
        'agent_styling': {
            'default': {'lab_color': 'blue', 'marker': 'o'}
        },
        'legend_order': list(regressions['agent'].unique()),
        'ax_label_fontsize': 14,
        'title_fontsize': 16,
        'annotation_fontsize': 12,
        'xlabelpad': 10,
        'ylabelpad': 10,
        'suptitle_fontsize': 14
    }
    
    # Add agent-specific styling
    colors = ['#3e805f', '#e26e2f', '#4285f4', '#9333ea', '#f59e0b']
    markers = ['o', 's', '^', 'D', 'v']
    
    for i, agent in enumerate(regressions['agent'].unique()):
        plot_params['agent_styling'][agent] = {
            'lab_color': colors[i % len(colors)],
            'marker': markers[i % len(markers)]
        }
    
    # Create horizon plots using METR's function
    plot_files = []
    for success_rate in success_rates:
        import matplotlib.pyplot as plt
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Call METR's plotting function
        metr_funcs['plot_horizon_graph'](
            plot_params=plot_params,
            all_agent_summaries=regressions,
            runs_df=runs_df,
            release_dates=release_dates_dict,
            lower_y_lim=0.01,  # ~30 seconds
            upper_y_lim=10000,  # ~7 days
            x_lim_start='2024-01-01',
            x_lim_end='2025-12-31',
            subtitle='',
            title=f'AI Agent Horizon at {success_rate}% Success Rate',
            weight_key='weight',
            exclude_agents=[],
            success_percent=success_rate,
            confidence_level=0.8,  # Match the confidence level we generated
            fig=fig
        )
        
        plot_file = output_dir / f"horizon_plot_p{success_rate}.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        plot_files.append(plot_file)
        logger.info(f"Saved horizon plot to {plot_file}")
    
    logger.info(f"Generated {len(plot_files)} horizon plots")
    return plot_files 

def generate_individual_histograms(
    all_runs_file: Path,
    logistic_fits_file: Path,
    models_registry_file: Path, # Used to get agent display names for styling if needed
    output_dir: Path,
    metr_plot_utils_module: Any # Pass the imported src.utils.plots module
) -> Path:
    """
    Generates individual histogram plots with logistic regression overlays for each agent.
    Args:
        all_runs_file: Path to the METR formatted all_runs.jsonl file.
        logistic_fits_file: Path to the logistic_fits.csv file (output of METR's run_logistic_regressions).
        models_registry_file: Path to the models.json registry (for agent styling).
        output_dir: Directory to save the generated plot.
        metr_plot_utils_module: The imported src.utils.plots module from METR.
    Returns:
        Path to the generated plot file.
    """
    logger.info(f"Generating individual histograms from {all_runs_file} and {logistic_fits_file}")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "individual_histograms.png"

    all_runs_df = pd.read_json(all_runs_file, lines=True)
    agent_summaries_df = pd.read_csv(logistic_fits_file)

    try:
        with open(models_registry_file, 'r') as f:
            models_data_from_file = json.load(f)
        
        temp_agent_aliases_map = {}
        # Expect models_data_from_file to be a dict like: {"models": [{"metr_id": ..., "display_name": ...}, ...] }
        if isinstance(models_data_from_file, dict) and "models" in models_data_from_file:
            models_list = models_data_from_file["models"]
            if isinstance(models_list, list):
                for model_info_dict in models_list:
                    if isinstance(model_info_dict, dict):
                        metr_id = model_info_dict.get("metr_id")
                        display_name = model_info_dict.get("display_name")
                        if metr_id and display_name:
                            temp_agent_aliases_map[metr_id] = display_name
                        elif metr_id and not display_name: # Use metr_id if display_name is missing
                            temp_agent_aliases_map[metr_id] = metr_id
                if not temp_agent_aliases_map and models_list:
                    logger.warning(f"Loaded models list from {models_registry_file}, but no usable metr_id/display_name pairs were extracted.")
            else:
                logger.warning(f"Content under 'models' key in {models_registry_file} is a {type(models_list)}, not a list as expected for alias mapping. Model display names might not be used.")
        else:
            logger.warning(f"Models registry file {models_registry_file} is not a dictionary with a 'models' key, or the key is missing. Model display names might not be used.")
        
        agent_aliases_map = temp_agent_aliases_map

    except Exception as e: 
        logger.warning(f"Could not load or process models registry from {models_registry_file} for alias mapping: {e}. Using raw agent names for histograms.")
        agent_aliases_map = {}

    if 'alias' not in all_runs_df.columns and 'model' in all_runs_df.columns:
        all_runs_df['alias'] = all_runs_df['model'].apply(lambda x: agent_aliases_map.get(x, x))
    elif 'alias' not in all_runs_df.columns:
        logger.error("'alias' or 'model' column not found in all_runs_df. Cannot proceed.")
        return output_file

    focus_agents = agent_summaries_df["agent"].unique().tolist()
    if not focus_agents:
        logger.warning("No agents found in logistic_fits_file. Skipping histogram generation.")
        return output_file

    # Simplified plot_params, similar to METR's defaults
    base_colors = ['#3e805f', '#e26e2f', '#4285f4', '#9333ea', '#f59e0b']
    base_markers = ['o', 's', '^', 'D', 'v']

    agent_specific_styling = {}
    for i, agent_name in enumerate(focus_agents):
        color_value = base_colors[i % len(base_colors)]
        agent_specific_styling[agent_name] = {
            'color': color_value,
            'marker': base_markers[i % len(base_markers)],
            'unique_color': color_value
        }

    plot_params_dict = {
        'scatter_styling': {
            'error_bar': {'color': 'grey', 'fmt': 'none', 'capsize': 2, 'alpha': 1, 'zorder': 9, 'linewidth': 1.5, 'capthick': 1.5},
            'grid': {'which': 'major', 'linestyle': '-', 'alpha': 0.2, 'color': 'grey'},
            'scatter': {'s': 150, 'edgecolor': 'black', 'linewidth': 0.5, 'zorder': 10}
        },
        'agent_styling': {
            'default': {'color': 'blue', 'marker': 'o', 'unique_color': 'blue'}
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
    plot_params_dict['agent_styling'].update(agent_specific_styling)

    # Script_params based on METR's params.yaml for individual_histograms
    script_params = {
        'title': 'Individual Model Success Rates vs. Task Length',
        'n_subplot_cols': 3, 
        'horizontal_lines': [
            {
                'p_success': 0.5,
                'styling': {
                    'color': '#b30c00',
                    'linestyle': 'dashed',
                    'linewidth': 1.5, 
                    'alpha': 0.8      
                }
            }
        ],
        'annotate_p50': True,
        'exclude': [], 
        'include_agents': focus_agents,
        'weighting': 'weight' 
    }

    metr_funcs = _import_metr_functions()
    plot_func = metr_funcs['plot_logistic_regression_on_histogram']

    logger.info(f"Calling METR's plot_logistic_regression_on_histogram for agents: {focus_agents}")
    try:
        plot_func(
            plot_params=plot_params_dict, 
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