"""
Generate horizon plots using METR's methodology.

Wrapper around METR's logistic regression and plotting functions.
"""

import sys
import logging
from pathlib import Path
from typing import List
import yaml
import pandas as pd

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
            import src.utils.plots
            
            return {
                'run_logistic_regressions': run_logistic_regressions,
                'WrangleParams': WrangleParams,
                'logistic_regression': logistic_regression,
                'get_x_for_quantile': get_x_for_quantile,
                'plot_horizon_graph': plot_horizon_graph,
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
    regressions = metr_funcs['run_logistic_regressions'](
        runs=runs_df,
        release_dates_file=release_dates_file,
        wrangle_params=wrangle_params,
        bootstrap_file=None,
        include_empirical_rates=True
    )
    
    # Save regression results  
    logistic_fits_file = metr_data_dir / "logistic_fits.csv"
    regressions.to_csv(logistic_fits_file, index=False)
    logger.info(f"Saved logistic fits to {logistic_fits_file}")
    
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