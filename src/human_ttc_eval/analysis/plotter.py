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
            
            return {
                'run_logistic_regressions': run_logistic_regressions,
                'WrangleParams': WrangleParams,
                'logistic_regression': logistic_regression,
                'get_x_for_quantile': get_x_for_quantile
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
        regularization=1e-6,  # Small value to avoid division by zero
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
    
    # Create horizon plots
    plot_files = []
    for success_rate in success_rates:
        plot_file = _create_simple_horizon_plot(
            regressions, success_rate, output_dir
        )
        plot_files.append(plot_file)
    
    logger.info(f"Generated {len(plot_files)} horizon plots")
    return plot_files

def _create_simple_horizon_plot(regressions: pd.DataFrame, success_rate: int, output_dir: Path) -> Path:
    """Create a simple horizon plot using matplotlib."""
    import matplotlib.pyplot as plt
    import numpy as np
    from datetime import datetime
    
    # Prepare data
    agents = regressions['agent'].tolist()
    horizons = regressions[f'p{success_rate}'].tolist()
    release_dates = pd.to_datetime(regressions['release_date']).tolist()
    
    # Create plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot horizons vs release dates
    colors = plt.cm.Set1(np.linspace(0, 1, len(agents)))
    for i, (agent, horizon, date) in enumerate(zip(agents, horizons, release_dates)):
        # Show all agents, even those with 0% success rate
        if np.isnan(horizon):
            continue  # Skip only NaN values
        
        # For zero or very low horizons, show at minimum visible level
        plot_horizon = max(horizon, 0.1) if horizon > 0 else 0.05
        
        ax.scatter(date, plot_horizon, color=colors[i], s=100, label=agent, alpha=0.8)
        
        # Annotate with actual horizon value (or "0%" for zero success)
        if horizon <= 0:
            ax.annotate(f"{agent} (0% success)", (date, plot_horizon), xytext=(5, 5), 
                       textcoords='offset points', fontsize=9, style='italic')
        else:
            ax.annotate(agent, (date, plot_horizon), xytext=(5, 5), 
                       textcoords='offset points', fontsize=9)
    
    # Format plot
    ax.set_xlabel('Model Release Date')
    ax.set_ylabel(f'{success_rate}% Success Horizon (minutes)')
    ax.set_title(f'Model Performance Horizon at {success_rate}% Success Rate')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Save plot
    plot_file = output_dir / f"horizon_plot_p{success_rate}.png"
    plt.tight_layout()
    plt.savefig(plot_file, dpi=150, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved horizon plot to {plot_file}")
    return plot_file 