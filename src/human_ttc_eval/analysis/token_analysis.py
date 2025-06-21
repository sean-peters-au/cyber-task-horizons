"""
Token usage analysis for cybersecurity benchmark evaluations.

Extracts token usage data from inspect_ai evaluation logs and creates
research-quality visualizations showing token consumption patterns.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, NamedTuple, Tuple
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pandas as pd
import numpy as np
from scipy import stats
from matplotlib.ticker import FuncFormatter

from ..core.run import Run

logger = logging.getLogger(__name__)


class TokenUsage(NamedTuple):
    """Token usage data for a single task."""
    task_id: str
    model: str
    input_tokens: int
    output_tokens: int
    total_tokens: int
    reasoning_tokens: int = 0


def extract_tokens_from_eval_logs(log_dir: Path) -> Dict[str, TokenUsage]:
    """
    Extract token usage data from inspect_ai evaluation logs.
    
    Args:
        log_dir: Directory containing .eval files
        
    Returns:
        Dict mapping task_id to TokenUsage data
    """
    token_data = {}
    
    if not log_dir.exists():
        logger.warning(f"Log directory not found: {log_dir}")
        return token_data
    
    try:
        from inspect_ai.log import read_eval_log
    except ImportError:
        logger.warning("inspect_ai not available - skipping token analysis")
        return token_data
    
    eval_files = list(log_dir.glob("*.eval"))
    if not eval_files:
        logger.info(f"No .eval files found in {log_dir}")
        return token_data
    
    logger.info(f"Processing {len(eval_files)} evaluation logs for token data")
    
    for eval_file in eval_files:
        try:
            log = read_eval_log(str(eval_file))
            
            if not hasattr(log, 'samples') or not log.samples:
                continue
                
            for sample in log.samples:
                if not hasattr(sample, 'model_usage') or not sample.model_usage:
                    continue
                    
                # Get the first (and usually only) model usage entry
                model_name = list(sample.model_usage.keys())[0]
                usage = sample.model_usage[model_name]
                
                token_usage = TokenUsage(
                    task_id=sample.id,
                    model=model_name,
                    input_tokens=getattr(usage, 'input_tokens', 0),
                    output_tokens=getattr(usage, 'output_tokens', 0),
                    total_tokens=getattr(usage, 'total_tokens', 0),
                    reasoning_tokens=getattr(usage, 'reasoning_tokens', 0)
                )
                
                token_data[sample.id] = token_usage
                
        except Exception as e:
            logger.warning(f"Failed to process {eval_file}: {e}")
            continue
    
    logger.info(f"Extracted token data for {len(token_data)} samples")
    return token_data


def create_token_plots(runs_data: List[Run], token_data: Dict[str, TokenUsage], output_dir: Path) -> None:
    """
    Create research-quality token analysis plots.
    
    Args:
        runs_data: List of Run objects with evaluation results
        token_data: Token usage data mapped by task_id
        output_dir: Directory to save plots
    """
    if not token_data:
        logger.info("No token data available - skipping token plots")
        return
    
    # Merge runs with token data
    plot_data = []
    for run in runs_data:
        if run.task_id in token_data:
            tokens = token_data[run.task_id]
            plot_data.append({
                'task_id': run.task_id,
                'task_family': run.task_family,
                'model': run.model,
                'alias': run.alias,
                'human_minutes': run.human_minutes,
                'score_binarized': run.score_binarized,
                'score_cont': run.score_cont or 0.0,
                'input_tokens': tokens.input_tokens,
                'output_tokens': tokens.output_tokens,
                'total_tokens': tokens.total_tokens,
                'reasoning_tokens': tokens.reasoning_tokens,
            })
    
    if not plot_data:
        logger.info("No runs matched with token data - skipping token plots")
        return
    
    df = pd.DataFrame(plot_data)
    logger.info(f"Creating token plots for {len(df)} samples")
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Apply consistent styling
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams.update({
        'font.size': 12,
        'axes.labelsize': 14,
        'axes.titlesize': 16,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 12,
        'figure.dpi': 300,
    })
    
    # 1. Token consumption vs task horizon (multi-model)
    _plot_token_vs_horizon_multimodel(df, output_dir)
    
    # 2. Token efficiency curves (success rate vs token budget)
    _plot_token_efficiency_curves(df, output_dir)
    
    # 3. Tokens per minute of human work
    _plot_tokens_per_minute(df, output_dir)


def _plot_token_vs_horizon_multimodel(df: pd.DataFrame, output_dir: Path) -> None:
    """Create multi-panel plot of token consumption vs task difficulty for each model."""
    models = sorted(df['model'].unique())
    n_models = len(models)
    
    if n_models == 0:
        logger.warning("No models found in data")
        return
    
    # Calculate grid dimensions
    n_cols = min(3, n_models)
    n_rows = (n_models + n_cols - 1) // n_cols
    
    fig = plt.figure(figsize=(6 * n_cols, 5 * n_rows))
    gs = gridspec.GridSpec(n_rows, n_cols, figure=fig, hspace=0.3, wspace=0.3)
    
    # Common axis limits based on all data
    x_min, x_max = df['human_minutes'].min() * 0.8, df['human_minutes'].max() * 1.2
    y_min, y_max = df['total_tokens'].min() * 0.8, df['total_tokens'].max() * 1.2
    
    for idx, model in enumerate(models):
        ax = fig.add_subplot(gs[idx])
        model_df = df[df['model'] == model].copy()
        
        if len(model_df) == 0:
            continue
        
        # Separate successful and failed attempts
        success_df = model_df[model_df['score_binarized'] == 1]
        fail_df = model_df[model_df['score_binarized'] == 0]
        
        # Plot scatter points
        if len(fail_df) > 0:
            ax.scatter(fail_df['human_minutes'], fail_df['total_tokens'],
                      c='#e74c3c', alpha=0.4, s=30, label='Failed', edgecolors='none')
        if len(success_df) > 0:
            ax.scatter(success_df['human_minutes'], success_df['total_tokens'],
                      c='#27ae60', alpha=0.6, s=30, label='Solved', edgecolors='none')
        
        # Fit power law to all data points
        if len(model_df) > 5:  # Need sufficient data for fitting
            # Remove any zero or negative values for log transform
            valid_data = model_df[(model_df['human_minutes'] > 0) & (model_df['total_tokens'] > 0)]
            
            if len(valid_data) > 5:
                X_log = np.log(valid_data['human_minutes'].values)
                y_log = np.log(valid_data['total_tokens'].values)
                
                # Fit linear regression in log space
                coeffs = np.polyfit(X_log, y_log, 1)
                
                # Create smooth curve for plotting
                x_range = np.logspace(np.log10(x_min), np.log10(x_max), 100)
                y_pred = np.exp(coeffs[1]) * x_range ** coeffs[0]
                
                ax.plot(x_range, y_pred, 'k--', alpha=0.8, linewidth=2,
                       label=f'Power law: tokens ∝ minutes^{coeffs[0]:.2f}')
                
                # Add R² value
                y_pred_points = np.exp(coeffs[1]) * valid_data['human_minutes'].values ** coeffs[0]
                r2 = 1 - np.sum((valid_data['total_tokens'].values - y_pred_points)**2) / \
                         np.sum((valid_data['total_tokens'].values - valid_data['total_tokens'].mean())**2)
                ax.text(0.05, 0.95, f'R² = {r2:.3f}', transform=ax.transAxes,
                       verticalalignment='top', fontsize=10,
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Formatting
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_xlabel('Human Minutes')
        ax.set_ylabel('Total Tokens')
        ax.set_title(f'{model}', fontsize=14)
        ax.legend(loc='lower right', framealpha=0.9)
        ax.grid(True, alpha=0.3)
    
    # Add overall title
    fig.suptitle('Token Consumption vs Task Horizon', fontsize=16, y=1.02)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'token_vs_horizon_multimodel.png', dpi=300, bbox_inches='tight')
    plt.close()


def _plot_token_efficiency_curves(df: pd.DataFrame, output_dir: Path) -> None:
    """Plot success rate vs token budget curves for each model."""
    models = sorted(df['model'].unique())
    
    # Define token budget thresholds
    token_budgets = [500, 1000, 2500, 5000, 10000, 25000, 50000, 100000, 250000]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Color palette
    colors = plt.cm.Set1(np.linspace(0, 1, len(models)))
    
    for idx, model in enumerate(models):
        model_df = df[df['model'] == model]
        
        if len(model_df) < 10:  # Skip if too few samples
            continue
        
        success_rates = []
        ci_lower = []
        ci_upper = []
        valid_budgets = []
        
        for budget in token_budgets:
            # Find tasks that could be attempted within this token budget
            within_budget = model_df[model_df['total_tokens'] <= budget]
            
            if len(within_budget) >= 5:  # Need minimum samples for meaningful rate
                # Bootstrap confidence intervals
                n_bootstrap = 1000
                success_samples = []
                
                for _ in range(n_bootstrap):
                    sample = within_budget.sample(n=len(within_budget), replace=True)
                    success_samples.append(sample['score_binarized'].mean())
                
                success_rates.append(np.mean(success_samples))
                ci_lower.append(np.percentile(success_samples, 2.5))
                ci_upper.append(np.percentile(success_samples, 97.5))
                valid_budgets.append(budget)
        
        if valid_budgets:
            # Plot line with confidence bands
            ax.plot(valid_budgets, success_rates, 'o-', color=colors[idx], 
                   linewidth=2, markersize=8, label=model)
            ax.fill_between(valid_budgets, ci_lower, ci_upper, 
                           color=colors[idx], alpha=0.2)
    
    # Formatting
    ax.set_xscale('log')
    ax.set_xlabel('Token Budget')
    ax.set_ylabel('Success Rate')
    ax.set_title('Model Success Rate vs Token Budget\n(with 95% confidence intervals)')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='lower right', framealpha=0.9)
    ax.set_ylim(-0.05, 1.05)
    
    # Add token budget labels
    ax.set_xticks(token_budgets)
    ax.set_xticklabels([f'{b//1000}k' if b >= 1000 else str(b) for b in token_budgets])
    
    plt.tight_layout()
    plt.savefig(output_dir / 'token_efficiency_curves.png', dpi=300, bbox_inches='tight')
    plt.close()


def _plot_tokens_per_minute(df: pd.DataFrame, output_dir: Path) -> None:
    """Plot token efficiency (tokens per minute of human work) by task difficulty."""
    models = sorted(df['model'].unique())
    
    # Define time buckets
    time_buckets = [(0, 1), (1, 5), (5, 30), (30, 180), (180, float('inf'))]
    bucket_labels = ['<1 min', '1-5 min', '5-30 min', '30-180 min', '>3 hours']
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Color palette and positioning
    colors = plt.cm.Set1(np.linspace(0, 1, len(models)))
    n_models = len(models)
    bar_width = 0.8 / n_models
    
    for model_idx, model in enumerate(models):
        model_df = df[df['model'] == model]
        
        bucket_positions = []
        medians = []
        q1_values = []
        q3_values = []
        
        for bucket_idx, (min_time, max_time) in enumerate(time_buckets):
            bucket_data = model_df[
                (model_df['human_minutes'] >= min_time) & 
                (model_df['human_minutes'] < max_time)
            ]
            
            if len(bucket_data) >= 3:  # Need minimum samples
                # Calculate tokens per minute
                tokens_per_minute = bucket_data['total_tokens'] / bucket_data['human_minutes']
                
                # Use median and IQR for robustness against outliers
                median = tokens_per_minute.median()
                q1 = tokens_per_minute.quantile(0.25)
                q3 = tokens_per_minute.quantile(0.75)
                
                bucket_positions.append(bucket_idx)
                medians.append(median)
                q1_values.append(q1)
                q3_values.append(q3)
        
        if bucket_positions:
            # Calculate x positions for grouped bars
            x_positions = [p + (model_idx - n_models/2 + 0.5) * bar_width for p in bucket_positions]
            
            # Plot bars with error bars
            bars = ax.bar(x_positions, medians, bar_width, 
                          color=colors[model_idx], alpha=0.8, label=model)
            
            # Add error bars (IQR)
            errors = [[m - q1 for m, q1 in zip(medians, q1_values)],
                     [q3 - m for m, q3 in zip(medians, q3_values)]]
            ax.errorbar(x_positions, medians, yerr=errors, 
                       fmt='none', color='black', capsize=3, alpha=0.5)
            
            # Add value labels on bars
            for x, median in zip(x_positions, medians):
                ax.text(x, median + 50, f'{median:.0f}', 
                       ha='center', va='bottom', fontsize=9)
    
    # Formatting
    ax.set_xticks(range(len(bucket_labels)))
    ax.set_xticklabels(bucket_labels)
    ax.set_xlabel('Task Difficulty (Human Time)')
    ax.set_ylabel('Tokens per Human Minute (median)')
    ax.set_title('Token Efficiency by Task Difficulty\n(bars show median, error bars show IQR)')
    ax.legend(loc='upper right', framealpha=0.9)
    ax.grid(True, axis='y', alpha=0.3)
    
    # Use log scale if range is large
    if df['total_tokens'].max() / df['total_tokens'].min() > 100:
        ax.set_yscale('log')
        ax.set_ylabel('Tokens per Human Minute (median, log scale)')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'tokens_per_minute_efficiency.png', dpi=300, bbox_inches='tight')
    plt.close()