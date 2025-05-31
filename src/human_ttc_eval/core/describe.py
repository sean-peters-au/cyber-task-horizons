"""
Base class for dataset describers.

Describers generate summary statistics and visualizations from processed
Run data to understand human baseline performance characteristics.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Optional, Dict, Any
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import logging

from .run import Run

logger = logging.getLogger(__name__)


class Describe(ABC):
    """
    Abstract base class for dataset describers.
    
    Describers read processed Run data (METR format JSONL) and generate:
    - Summary statistics (CSV files)
    - Visualizations (PNG plots)
    - Dataset-specific analyses
    
    Key principles:
    - Input: Processed JSONL file from data/processed/<dataset>/all_tasks.jsonl
    - Output: Statistics and plots in results/dataset-summaries/<dataset>/
    - Works with both single and multiple human runs per task
    - Generates standardized outputs for cross-dataset comparison
    """
    
    def __init__(self, input_files: List[Path], output_dir: Path):
        """
        Initialize describer with input files and output directory.
        
        Args:
            input_files: List of paths to processed JSONL files (typically data/processed/<dataset>/all_tasks.jsonl)
            output_dir: Directory for outputs (typically results/dataset-summaries/<dataset>/)
            dataset_name: Optional dataset identifier (lowercase, no spaces)
        """
        self.input_files: List[Path] = [Path(p) for p in input_files]
        self.output_dir: Path = Path(output_dir)

        if not self.input_files:
            raise ValueError("At least one input file must be provided to Describe.")

        self.output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Initialized {self.__class__.__name__} for dataset '{self.dataset_name}'. Output directory: {self.output_dir}")
        
        self.runs: List[Run] = []
        self.df: Optional[pd.DataFrame] = None
    
    @property
    @abstractmethod
    def dataset_name(self) -> str:
        """
        Returns the dataset identifier.
        
        This should match the name used in the CLI and directory structure.
        Examples: "nl2bash", "cybench"
        
        Returns:
            Dataset name as a slug (lowercase, no spaces)
        """
        raise NotImplementedError("dataset_name must be implemented in the subclass")
    
    def load_runs(self) -> List[Run]:
        """
        Load runs from JSONL file.
        
        Returns:
            List of all loaded Run objects.
        """
        self.runs = [] # Initialize/clear existing runs
        
        if not self.input_files: # Should have been caught by __init__ but good to check
            logger.warning("No input files specified for loading runs.")
            self.df = pd.DataFrame()
            return self.runs

        for file_path in self.input_files:
            if file_path.exists():
                logger.info(f"Loading runs from {file_path}...")
                try:
                    loaded_from_file = Run.load_jsonl(str(file_path))
                    self.runs.extend(loaded_from_file)
                    logger.info(f"Loaded {len(loaded_from_file)} runs from {file_path}.")
                except Exception as e:
                    logger.error(f"Failed to load or parse runs from {file_path}: {e}", exc_info=True)
            else:
                logger.warning(f"Input file not found: {file_path}")
        
        logger.info(f"Total {len(self.runs)} runs loaded from {len(self.input_files)} file(s).")
        
        if self.runs:
            self.df = pd.DataFrame([run.to_jsonl_dict() for run in self.runs])
            numeric_cols = ['score_binarized', 'score_cont', 'human_minutes', 'human_score', 
                          'human_cost', 'generation_cost', 'equal_task_weight', 'invsqrt_task_weight']
            for col in numeric_cols:
                if col in self.df.columns:
                    self.df[col] = pd.to_numeric(self.df[col], errors='coerce')
        else:
            logger.warning("No runs were successfully loaded, DataFrame will be empty.")
            self.df = pd.DataFrame() 
            
        return self.runs
    
    def generate_overall_stats(self) -> Dict[str, Any]:
        """
        Generate overall dataset statistics.
        
        Returns:
            Dictionary of statistics
        """
        if self.df is None or self.df.empty:
            logger.warning("No data available for overall statistics")
            return {"error": "No data available"}
        
        # Get unique tasks (for handling multiple human runs) - get first run per task
        unique_tasks = []
        seen_tasks = set()
        for run in self.runs:
            if run.task_id not in seen_tasks:
                unique_tasks.append(run)
                seen_tasks.add(run.task_id)
        
        stats = {
            "Total Runs": len(self.df),
            "Unique Tasks": len(unique_tasks),
            "Task Families": self.df['task_family'].nunique() if 'task_family' in self.df.columns else 0,
            "Human Runs": len(self.df[self.df['model'] == 'human']) if 'model' in self.df.columns else 0,
            "AI Runs": len(self.df[self.df['model'] != 'human']) if 'model' in self.df.columns else 0,
        }
        
        # Success analysis (on unique tasks)
        unique_df = pd.DataFrame([run.to_jsonl_dict() for run in unique_tasks])
        if 'score_binarized' in unique_df.columns:
            successful_tasks = unique_df['score_binarized'].sum()
            stats.update({
                "Successful Tasks": int(successful_tasks),
                "Task Success Rate (%)": round(successful_tasks / len(unique_tasks) * 100, 2) if unique_tasks else 0,
            })
        
        # Timing analysis (on unique tasks)
        if 'human_minutes' in unique_df.columns:
            stats.update({
                "Human Time (min) - Min": round(unique_df['human_minutes'].min(), 2),
                "Human Time (min) - Max": round(unique_df['human_minutes'].max(), 2),
                "Human Time (min) - Mean": round(unique_df['human_minutes'].mean(), 2),
                "Human Time (min) - Median": round(unique_df['human_minutes'].median(), 2),
                "Human Time (min) - Total": round(unique_df['human_minutes'].sum(), 2),
            })
        
        # Cost analysis
        if 'human_cost' in unique_df.columns:
            stats.update({
                "Total Human Cost ($)": round(unique_df['human_cost'].sum(), 2),
                "Mean Human Cost per Task ($)": round(unique_df['human_cost'].mean(), 2),
            })
        
        return stats
    
    def generate_task_family_stats(self) -> pd.DataFrame:
        """
        Generate per-task-family statistics.
        
        Returns:
            DataFrame with task family statistics
        """
        if self.df is None or self.df.empty or 'task_family' not in self.df.columns:
            logger.warning("No data available for task family statistics")
            return pd.DataFrame()
        
        # Use unique tasks for task-level metrics - get first run per task
        unique_tasks = []
        seen_tasks = set()
        for run in self.runs:
            if run.task_id not in seen_tasks:
                unique_tasks.append(run)
                seen_tasks.add(run.task_id)
        
        unique_df = pd.DataFrame([run.to_jsonl_dict() for run in unique_tasks])
        
        # Aggregate by task family
        agg_dict = {
            'task_id': 'count',
        }
        
        if 'score_binarized' in unique_df.columns:
            agg_dict['score_binarized'] = ['sum', 'mean']
        if 'human_minutes' in unique_df.columns:
            agg_dict['human_minutes'] = ['min', 'max', 'mean', 'median', 'sum']
        if 'human_cost' in unique_df.columns:
            agg_dict['human_cost'] = ['sum', 'mean']
        
        grouped = unique_df.groupby('task_family').agg(agg_dict).round(2)
        
        # Flatten column names
        grouped.columns = ['_'.join(col).strip('_') if isinstance(col, tuple) else col 
                          for col in grouped.columns.values]
        grouped = grouped.reset_index()
        
        # Rename columns for clarity
        column_renames = {
            'task_id_count': 'num_tasks',
            'score_binarized_sum': 'successful_tasks',
            'score_binarized_mean': 'success_rate',
            'human_minutes_sum': 'total_human_minutes',
            'human_minutes_mean': 'mean_human_minutes',
            'human_minutes_median': 'median_human_minutes',
            'human_cost_sum': 'total_human_cost',
            'human_cost_mean': 'mean_human_cost',
        }
        grouped.rename(columns=column_renames, inplace=True)
        
        # Convert success rate to percentage
        if 'success_rate' in grouped.columns:
            grouped['success_rate'] = (grouped['success_rate'] * 100).round(2)
        
        return grouped
    
    def plot_human_time_distribution(self) -> None:
        """Plot distribution of human completion times."""
        if self.df is None or 'human_minutes' not in self.df.columns:
            logger.warning("No human_minutes data available for plotting")
            return
        
        # Use unique tasks only - get first run per task
        unique_tasks = []
        seen_tasks = set()
        for run in self.runs:
            if run.task_id not in seen_tasks:
                unique_tasks.append(run)
                seen_tasks.add(run.task_id)
        
        unique_df = pd.DataFrame([run.to_jsonl_dict() for run in unique_tasks])
        
        plt.figure(figsize=(10, 6))
        sns.histplot(unique_df['human_minutes'].dropna(), kde=True, bins=30)
        plt.title(f'{self.dataset_name.title()} - Human Completion Time Distribution')
        plt.xlabel('Time (minutes)')
        plt.ylabel('Number of Tasks')
        plt.tight_layout()
        plt.savefig(self.output_dir / 'human_time_distribution.png')
        plt.close()
        logger.info("Human time distribution plot saved")
    
    def plot_task_family_comparison(self) -> None:
        """Plot task family comparisons."""
        family_stats = self.generate_task_family_stats()
        if family_stats.empty:
            logger.warning("No task family data available for plotting")
            return
        
        # Create subplots for different metrics
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'{self.dataset_name.title()} - Task Family Comparison', fontsize=16)
        
        # Plot 1: Number of tasks per family
        if 'num_tasks' in family_stats.columns:
            ax = axes[0, 0]
            sns.barplot(data=family_stats, x='task_family', y='num_tasks', ax=ax)
            ax.set_title('Number of Tasks per Family')
            ax.set_xlabel('Task Family')
            ax.set_ylabel('Number of Tasks')
            ax.tick_params(axis='x', rotation=45)
        
        # Plot 2: Success rate per family
        if 'success_rate' in family_stats.columns:
            ax = axes[0, 1]
            sns.barplot(data=family_stats, x='task_family', y='success_rate', ax=ax)
            ax.set_title('Success Rate per Family')
            ax.set_xlabel('Task Family')
            ax.set_ylabel('Success Rate (%)')
            ax.set_ylim(0, 100)
            ax.tick_params(axis='x', rotation=45)
        
        # Plot 3: Median completion time per family
        if 'median_human_minutes' in family_stats.columns:
            ax = axes[1, 0]
            sns.barplot(data=family_stats, x='task_family', y='median_human_minutes', ax=ax)
            ax.set_title('Median Completion Time per Family')
            ax.set_xlabel('Task Family')
            ax.set_ylabel('Time (minutes)')
            ax.tick_params(axis='x', rotation=45)
        
        # Plot 4: Total human cost per family
        if 'total_human_cost' in family_stats.columns:
            ax = axes[1, 1]
            sns.barplot(data=family_stats, x='task_family', y='total_human_cost', ax=ax)
            ax.set_title('Total Human Cost per Family')
            ax.set_xlabel('Task Family')
            ax.set_ylabel('Cost ($)')
            ax.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'task_family_comparison.png')
        plt.close()
        logger.info("Task family comparison plot saved")
    
    def save_summary_files(self) -> None:
        """Save summary statistics to CSV files."""
        # Overall statistics
        overall_stats = self.generate_overall_stats()
        overall_df = pd.DataFrame.from_dict(overall_stats, orient='index', columns=['Value'])
        overall_df.to_csv(self.output_dir / 'overall_summary.csv')
        logger.info("Overall summary saved")
        
        # Task family statistics
        family_stats = self.generate_task_family_stats()
        if not family_stats.empty:
            family_stats.to_csv(self.output_dir / 'task_family_summary.csv', index=False)
            logger.info("Task family summary saved")
    
    @abstractmethod
    def generate_custom_analysis(self) -> None:
        """
        Generate dataset-specific analyses.
        
        Override this method to add custom statistics and visualizations
        specific to your dataset beyond the standard METR analyses.
        """
        raise NotImplementedError("generate_custom_analysis must be implemented in the subclass")
    
    def run(self) -> None:
        """
        Execute the full description pipeline.
        
        This method orchestrates the description process:
        1. Load runs from JSONL
        2. Generate standard statistics and plots
        3. Generate custom analyses
        4. Save all outputs
        """
        logger.info(f"Starting description for {self.dataset_name} dataset...")
        
        # Load the data
        self.load_runs()
        
        if not self.runs:
            logger.warning("No runs loaded. Creating empty summary.")
            with open(self.output_dir / 'summary.txt', 'w') as f:
                f.write(f"No data available for {self.dataset_name} dataset.\n")
            return
        
        # Generate standard outputs
        self.save_summary_files()
        self.plot_human_time_distribution()
        self.plot_task_family_comparison()
        
        # Generate custom analyses
        self.generate_custom_analysis()
        
        logger.info(f"Description complete. Results saved to {self.output_dir}") 