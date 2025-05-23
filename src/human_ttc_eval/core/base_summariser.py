from abc import ABC, abstractmethod
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional
import json
import logging

logger = logging.getLogger(__name__)

class BaseSummariser(ABC):
    """
    Abstract base class for dataset summarisers.
    Each summariser will take a path to an all_runs.jsonl file (or equivalent)
    and produce summary statistics and plots.
    """

    def __init__(self, jsonl_file_path: Path, output_dir: Path):
        self.jsonl_file_path = jsonl_file_path
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.df: Optional[pd.DataFrame] = None # Initialize df as None

    def load_data(self) -> None:
        """Loads the JSONL file into a pandas DataFrame."""
        records = []
        with open(self.jsonl_file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError as e:
                        logger.warning(f"Skipping malformed JSON line: {line.strip()[:100]} - Error: {e}")
        
        if not records:
            logger.warning(f"No valid records found in {self.jsonl_file_path}. DataFrame will be empty.")
            self.df = pd.DataFrame()
        else:
            self.df = pd.DataFrame(records)
            logger.info(f"Loaded {len(self.df)} records from {self.jsonl_file_path}.")
            # Basic type coercion, can be overridden by subclasses
            numeric_cols = ['score_binarized', 'score_cont', 'human_minutes', 'human_score', 'command_count']
            for col in numeric_cols:
                if col in self.df.columns:
                    self.df[col] = pd.to_numeric(self.df[col], errors='coerce')

    # Common plotting functions
    def _plot_histogram(self, df: pd.DataFrame, column: str, title: str, output_path: Path, bins=30):
        """Plot histogram for a given column."""
        if df.empty or column not in df.columns or df[column].isnull().all():
            logger.warning(f"DataFrame is empty or column '{column}' is missing/all NaNs. Skipping histogram: {title}")
            return
        plt.figure(figsize=(10, 6))
        sns.histplot(df[column].dropna(), kde=True, bins=bins)
        plt.title(title)
        plt.xlabel(column.replace('_', ' ').title())
        plt.ylabel('Frequency')
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()
        logger.info(f"Histogram '{title}' saved to {output_path}")

    def _plot_bar_chart(self, data: pd.DataFrame, x_col: str, y_col: str, title: str, output_path: Path, y_label: str = None):
        """Plot bar chart for given columns."""
        if data.empty or x_col not in data.columns or y_col not in data.columns:
            logger.warning(f"DataFrame is empty or columns '{x_col}'/'{y_col}' missing. Skipping bar chart: {title}")
            return
        plt.figure(figsize=(12, 7))
        sns.barplot(x=x_col, y=y_col, data=data, palette="viridis")
        plt.title(title)
        plt.xlabel(x_col.replace('_', ' ').title())
        plt.ylabel(y_label if y_label else y_col.replace('_', ' ').title())
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()
        logger.info(f"Bar chart '{title}' saved to {output_path}")

    def _plot_scatter(self, df: pd.DataFrame, x_col: str, y_col: str, hue_col: Optional[str], title: str, output_path: Path):
        """Plot scatter plot for given columns."""
        if df.empty or x_col not in df.columns or y_col not in df.columns:
            logger.warning(f"DataFrame is empty or columns '{x_col}'/'{y_col}' missing. Skipping scatter plot: {title}")
            return
        if hue_col and hue_col not in df.columns:
            logger.warning(f"Hue column '{hue_col}' not found. Plotting without hue.")
            hue_col = None
            
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x=x_col, y=y_col, hue=hue_col, data=df, alpha=0.6)
        plt.title(title)
        plt.xlabel(x_col.replace('_', ' ').title())
        plt.ylabel(y_col.replace('_', ' ').title())
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()
        logger.info(f"Scatter plot '{title}' saved to {output_path}")

    # Standard METR format analysis methods
    def _generate_standard_overall_stats(self, df: pd.DataFrame) -> None:
        """Generate standard overall statistics for METR format data."""
        output_path = self.output_dir / "overall_summary_stats.csv"
        if df.empty:
            logger.warning("DataFrame is empty. Skipping overall stats generation.")
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write("No data available to generate overall statistics.\n")
            return

        total_records = len(df)
        
        stats = {
            "Total Records": total_records,
            "Total Task Families": df['task_family'].nunique() if 'task_family' in df.columns else 0,
            "Total Unique Tasks": df['task_id'].nunique() if 'task_id' in df.columns else 0,
        }

        # Success analysis
        if 'score_binarized' in df.columns:
            successful_records = df['score_binarized'].sum()
            stats.update({
                "Successful Records": successful_records,
                "Success Rate (%)": (successful_records / total_records * 100) if total_records > 0 else 0,
            })

        # Timing analysis
        if 'human_minutes' in df.columns and not df['human_minutes'].isnull().all():
            stats.update({
                "Human Time (minutes) - Min": df['human_minutes'].min(),
                "Human Time (minutes) - Max": df['human_minutes'].max(),
                "Human Time (minutes) - Mean": df['human_minutes'].mean(),
                "Human Time (minutes) - Median": df['human_minutes'].median(),
                "Human Time (minutes) - StdDev": df['human_minutes'].std(),
            })

        # Score analysis
        if 'score_cont' in df.columns and not df['score_cont'].isnull().all():
            stats.update({
                "Continuous Score - Min": df['score_cont'].min(),
                "Continuous Score - Max": df['score_cont'].max(),
                "Continuous Score - Mean": df['score_cont'].mean(),
                "Continuous Score - Median": df['score_cont'].median(),
                "Continuous Score - StdDev": df['score_cont'].std(),
            })

        stats_df = pd.DataFrame.from_dict(stats, orient='index', columns=['Value'])
        stats_df.to_csv(output_path)
        logger.info(f"Standard overall statistics saved to {output_path}")

    def _generate_task_family_stats(self, df: pd.DataFrame) -> Optional[pd.DataFrame]:
        """Generate per-task-family statistics."""
        output_path = self.output_dir / "per_task_family_stats.csv"
        if df.empty or 'task_family' not in df.columns:
            logger.warning("DataFrame is empty or missing 'task_family'. Skipping task family stats.")
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write("No data available to generate task family statistics.\n")
            return None

        # Build aggregation dict based on available columns
        agg_dict = {'task_id': 'count'}
        
        if 'score_binarized' in df.columns:
            agg_dict['score_binarized'] = 'sum'
        if 'human_minutes' in df.columns:
            agg_dict['human_minutes'] = ['min', 'max', 'mean', 'median', 'std']
        if 'score_cont' in df.columns:
            agg_dict['score_cont'] = ['mean', 'median', 'std']

        grouped = df.groupby('task_family').agg(agg_dict).round(2)
        
        # Flatten column names
        grouped.columns = ['_'.join(col).strip() if isinstance(col, tuple) else col for col in grouped.columns.values]
        grouped = grouped.reset_index()
        
        # Calculate success rate if possible
        if 'score_binarized_sum' in grouped.columns and 'task_id_count' in grouped.columns:
            grouped['success_rate_percent'] = (grouped['score_binarized_sum'] / grouped['task_id_count'] * 100).round(2)
        
        grouped.to_csv(output_path, index=False)
        logger.info(f"Task family statistics saved to {output_path}")
        return grouped

    def _generate_standard_plots(self) -> None:
        """Generate standard plots for METR format data."""
        if self.df is None or self.df.empty:
            logger.warning("No data available for standard plots.")
            return

        # Timing distribution
        if 'human_minutes' in self.df.columns:
            self._plot_histogram(self.df, 'human_minutes', 'Distribution of Human Completion Times', 
                               self.output_dir / "hist_human_minutes.png")

        # Score distribution  
        if 'score_cont' in self.df.columns:
            self._plot_histogram(self.df, 'score_cont', 'Distribution of Continuous Scores', 
                               self.output_dir / "hist_score_cont.png")

        # Task family analysis
        if hasattr(self, 'task_family_df') and self.task_family_df is not None and not self.task_family_df.empty:
            if 'human_minutes_median' in self.task_family_df.columns:
                self._plot_bar_chart(self.task_family_df, 'task_family', 'human_minutes_median', 
                                   'Median Completion Time by Task Family', 
                                   self.output_dir / "bar_median_time_by_task_family.png", 
                                   y_label="Median Time (minutes)")
            if 'task_id_count' in self.task_family_df.columns:
                self._plot_bar_chart(self.task_family_df, 'task_family', 'task_id_count', 
                                   'Record Count by Task Family', 
                                   self.output_dir / "bar_count_by_task_family.png", 
                                   y_label="Number of Records")
            if 'success_rate_percent' in self.task_family_df.columns:
                self._plot_bar_chart(self.task_family_df, 'task_family', 'success_rate_percent', 
                                   'Success Rate by Task Family', 
                                   self.output_dir / "bar_success_rate_by_task_family.png", 
                                   y_label="Success Rate (%)")

        logger.info("Standard METR format plots generated")

    @abstractmethod
    def summarise(self) -> None:
        """
        Generates summary statistics from the loaded DataFrame.
        Subclasses should implement this to save specific stats (e.g., CSVs).
        """
        if self.df is None or self.df.empty:
            logger.warning("DataFrame not loaded or empty. Skipping summary generation.")
            return
        
        # Generate standard METR format analysis
        self._generate_standard_overall_stats(self.df)
        self.task_family_df = self._generate_task_family_stats(self.df)

    def save_plots(self) -> None:
        """
        Generates and saves plots from the loaded DataFrame.
        Base implementation creates standard METR format plots.
        Subclasses can override to add dataset-specific visualizations.
        """
        if self.df is None or self.df.empty:
            logger.warning("DataFrame not loaded or empty. Skipping plot generation.")
            return
        
        # Generate standard plots
        self._generate_standard_plots()
