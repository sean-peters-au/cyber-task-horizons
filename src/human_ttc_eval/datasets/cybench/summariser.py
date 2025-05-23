"""Summarise CyBench dataset results, generating stats and plots."""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import logging
from typing import Optional

from ...core.base_summariser import BaseSummariser
from ...core.registry import register_summariser

logger = logging.getLogger(__name__)

@register_summariser("cybench")
class CyBenchSummariser(BaseSummariser):
    """Summarises CyBench dataset results, generating stats and plots."""

    def __init__(self, jsonl_file_path: Path, output_dir: Path):
        super().__init__(jsonl_file_path, output_dir)
        logger.info(f"CyBenchSummariser initialized. Input JSONL: {self.jsonl_file_path}, Output Dir: {self.output_dir}")

    def _generate_cybench_specific_stats(self, df: pd.DataFrame) -> None:
        """Generate CyBench-specific statistics."""
        output_path = self.output_dir / "cybench_specific_stats.csv"
        if df.empty:
            logger.warning("DataFrame is empty. Skipping CyBench-specific stats generation.")
            return

        stats = {}

        # CyBench-specific dimensions
        if 'organization' in df.columns:
            stats["Total Organizations"] = df['organization'].nunique()
        if 'event' in df.columns:
            stats["Total Events"] = df['event'].nunique()
        if 'category' in df.columns:
            stats["Total Categories"] = df['category'].nunique()
        if 'difficulty' in df.columns:
            stats["Total Difficulty Levels"] = df['difficulty'].nunique()

        # Additional timing analysis (hours for CyBench's longer tasks)
        if 'human_minutes' in df.columns and not df['human_minutes'].isnull().all():
            stats.update({
                "Solve Time (hours) - Min": df['human_minutes'].min() / 60,
                "Solve Time (hours) - Max": df['human_minutes'].max() / 60,
                "Solve Time (hours) - Mean": df['human_minutes'].mean() / 60,
                "Solve Time (hours) - Median": df['human_minutes'].median() / 60,
            })

        if stats:  # Only save if we have stats
            stats_df = pd.DataFrame.from_dict(stats, orient='index', columns=['Value'])
            stats_df.to_csv(output_path)
            logger.info(f"CyBench-specific statistics saved to {output_path}")

    def _generate_per_organization_stats(self, df: pd.DataFrame) -> Optional[pd.DataFrame]:
        """Generate per-organization statistics."""
        output_path = self.output_dir / "per_organization_stats.csv"
        if df.empty or 'organization' not in df.columns:
            logger.warning("DataFrame is empty or missing 'organization'. Skipping per-organization stats.")
            return None

        grouped = df.groupby('organization').agg({
            'task_id': 'count',
            'human_minutes': ['min', 'max', 'mean', 'median', 'std'],
            'category': 'nunique',
            'event': 'nunique'
        }).round(2)
        
        # Flatten column names
        grouped.columns = ['_'.join(col).strip() for col in grouped.columns.values]
        grouped = grouped.reset_index()
        
        grouped.to_csv(output_path, index=False)
        logger.info(f"Per-organization statistics saved to {output_path}")
        return grouped

    def _generate_per_category_stats(self, df: pd.DataFrame) -> Optional[pd.DataFrame]:
        """Generate per-category statistics."""
        output_path = self.output_dir / "per_category_stats.csv"
        if df.empty or 'category' not in df.columns:
            logger.warning("DataFrame is empty or missing 'category'. Skipping per-category stats.")
            return None

        grouped = df.groupby('category').agg({
            'task_id': 'count',
            'human_minutes': ['min', 'max', 'mean', 'median', 'std'],
            'organization': 'nunique'
        }).round(2)
        
        # Flatten column names
        grouped.columns = ['_'.join(col).strip() for col in grouped.columns.values]
        grouped = grouped.reset_index()
        
        grouped.to_csv(output_path, index=False)
        logger.info(f"Per-category statistics saved to {output_path}")
        return grouped

    def summarise(self) -> None:
        """Generate comprehensive summary statistics for CyBench dataset."""
        # Call parent to generate standard METR format analysis
        super().summarise()
        
        if self.df is None or self.df.empty:
            logger.error("No data loaded by BaseSummariser. Exiting CyBenchSummariser.summarise.")
            return
        
        # Add CyBench-specific analysis
        self._generate_cybench_specific_stats(self.df)
        self.per_organization_df = self._generate_per_organization_stats(self.df)
        self.per_category_df = self._generate_per_category_stats(self.df)

    def save_plots(self) -> None:
        """Generate CyBench-specific plots in addition to standard ones."""
        # Call parent to generate standard METR format plots
        super().save_plots()
        
        if self.df is None or self.df.empty:
            logger.error("No data loaded by BaseSummariser. Exiting CyBenchSummariser.save_plots.")
            return

        # CyBench-specific plots

        # Also plot timing in hours for better readability of long solve times
        if 'human_minutes' in self.df.columns:
            self.df['human_hours'] = self.df['human_minutes'] / 60
            self._plot_histogram(self.df, 'human_hours', 'Distribution of Human Solve Times (Hours)', 
                               self.output_dir / "hist_solve_times_hours.png")

        # Bar charts by organization
        if hasattr(self, 'per_organization_df') and self.per_organization_df is not None and not self.per_organization_df.empty:
            if 'human_minutes_median' in self.per_organization_df.columns:
                self._plot_bar_chart(self.per_organization_df, 'organization', 'human_minutes_median', 
                                   'Median Solve Time by Organization', 
                                   self.output_dir / "bar_median_time_by_organization.png", 
                                   y_label="Median Solve Time (minutes)")
            if 'task_id_count' in self.per_organization_df.columns:
                self._plot_bar_chart(self.per_organization_df, 'organization', 'task_id_count', 
                                   'Task Count by Organization', 
                                   self.output_dir / "bar_task_count_by_organization.png", 
                                   y_label="Number of Tasks")

        # Bar charts by category
        if hasattr(self, 'per_category_df') and self.per_category_df is not None and not self.per_category_df.empty:
            if 'human_minutes_median' in self.per_category_df.columns:
                self._plot_bar_chart(self.per_category_df, 'category', 'human_minutes_median', 
                                   'Median Solve Time by Category', 
                                   self.output_dir / "bar_median_time_by_category.png", 
                                   y_label="Median Solve Time (minutes)")
            if 'task_id_count' in self.per_category_df.columns:
                self._plot_bar_chart(self.per_category_df, 'category', 'task_id_count', 
                                   'Task Count by Category', 
                                   self.output_dir / "bar_task_count_by_category.png", 
                                   y_label="Number of Tasks")

        # Strip plots for detailed distributions
        if 'category' in self.df.columns and 'human_minutes' in self.df.columns:
            plt.figure(figsize=(12, 8))
            sns.stripplot(x='category', y='human_minutes', data=self.df, size=8, alpha=0.7)
            plt.title('Solve Time Distribution by Category')
            plt.xlabel('Category')
            plt.ylabel('Solve Time (minutes)')
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            plt.savefig(self.output_dir / "strip_solve_time_by_category.png")
            plt.close()
            logger.info(f"Strip plot saved to {self.output_dir / 'strip_solve_time_by_category.png'}")

        if 'organization' in self.df.columns and 'human_minutes' in self.df.columns:
            plt.figure(figsize=(12, 8))
            sns.stripplot(x='organization', y='human_minutes', data=self.df, size=8, alpha=0.7)
            plt.title('Solve Time Distribution by Organization')
            plt.xlabel('Organization')
            plt.ylabel('Solve Time (minutes)')
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            plt.savefig(self.output_dir / "strip_solve_time_by_organization.png")
            plt.close()
            logger.info(f"Strip plot saved to {self.output_dir / 'strip_solve_time_by_organization.png'}")

        # Difficulty analysis (if available)
        if 'difficulty' in self.df.columns and 'human_minutes' in self.df.columns:
            difficulty_df = self.df[self.df['difficulty'].notna()]
            if not difficulty_df.empty:
                plt.figure(figsize=(10, 6))
                sns.boxplot(x='difficulty', y='human_minutes', data=difficulty_df)
                plt.title('Solve Time Distribution by Difficulty Level')
                plt.xlabel('Difficulty')
                plt.ylabel('Solve Time (minutes)')
                plt.xticks(rotation=45, ha='right')
                plt.tight_layout()
                plt.savefig(self.output_dir / "box_solve_time_by_difficulty.png")
                plt.close()
                logger.info(f"Box plot saved to {self.output_dir / 'box_solve_time_by_difficulty.png'}")

        logger.info(f"CyBench dataset plots saved to {self.output_dir}") 