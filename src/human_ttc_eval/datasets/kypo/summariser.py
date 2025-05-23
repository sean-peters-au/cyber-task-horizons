"""Summarise KYPO dataset results, generating stats and plots."""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import logging
from typing import Optional

from ...core.base_summariser import BaseSummariser
from ...core.registry import register_summariser

logger = logging.getLogger(__name__)

@register_summariser("kypo")
class KypoSummariser(BaseSummariser):
    """Summarises KYPO dataset results, generating stats and plots."""

    def __init__(self, jsonl_file_path: Path, output_dir: Path):
        super().__init__(jsonl_file_path, output_dir)
        logger.info(f"KypoSummariser initialized. Input JSONL: {self.jsonl_file_path}, Output Dir: {self.output_dir}")

    def _generate_kypo_specific_stats(self, df: pd.DataFrame) -> None:
        """Generate KYPO-specific statistics."""
        output_path = self.output_dir / "kypo_specific_stats.csv"
        if df.empty:
            logger.warning("DataFrame is empty. Skipping KYPO-specific stats generation.")
            return

        stats = {}

        # KYPO-specific: Focus on successful sessions for detailed analysis
        if 'score_binarized' in df.columns:
            successful_sessions_df = df[df['score_binarized'] == 1]
            num_successful_sessions = len(successful_sessions_df)
            
            if num_successful_sessions > 0 and 'human_minutes' in successful_sessions_df.columns:
                stats.update({
                    "Successful Sessions - Duration (min) - Min": successful_sessions_df['human_minutes'].min(),
                    "Successful Sessions - Duration (min) - Max": successful_sessions_df['human_minutes'].max(),
                    "Successful Sessions - Duration (min) - Mean": successful_sessions_df['human_minutes'].mean(),
                    "Successful Sessions - Duration (min) - Median": successful_sessions_df['human_minutes'].median(),
                    "Successful Sessions - Duration (min) - StdDev": successful_sessions_df['human_minutes'].std(),
                })
            
            if num_successful_sessions > 0 and 'command_count' in successful_sessions_df.columns:
                stats.update({
                    "Successful Sessions - Command Count - Min": successful_sessions_df['command_count'].min(),
                    "Successful Sessions - Command Count - Max": successful_sessions_df['command_count'].max(),
                    "Successful Sessions - Command Count - Mean": successful_sessions_df['command_count'].mean(),
                    "Successful Sessions - Command Count - Median": successful_sessions_df['command_count'].median(),
                    "Successful Sessions - Command Count - StdDev": successful_sessions_df['command_count'].std(),
                })

        if stats:  # Only save if we have stats
            stats_df = pd.DataFrame.from_dict(stats, orient='index', columns=['Value'])
            stats_df.to_csv(output_path)
            logger.info(f"KYPO-specific statistics saved to {output_path}")

    def summarise(self) -> None:
        """Generate comprehensive summary statistics for KYPO dataset."""
        # Call parent to generate standard METR format analysis
        super().summarise()
        
        if self.df is None or self.df.empty:
            logger.error("No data loaded by BaseSummariser. Exiting KypoSummariser.summarise.")
            return
        
        # Add KYPO-specific analysis
        self._generate_kypo_specific_stats(self.df)

    def save_plots(self) -> None:
        """Generate KYPO-specific plots in addition to standard ones."""
        # Call parent to generate standard METR format plots
        super().save_plots()
        
        if self.df is None or self.df.empty:
            logger.error("No data loaded by BaseSummariser. Exiting KypoSummariser.save_plots.")
            return

        # KYPO-specific plots

        # Histogram for successful sessions only
        if 'score_binarized' in self.df.columns and 'human_minutes' in self.df.columns:
            successful_sessions_df = self.df[self.df['score_binarized'] == 1]
            if not successful_sessions_df.empty:
                self._plot_histogram(successful_sessions_df, 'human_minutes', 
                                   'Distribution of Session Durations (Successful Sessions Only)', 
                                   self.output_dir / "hist_duration_successful.png")

        # Command count analysis
        if 'command_count' in self.df.columns:
            self._plot_histogram(self.df, 'command_count', 'Distribution of Command Counts (All Valid Sessions)', 
                               self.output_dir / "hist_command_count_all.png")

        # Scatter plot: Command Count vs. Duration
        if 'command_count' in self.df.columns and 'human_minutes' in self.df.columns:
            hue_col = 'score_binarized' if 'score_binarized' in self.df.columns else None
            self._plot_scatter(self.df, 'command_count', 'human_minutes', hue_col=hue_col, 
                             title='Command Count vs. Duration (Colored by Success)', 
                             output_path=self.output_dir / "scatter_commands_vs_duration.png")
        
        logger.info(f"KYPO dataset plots saved to {self.output_dir}")
