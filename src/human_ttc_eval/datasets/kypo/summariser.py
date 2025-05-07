import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import logging
from typing import Optional, List, Dict, Any # Added List, Dict, Any

from ...core.base_summariser import BaseSummariser
from ...core.registry import register_summariser

logger = logging.getLogger(__name__)

# Constants for plotting if any were module-level, otherwise define in class or methods

# Helper functions for plotting (migrated from summarise_datasets.py)
# These could be methods of the class or static/module-level helpers if preferred.
# For now, making them private static methods or module-level functions prefixed with _

def _plot_histograms(df: pd.DataFrame, column: str, title: str, output_path: Path, bins=30):
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

def _plot_bar_chart(data: pd.DataFrame, x_col: str, y_col: str, title: str, output_path: Path, y_label: str = None):
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

def _plot_scatter(df: pd.DataFrame, x_col: str, y_col: str, hue_col: Optional[str], title: str, output_path: Path):
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

@register_summariser("kypo")
class KypoSummariser(BaseSummariser):
    """Summarises KYPO dataset results, generating stats and plots."""

    def __init__(self, jsonl_file_path: Path, output_dir: Path):
        super().__init__(jsonl_file_path, output_dir)
        # KYPO specific initialization if any
        logger.info(f"KypoSummariser initialized. Input JSONL: {self.jsonl_file_path}, Output Dir: {self.output_dir}")

    def _generate_overall_stats(self, df: pd.DataFrame) -> None:
        output_path = self.output_dir / "overall_summary_stats.csv"
        if df.empty:
            logger.warning("DataFrame is empty. Skipping overall stats generation.")
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write("No data available to generate overall statistics.\n")
            return

        total_sessions = len(df)
        successful_sessions_df = df[df['score_binarized'] == 1]
        num_successful_sessions = len(successful_sessions_df)
        overall_success_rate = (num_successful_sessions / total_sessions * 100) if total_sessions > 0 else 0
        
        stats = {
            "Total Sessions Processed": total_sessions,
            "Total Unique Task Families": df['task_family'].nunique() if 'task_family' in df.columns else 0,
            "Total Unique Task IDs": df['task_id'].nunique() if 'task_id' in df.columns else 0,
            "Number of Successful Sessions (score_binarized=1)": num_successful_sessions,
            "Overall Success Rate (%)": f"{overall_success_rate:.2f}",
        }

        if not successful_sessions_df.empty and 'human_minutes' in successful_sessions_df.columns and 'command_count' in successful_sessions_df.columns:
            stats.update({
                "Duration (human_minutes) - Successful Sessions - Min": successful_sessions_df['human_minutes'].min(),
                "Duration (human_minutes) - Successful Sessions - Max": successful_sessions_df['human_minutes'].max(),
                "Duration (human_minutes) - Successful Sessions - Mean": successful_sessions_df['human_minutes'].mean(),
                "Duration (human_minutes) - Successful Sessions - Median": successful_sessions_df['human_minutes'].median(),
                "Duration (human_minutes) - Successful Sessions - StdDev": successful_sessions_df['human_minutes'].std(),
                "Command Count (command_count) - Successful Sessions - Min": successful_sessions_df['command_count'].min(),
                "Command Count (command_count) - Successful Sessions - Max": successful_sessions_df['command_count'].max(),
                "Command Count (command_count) - Successful Sessions - Mean": successful_sessions_df['command_count'].mean(),
                "Command Count (command_count) - Successful Sessions - Median": successful_sessions_df['command_count'].median(),
                "Command Count (command_count) - Successful Sessions - StdDev": successful_sessions_df['command_count'].std(),
            })
        else:
            logger.warning("No successful sessions or required columns missing. Duration/command stats for successful sessions will be N/A.")

        stats_df = pd.DataFrame.from_dict(stats, orient='index', columns=['Value'])
        stats_df.to_csv(output_path)
        logger.info(f"Overall statistics saved to {output_path}")

    def _generate_per_task_family_stats(self, df: pd.DataFrame) -> Optional[pd.DataFrame]:
        output_path = self.output_dir / "per_task_family_stats.csv"
        if df.empty or 'task_family' not in df.columns:
            logger.warning("DataFrame is empty or missing 'task_family'. Skipping per-task_family stats.")
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write("No data available to generate per-task family statistics.\n")
            return None

        # Ensure columns used in agg exist and handle missing ones gracefully
        agg_funcs = {
            'total_sessions': pd.NamedAgg(column='task_id', aggfunc='count')
        }
        if 'score_binarized' in df.columns:
            agg_funcs['successful_sessions'] = pd.NamedAgg(column='score_binarized', aggfunc='sum')
            if 'human_minutes' in df.columns:
                 agg_funcs['median_duration_successful_min'] = pd.NamedAgg(column='human_minutes', aggfunc=lambda x: x[df.loc[x.index, 'score_binarized'] == 1].median() if 'score_binarized' in df.columns and not x[df.loc[x.index, 'score_binarized'] == 1].empty else None)
            if 'command_count' in df.columns:
                agg_funcs['median_commands_successful'] = pd.NamedAgg(column='command_count', aggfunc=lambda x: x[df.loc[x.index, 'score_binarized'] == 1].median() if 'score_binarized' in df.columns and not x[df.loc[x.index, 'score_binarized'] == 1].empty else None)
        
        grouped = df.groupby('task_family').agg(**agg_funcs).reset_index()
        
        if 'successful_sessions' in grouped.columns and 'total_sessions' in grouped.columns:
            grouped['success_rate_perc'] = (grouped['successful_sessions'] / grouped['total_sessions'] * 100).round(2)
        else:
            grouped['success_rate_perc'] = 0.0 # Or None
            
        grouped.rename(columns={'task_family': 'Task Family'}, inplace=True)
        
        grouped.to_csv(output_path, index=False)
        logger.info(f"Per-task_family statistics saved to {output_path}")
        return grouped

    def summarise(self) -> None:
        super().summarise() # Call base to check df
        if self.df is None or self.df.empty:
            logger.error("No data loaded by BaseSummariser. Exiting KypoSummariser.summarise.")
            return
        
        self._generate_overall_stats(self.df)
        self.per_task_family_df = self._generate_per_task_family_stats(self.df) # Store for plotting

    def save_plots(self) -> None:
        super().save_plots() # Call base to check df
        if self.df is None or self.df.empty:
            logger.error("No data loaded by BaseSummariser. Exiting KypoSummariser.save_plots.")
            return

        if 'human_minutes' in self.df.columns:
            _plot_histograms(self.df, 'human_minutes', 'Distribution of Session Durations (All Valid Sessions)', self.output_dir / "hist_duration_all.png")
            successful_sessions_df = self.df[self.df['score_binarized'] == 1]
            if not successful_sessions_df.empty:
                _plot_histograms(successful_sessions_df, 'human_minutes', 'Distribution of Session Durations (Successful Sessions Only)', self.output_dir / "hist_duration_successful.png")
            else:
                logger.warning("No successful sessions to plot duration histogram.")

        if 'command_count' in self.df.columns:
            _plot_histograms(self.df, 'command_count', 'Distribution of Command Counts (All Valid Sessions)', self.output_dir / "hist_command_count_all.png")

        if hasattr(self, 'per_task_family_df') and self.per_task_family_df is not None and not self.per_task_family_df.empty:
            if 'success_rate_perc' in self.per_task_family_df.columns:
                 _plot_bar_chart(self.per_task_family_df, 'Task Family', 'success_rate_perc', 'Success Rate by Task Family', self.output_dir / "bar_success_rate_per_task_family.png", y_label="Success Rate (%)")
            if 'median_duration_successful_min' in self.per_task_family_df.columns:
                _plot_bar_chart(self.per_task_family_df, 'Task Family', 'median_duration_successful_min', 'Median Duration (Successful) by Task Family', self.output_dir / "bar_median_duration_per_task_family.png", y_label="Median Duration (minutes)")
        else:
            logger.warning("Per-task family stats not available for plotting bar charts.")

        if 'command_count' in self.df.columns and 'human_minutes' in self.df.columns:
            hue_col = 'score_binarized' if 'score_binarized' in self.df.columns else None
            _plot_scatter(self.df, 'command_count', 'human_minutes', hue_col=hue_col, title='Command Count vs. Duration (Colored by Success)', output_path=self.output_dir / "scatter_commands_vs_duration.png")
        
        logger.info(f"KYPO dataset plots saved to {self.output_dir}")

# Example of how this might be used by a CLI later:
# if __name__ == '__main__':
#     import sys
#     if len(sys.argv) != 3:
#         print("Usage: python -m human_ttc_eval.datasets.kypo.summariser <path_to_input.jsonl> <path_to_output_dir>")
#         sys.exit(1)

#     input_jsonl = Path(sys.argv[1])
#     output_results_dir = Path(sys.argv[2])
    
#     # Setup basic logging for standalone script run
#     logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(module)s - %(message)s')

#     summariser = KypoSummariser(jsonl_file_path=input_jsonl, output_dir=output_results_dir)
#     summariser.load_data() # Provided by BaseSummariser
#     summariser.summarise()
#     summariser.save_plots()
#     logger.info("KYPO summarisation complete.") 