"""
CyBench-specific describer.

Generates custom analyses for the CyBench dataset, such as stats per event.
"""
import logging
from pathlib import Path
from typing import List

from human_ttc_eval.core import Describe, register_describer

logger = logging.getLogger(__name__)

@register_describer("cybench")
class CybenchDescribe(Describe):
    """
    CyBench-specific implementation of the Describe class.
    Provides custom analysis based on CyBench task structures (e.g., events).
    """

    def __init__(self, input_files: List[Path], output_dir: Path):
        """
        Args:
            input_files: List of Paths to the processed JSONL files for CyBench 
                         (e.g., data/processed/cybench/cybench_human_runs.jsonl).
            output_dir: Directory to save the generated summaries and plots for CyBench.
        """
        super().__init__(input_files=input_files, output_dir=output_dir)

    @property
    def dataset_name(self) -> str:
        return "cybench"

    def _extract_event_from_task_id(self, task_id: str) -> str:
        """
        Extracts the event name from the CyBench task_id (task_path_in_repo).
        Expected format: benchmark/{organization}/{event}/{category}/{challenge_name}
        Returns 'UnknownEvent' if parsing fails.
        """
        parts = task_id.split('/')
        if len(parts) >= 3 and parts[0] == 'benchmark':
            return parts[2] # {event} is the 3rd component (index 2)
        logger.warning(f"Could not parse event from task_id: {task_id}")
        return "UnknownEvent"

    def generate_custom_analysis(self) -> None:
        """
        Generates CyBench-specific analyses, focusing on event-based statistics.
        Saves the analysis to a CSV file in the output directory.
        """
        if self.df is None or self.df.empty:
            logger.warning(f"No data loaded for {self.dataset_name}, skipping custom CyBench analysis.")
            return

        logger.info(f"Generating custom CyBench analysis for {self.dataset_name}...")

        if self.df['task_id'].nunique() < len(self.df):
            logger.debug("DataFrame contains multiple runs per task_id. Aggregating to unique tasks first.")
            unique_task_df = self.df.sort_values(by=['task_id', 'model']).drop_duplicates(subset='task_id', keep='first')
        else:
            unique_task_df = self.df.copy()

        # Extract event for each task
        unique_task_df['event'] = unique_task_df['task_id'].apply(self._extract_event_from_task_id)

        if (unique_task_df['event'] == "UnknownEvent").all():
            logger.warning(f"All task_ids resulted in 'UnknownEvent' for {self.dataset_name}. "
                           "Skipping event-based custom analysis as no specific events found.")
            return

        # Define aggregations for event stats
        agg_functions = {
            'task_id': 'count',
            'human_minutes': ['sum', 'mean', 'median'],
            'score_binarized': ['sum', 'mean'] # Sum for successful tasks, mean for success rate
        }
        
        # Filter out columns not present in df to avoid KeyError in .agg()
        cols_present = [col for col in agg_functions.keys() if col in unique_task_df.columns]
        agg_to_apply = {key: val for key, val in agg_functions.items() if key in cols_present}

        if not agg_to_apply:
            logger.warning("No suitable columns found for event aggregation. Skipping.")
            return

        event_stats_df = unique_task_df.groupby('event').agg(agg_to_apply)

        # Flatten MultiIndex columns (e.g., from ('human_minutes', 'sum') to 'human_minutes_sum')
        event_stats_df.columns = ['_'.join(col_name).strip() if isinstance(col_name, tuple) else col_name 
                                  for col_name in event_stats_df.columns.values]
        event_stats_df = event_stats_df.reset_index() # Make 'event' a column again

        # Rename columns for clarity
        column_renames = {
            'task_id_count': 'num_tasks',
            'human_minutes_sum': 'total_human_minutes',
            'human_minutes_mean': 'mean_human_minutes',
            'human_minutes_median': 'median_human_minutes',
            'score_binarized_sum': 'successful_tasks',
            'score_binarized_mean': 'success_rate'
        }
        event_stats_df.rename(columns=lambda c: column_renames.get(c, c), inplace=True)

        if 'success_rate' in event_stats_df.columns:
            event_stats_df['success_rate'] = event_stats_df['success_rate'] * 100

        if not event_stats_df.empty:
            output_path = self.output_dir / f"{self.dataset_name}_event_summary.csv"
            try:
                event_stats_df.to_csv(output_path, index=False)
                logger.info(f"Custom CyBench event summary saved to: {output_path}")
            except Exception as e:
                logger.error(f"Failed to save CyBench event summary to {output_path}: {e}", exc_info=True)
        else:
            logger.warning("No event statistics generated for CyBench custom analysis.") 