"""
CyBashBench dataset describer.

Generates summary statistics and visualizations specific to the CyBashBench dataset,
analyzing the distribution of different task types and security categories.
"""

import json
import logging
from pathlib import Path
from typing import List, Dict, Any
from collections import Counter
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from human_ttc_eval.core.describe import Describe
from human_ttc_eval.core.registry import register_describer

logger = logging.getLogger(__name__)


@register_describer("cybashbench")
class CyBashBenchDescribe(Describe):
    """
    CyBashBench-specific implementation of the Describe class.
    
    Adds custom analyses for task types (e.g., nl2bash, contextual) and
    security category distributions.
    """
    
    @property
    def dataset_name(self) -> str:
        """Returns the dataset identifier."""
        return "cybashbench"
    
    def generate_custom_analysis(self) -> None:
        """Generate CyBashBench-specific analyses."""
        if self.df is None or self.df.empty:
            logger.warning("No data loaded for custom CyBashBench analysis")
            return
        
        logger.info("Generating custom CyBashBench analyses...")
        
        task_metadata = self._load_task_metadata()
        
        if task_metadata:
            self._generate_task_type_analysis(task_metadata)
            self._generate_security_category_analysis(task_metadata)
            self._generate_custom_plots(task_metadata)
        else:
            logger.warning("Could not load task metadata for detailed analysis")
    
    def _load_task_metadata(self) -> Dict[str, Dict[str, Any]]:
        """Load task metadata from the tasks.jsonl file."""
        tasks_file = self.input_files[0].parent / f"{self.dataset_name}_tasks.jsonl"
        
        if not tasks_file.exists():
            logger.warning(f"Tasks file not found: {tasks_file}")
            return {}
        
        task_metadata = {}
        try:
            with open(tasks_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        task = json.loads(line)
                        if 'task_id' in task and 'dataset_task_metadata' in task:
                            task_metadata[task['task_id']] = task['dataset_task_metadata']
            logger.info(f"Loaded metadata for {len(task_metadata)} tasks")
        except Exception as e:
            logger.error(f"Error loading task metadata: {e}")
        
        return task_metadata
    
    def _generate_task_type_analysis(self, task_metadata: Dict[str, Dict[str, Any]]):
        """Generate analysis of task types."""
        task_types = [metadata.get('task_type', 'unknown') for metadata in task_metadata.values()]
        type_counts = Counter(task_types)
        
        type_df = pd.DataFrame(type_counts.items(), columns=['Task_Type', 'Count'])
        type_df['Percentage'] = (type_df['Count'] / len(task_metadata) * 100).round(2)
        
        type_df.to_csv(self.output_dir / 'task_type_distribution.csv', index=False)
        logger.info("Saved task type distribution analysis.")

    def _generate_security_category_analysis(self, task_metadata: Dict[str, Dict[str, Any]]):
        """Generate analysis of security categories."""
        sec_categories = [metadata.get('security_category', 'unknown') for metadata in task_metadata.values()]
        cat_counts = Counter(sec_categories)
        
        cat_df = pd.DataFrame(cat_counts.items(), columns=['Security_Category', 'Count'])
        cat_df['Percentage'] = (cat_df['Count'] / len(task_metadata) * 100).round(2)
        
        cat_df.to_csv(self.output_dir / 'security_category_distribution.csv', index=False)
        logger.info("Saved security category distribution analysis.")
    
    def _generate_custom_plots(self, task_metadata: Dict[str, Dict[str, Any]]):
        """Generate CyBashBench-specific visualizations."""
        metadata_df = pd.DataFrame.from_dict(task_metadata, orient='index')
        metadata_df['human_minutes'] = self.df.set_index('task_id')['human_minutes']
        
        # Plot 1: Task Type Distribution
        plt.figure(figsize=(10, 6))
        sns.countplot(data=metadata_df, y='task_type', order=metadata_df['task_type'].value_counts().index)
        plt.title('Distribution of Task Types')
        plt.xlabel('Count')
        plt.ylabel('Task Type')
        plt.tight_layout()
        plt.savefig(self.output_dir / 'task_type_distribution.png')
        plt.close()
        logger.info("Saved task type distribution plot.")

        # Plot 2: Security Category Distribution
        plt.figure(figsize=(10, 6))
        sns.countplot(data=metadata_df, y='security_category', order=metadata_df['security_category'].value_counts().index)
        plt.title('Distribution of Security Categories')
        plt.xlabel('Count')
        plt.ylabel('Security Category')
        plt.tight_layout()
        plt.savefig(self.output_dir / 'security_category_distribution.png')
        plt.close()
        logger.info("Saved security category distribution plot.")

        # Plot 3: Time distribution by Task Type
        plt.figure(figsize=(12, 7))
        sns.boxplot(data=metadata_df, x='human_minutes', y='task_type')
        plt.title('Human Time Distribution by Task Type')
        plt.xlabel('Human Time (minutes)')
        plt.ylabel('Task Type')
        plt.tight_layout()
        plt.savefig(self.output_dir / 'time_dist_by_task_type.png')
        plt.close()
        logger.info("Saved time distribution by task type plot.")
        
        # Plot 4: Time distribution by Security Category
        plt.figure(figsize=(12, 7))
        sns.boxplot(data=metadata_df, x='human_minutes', y='security_category')
        plt.title('Human Time Distribution by Security Category')
        plt.xlabel('Human Time (minutes)')
        plt.ylabel('Security Category')
        plt.tight_layout()
        plt.savefig(self.output_dir / 'time_dist_by_sec_category.png')
        plt.close()
        logger.info("Saved time distribution by security category plot.") 