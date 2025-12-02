"""
CyberGym dataset describer.

Generates summary statistics and visualizations specific to the CyberGym dataset,
building on the standard analyses provided by the base Describe class.
"""

import json
import logging
from typing import Dict, Any, List
from collections import Counter
import pandas as pd
import matplotlib.pyplot as plt

from human_ttc_eval.core.describe import Describe
from human_ttc_eval.core.registry import register_describer

logger = logging.getLogger(__name__)


@register_describer("cybergym")
class CybergymDescribe(Describe):
    """
    CyberGym specific implementation of the Describe class.
    
    Adds custom analyses for vulnerability exploitation tasks including
    task type distributions, project language analysis, and difficulty levels.
    """

    @property
    def dataset_name(self) -> str:
        """Returns the dataset identifier."""
        return "cybergym"

    def generate_custom_analysis(self) -> None:
        """
        Generate CyberGym specific analyses.
        
        This includes:
        - Task type distribution (arvo vs oss-fuzz)
        - Project language distribution
        - Difficulty level analysis
        """
        if self.df is None or self.df.empty:
            logger.warning("No data loaded for custom CyberGym analysis")
            return

        logger.info("Generating custom CyberGym analyses...")
        
        # Load task metadata from tasks.jsonl for detailed analysis
        task_metadata = self._load_task_metadata()
        
        if task_metadata:
            self._generate_task_type_analysis()
            self._generate_language_analysis(task_metadata)
            self._generate_custom_plots(task_metadata)
        else:
            logger.warning("Could not load task metadata for detailed analysis")
    
    def _load_task_metadata(self) -> Dict[str, Dict[str, Any]]:
        """Load task metadata from tasks.jsonl file."""
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
    
    def _generate_task_type_analysis(self) -> None:
        """Generate analysis of task types in the dataset."""
        if 'task_family' not in self.df.columns:
            logger.warning("No task_family column found for type analysis")
            return
        
        # Extract task types from task_family (format: cybergym_{type})
        self.df['task_type'] = self.df['task_family'].str.replace('cybergym_', '')
        
        # Type statistics
        type_stats = []
        
        for task_type in self.df['task_type'].unique():
            type_df = self.df[self.df['task_type'] == task_type]
            
            stats = {
                'Task_Type': task_type,
                'Count': len(type_df),
                'Percentage': round((len(type_df) / len(self.df)) * 100, 2),
                'Mean_Minutes': round(type_df['human_minutes'].mean(), 2),
            }
            type_stats.append(stats)
        
        type_stats_df = pd.DataFrame(type_stats)
        type_stats_df = type_stats_df.sort_values('Count', ascending=False)
        type_stats_df.to_csv(self.output_dir / 'task_type_analysis.csv', index=False)
        logger.info("Saved task type analysis")
    
    def _generate_language_analysis(self, task_metadata: Dict[str, Dict[str, Any]]) -> None:
        """Generate analysis of project programming languages."""
        languages = []
        
        for task_id, metadata in task_metadata.items():
            language = metadata.get('project_language', 'Unknown')
            languages.append(language)
        
        # Language distribution
        lang_counts = Counter(languages)
        lang_data = []
        
        for lang, count in lang_counts.most_common():
            lang_data.append({
                'Language': lang,
                'Task_Count': count,
                'Percentage': round((count / len(languages)) * 100, 2),
            })
        
        lang_df = pd.DataFrame(lang_data)
        lang_df.to_csv(self.output_dir / 'language_analysis.csv', index=False)
        logger.info("Saved language analysis")
    
    def _generate_custom_plots(self, task_metadata: Dict[str, Dict[str, Any]]) -> None:
        """Generate CyberGym specific visualizations."""
        # Plot 1: Task type distribution
        self._plot_task_type_distribution()
        
        # Plot 2: Language distribution
        self._plot_language_distribution(task_metadata)
    
    def _plot_task_type_distribution(self) -> None:
        """Create bar chart of tasks per type."""
        if 'task_type' not in self.df.columns:
            return
        
        type_counts = self.df['task_type'].value_counts()
        
        plt.figure(figsize=(10, 6))
        colors = plt.cm.Set2(range(len(type_counts)))
        bars = plt.bar(range(len(type_counts)), type_counts.values, color=colors)
        plt.xticks(range(len(type_counts)), type_counts.index, rotation=45, ha='right')
        plt.xlabel('Task Type')
        plt.ylabel('Number of Tasks')
        plt.title('CyberGym Distribution by Task Type')
        
        # Add value labels on bars
        for bar, count in zip(bars, type_counts.values):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                    str(count), ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'task_type_distribution.png')
        plt.close()
        logger.info("Saved task type distribution plot")
    
    def _plot_language_distribution(self, task_metadata: Dict[str, Dict[str, Any]]) -> None:
        """Create pie chart of programming language distribution."""
        languages = [
            metadata.get('project_language', 'Unknown') 
            for metadata in task_metadata.values()
        ]
        
        lang_counts = Counter(languages)
        
        plt.figure(figsize=(8, 8))
        colors = plt.cm.Paired(range(len(lang_counts)))
        plt.pie(
            lang_counts.values(), 
            labels=lang_counts.keys(), 
            colors=colors,
            autopct='%1.1f%%',
            startangle=90
        )
        plt.title('CyberGym Distribution by Programming Language')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'language_distribution.png')
        plt.close()
        logger.info("Saved language distribution plot")

