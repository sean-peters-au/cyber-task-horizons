"""
CVE-Bench dataset describer.

Generates summary statistics and visualizations specific to the CVE-Bench dataset,
building on the standard analyses provided by the base Describe class.
"""

import json
import logging
from typing import Dict, Any, List
from collections import Counter, defaultdict
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from human_ttc_eval.core.describe import Describe
from human_ttc_eval.core.registry import register_describer

logger = logging.getLogger(__name__)


@register_describer("cvebench")
class CvebenchDescribe(Describe):
    """
    CVE-Bench specific implementation of the Describe class.
    
    Adds custom analyses for CVE exploitation challenges including attack category
    distributions, application analysis, and vulnerability timing patterns.
    """

    @property
    def dataset_name(self) -> str:
        """Returns the dataset identifier."""
        return "cvebench"

    def generate_custom_analysis(self) -> None:
        """
        Generate CVE-Bench specific analyses.
        
        This includes:
        - Attack category distribution
        - Target application analysis
        - Time horizon analysis for METR-style evaluation
        - CVE severity analysis
        """
        if self.df is None or self.df.empty:
            logger.warning("No data loaded for custom CVE-Bench analysis")
            return

        logger.info("Generating custom CVE-Bench analyses...")
        
        # Load task metadata from tasks.jsonl for detailed analysis
        task_metadata = self._load_task_metadata()
        
        if task_metadata:
            self._generate_attack_category_analysis()
            self._generate_application_analysis(task_metadata)
            self._generate_time_horizon_analysis()
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
    
    def _generate_attack_category_analysis(self) -> None:
        """Generate analysis of attack categories in the dataset."""
        if 'task_family' not in self.df.columns:
            logger.warning("No task_family column found for category analysis")
            return
        
        # Extract categories from task_family (format: cvebench_{category})
        self.df['category'] = self.df['task_family'].str.replace('cvebench_', '')
        
        # Category statistics
        category_stats = []
        
        for category in self.df['category'].unique():
            cat_df = self.df[self.df['category'] == category]
            
            stats = {
                'Attack_Category': category,
                'Count': len(cat_df),
                'Percentage': round((len(cat_df) / len(self.df)) * 100, 2),
                'Mean_Minutes': round(cat_df['human_minutes'].mean(), 2),
                'Median_Minutes': round(cat_df['human_minutes'].median(), 2),
                'Min_Minutes': round(cat_df['human_minutes'].min(), 2),
                'Max_Minutes': round(cat_df['human_minutes'].max(), 2),
            }
            category_stats.append(stats)
        
        category_df = pd.DataFrame(category_stats)
        category_df = category_df.sort_values('Count', ascending=False)
        category_df.to_csv(self.output_dir / 'attack_category_analysis.csv', index=False)
        logger.info("Saved attack category analysis")
    
    def _generate_application_analysis(self, task_metadata: Dict[str, Dict[str, Any]]) -> None:
        """Generate analysis of target applications."""
        applications = []
        app_categories = defaultdict(list)
        
        for task_id, metadata in task_metadata.items():
            application = metadata.get('application', 'Unknown')
            category = metadata.get('category', 'unknown')
            applications.append(application)
            app_categories[application].append(category)
        
        # Application distribution
        app_counts = Counter(applications)
        app_data = []
        
        for app, count in app_counts.most_common():
            categories = Counter(app_categories[app])
            app_data.append({
                'Application': app,
                'CVE_Count': count,
                'Percentage': round((count / len(applications)) * 100, 2),
                'Attack_Categories': ', '.join(f"{cat}({cnt})" for cat, cnt in categories.most_common()),
            })
        
        app_df = pd.DataFrame(app_data)
        app_df.to_csv(self.output_dir / 'application_analysis.csv', index=False)
        logger.info("Saved application analysis")
    
    def _generate_time_horizon_analysis(self) -> None:
        """Generate METR-style time horizon analysis."""
        if 'human_minutes' not in self.df.columns:
            return

        # Calculate cumulative success rates at different time horizons
        time_horizons_minutes = [30, 60, 120, 240, 480, 960, 1440]  # 0.5h to 24h
        horizon_stats = []
        
        total_tasks = len(self.df)
        
        for horizon in time_horizons_minutes:
            tasks_within_horizon = len(self.df[self.df['human_minutes'] <= horizon])
            percentage = (tasks_within_horizon / total_tasks) * 100
            
            # Category breakdown at this horizon
            horizon_df = self.df[self.df['human_minutes'] <= horizon]
            category_counts = horizon_df['category'].value_counts() if not horizon_df.empty else {}
            
            horizon_stats.append({
                'Time_Horizon_Hours': round(horizon / 60, 1),
                'Time_Horizon_Minutes': horizon,
                'CVEs_Exploitable': tasks_within_horizon,
                'Percentage_Exploitable': round(percentage, 2),
                'Categories': ', '.join(f"{cat}({cnt})" for cat, cnt in category_counts.items())
            })
        
        horizon_df = pd.DataFrame(horizon_stats)
        horizon_df.to_csv(self.output_dir / 'time_horizon_analysis.csv', index=False)
        logger.info("Saved time horizon analysis")
    
    def _generate_custom_plots(self, task_metadata: Dict[str, Dict[str, Any]]) -> None:
        """Generate CVE-Bench specific visualizations."""
        # Plot 1: Attack category distribution
        self._plot_attack_category_distribution()
        
        # Plot 2: Time distribution by category (violin plot)
        self._plot_time_by_category_violin()
        
        # Plot 3: Time horizon curve (METR-style)
        self._plot_time_horizon_curve()
        
        # Plot 4: Application distribution
        self._plot_application_distribution(task_metadata)
    
    def _plot_attack_category_distribution(self) -> None:
        """Create bar chart of CVEs per attack category."""
        if 'category' not in self.df.columns:
            return
        
        category_counts = self.df['category'].value_counts()
        
        plt.figure(figsize=(12, 6))
        colors = plt.cm.Set3(range(len(category_counts)))
        bars = plt.bar(range(len(category_counts)), category_counts.values, color=colors)
        plt.xticks(range(len(category_counts)), category_counts.index, rotation=45, ha='right')
        plt.xlabel('Attack Category')
        plt.ylabel('Number of CVEs')
        plt.title('CVE-Bench Distribution by Attack Category')
        
        # Add value labels on bars
        for bar, count in zip(bars, category_counts.values):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                    str(count), ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'attack_category_distribution.png')
        plt.close()
        logger.info("Saved attack category distribution plot")
    
    def _plot_time_by_category_violin(self) -> None:
        """Create violin plot of exploitation times by attack category."""
        if 'category' not in self.df.columns or 'human_minutes' not in self.df.columns:
            return
        
        # Order categories by median time
        category_order = (self.df.groupby('category')['human_minutes']
                         .median()
                         .sort_values()
                         .index.tolist())
        
        plt.figure(figsize=(12, 8))
        sns.violinplot(data=self.df, x='category', y='human_minutes', 
                      order=category_order, palette='Set2')
        plt.xlabel('Attack Category')
        plt.ylabel('Exploitation Time (minutes)')
        plt.title('CVE Exploitation Time Distribution by Attack Category')
        plt.xticks(rotation=45, ha='right')
        
        # Add median lines
        medians = self.df.groupby('category')['human_minutes'].median()
        for i, cat in enumerate(category_order):
            plt.hlines(medians[cat], i-0.4, i+0.4, colors='black', linestyles='dashed', alpha=0.7)
        
        # Convert y-axis to hours for readability
        ax = plt.gca()
        y_ticks = ax.get_yticks()
        ax.set_yticklabels([f'{int(y/60)}h' if y >= 60 else f'{int(y)}m' for y in y_ticks])
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'time_by_category_violin.png')
        plt.close()
        logger.info("Saved time by category violin plot")
    
    def _plot_time_horizon_curve(self) -> None:
        """Create METR-style time horizon curve."""
        if 'human_minutes' not in self.df.columns:
            return
        
        # Generate fine-grained time points
        max_time = self.df['human_minutes'].max()
        time_points = np.logspace(np.log10(1), np.log10(max_time), 100)
        
        # Calculate cumulative percentage at each time point
        percentages = []
        for t in time_points:
            pct = (self.df['human_minutes'] <= t).sum() / len(self.df) * 100
            percentages.append(pct)
        
        plt.figure(figsize=(10, 6))
        plt.semilogx(time_points / 60, percentages, 'b-', linewidth=2)
        plt.xlabel('Time Budget (hours)')
        plt.ylabel('Percentage of CVEs Exploitable (%)')
        plt.title('CVE-Bench Human Performance Horizon Curve')
        plt.grid(True, alpha=0.3)
        
        # Add reference lines
        plt.axvline(x=0.5, color='gray', linestyle='--', alpha=0.5, label='30 min')
        plt.axvline(x=1, color='gray', linestyle='--', alpha=0.5, label='1 hour')
        plt.axvline(x=4, color='gray', linestyle='--', alpha=0.5, label='4 hours')
        plt.axvline(x=24, color='gray', linestyle='--', alpha=0.5, label='24 hours')
        
        # Add percentage markers
        for pct in [25, 50, 75, 90]:
            if pct <= max(percentages):
                time_at_pct = np.interp(pct, percentages, time_points) / 60
                plt.axhline(y=pct, color='red', linestyle=':', alpha=0.3)
                plt.text(0.02, pct + 1, f'{pct}% @ {time_at_pct:.1f}h', fontsize=9)
        
        plt.xlim(0.01, max_time / 60 * 1.1)
        plt.ylim(0, 105)
        plt.tight_layout()
        plt.savefig(self.output_dir / 'time_horizon_curve.png')
        plt.close()
        logger.info("Saved time horizon curve")
    
    def _plot_application_distribution(self, task_metadata: Dict[str, Dict[str, Any]]) -> None:
        """Create bar chart of CVEs per target application."""
        applications = [metadata.get('application', 'Unknown') 
                       for metadata in task_metadata.values()]
        
        app_counts = Counter(applications)
        
        # Only show top 15 applications for readability
        top_apps = app_counts.most_common(15)
        
        plt.figure(figsize=(12, 6))
        apps, counts = zip(*top_apps)
        colors = plt.cm.Paired(range(len(apps)))
        bars = plt.barh(range(len(apps)), counts, color=colors)
        plt.yticks(range(len(apps)), apps)
        plt.xlabel('Number of CVEs')
        plt.ylabel('Target Application')
        plt.title('CVE-Bench Distribution by Target Application (Top 15)')
        
        # Add value labels
        for bar, count in zip(bars, counts):
            plt.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height()/2, 
                    str(count), ha='left', va='center')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'application_distribution.png')
        plt.close()
        logger.info("Saved application distribution plot")

