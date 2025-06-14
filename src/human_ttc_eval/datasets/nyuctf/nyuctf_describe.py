"""
NYUCTF dataset describer.

Generates summary statistics and visualizations specific to the NYUCTF dataset,
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


@register_describer("nyuctf")
class NyuctfDescribe(Describe):
    """
    NYUCTF specific implementation of the Describe class.
    
    Adds custom analyses for CSAW CTF challenges including competition
    analysis, category distributions, yearly trends, and event comparisons.
    """

    @property
    def dataset_name(self) -> str:
        """Returns the dataset identifier."""
        return "nyuctf"

    def generate_custom_analysis(self) -> None:
        """
        Generate NYUCTF specific analyses.
        
        This includes:
        - CSAW competition yearly analysis
        - CTF category distribution and performance
        - Finals vs Quals comparison
        - Challenge difficulty analysis
        - Event type analysis (static vs dynamic challenges)
        - Points distribution analysis
        """
        if self.df is None or self.df.empty:
            logger.warning("No data loaded for custom NYUCTF analysis")
            return

        logger.info("Generating custom NYUCTF analyses...")
        
        # Load task metadata from tasks.jsonl for detailed analysis
        task_metadata = self._load_task_metadata()
        
        if task_metadata:
            self._generate_yearly_analysis(task_metadata)
            self._generate_event_analysis(task_metadata)
            self._generate_category_performance_analysis()
            self._generate_challenge_type_analysis(task_metadata)
            self._generate_points_analysis(task_metadata)
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
    
    def _generate_yearly_analysis(self, task_metadata: Dict[str, Dict[str, Any]]) -> None:
        """Generate analysis of CSAW competitions by year."""
        yearly_data = []
        year_categories = defaultdict(lambda: defaultdict(int))
        year_events = defaultdict(lambda: defaultdict(int))
        
        for task_id, metadata in task_metadata.items():
            year = metadata.get('year', 'Unknown')
            category = metadata.get('category', 'unknown')
            event = metadata.get('event', 'Unknown')
            
            yearly_data.append({
                'task_id': task_id,
                'year': year,
                'category': category,
                'event': event
            })
            
            year_categories[year][category] += 1
            year_events[year][event] += 1
        
        # Year distribution with breakdown
        year_summary = []
        for year in sorted(year_categories.keys()):
            categories = year_categories[year]
            events = year_events[year]
            
            year_summary.append({
                'Year': year,
                'Total_Challenges': sum(categories.values()),
                'Categories': ', '.join(f"{cat}({cnt})" for cat, cnt in categories.items()),
                'Num_Categories': len(categories),
                'Finals_Challenges': events.get('CSAW-Finals', 0),
                'Quals_Challenges': events.get('CSAW-Quals', 0),
                'Most_Common_Category': max(categories.items(), key=lambda x: x[1])[0] if categories else 'None'
            })
        
        year_df = pd.DataFrame(year_summary)
        year_df.to_csv(self.output_dir / 'yearly_analysis.csv', index=False)
        logger.info("Saved yearly analysis")
    
    def _generate_event_analysis(self, task_metadata: Dict[str, Dict[str, Any]]) -> None:
        """Generate analysis comparing Finals vs Quals events."""
        event_stats = defaultdict(lambda: defaultdict(int))
        event_categories = defaultdict(Counter)
        
        for task_id, metadata in task_metadata.items():
            event = metadata.get('event', 'Unknown')
            category = metadata.get('category', 'unknown')
            
            event_stats[event]['total'] += 1
            event_categories[event][category] += 1
        
        # Event comparison
        event_data = []
        for event, stats in event_stats.items():
            categories = event_categories[event]
            event_data.append({
                'Event': event,
                'Total_Challenges': stats['total'],
                'Categories': ', '.join(f"{cat}({cnt})" for cat, cnt in categories.most_common()),
                'Num_Categories': len(categories),
                'Most_Common_Category': max(categories.items(), key=lambda x: x[1])[0] if categories else 'None'
            })
        
        event_df = pd.DataFrame(event_data)
        event_df.to_csv(self.output_dir / 'event_analysis.csv', index=False)
        logger.info("Saved event analysis")
    
    def _generate_category_performance_analysis(self) -> None:
        """Generate detailed analysis of performance by CTF category."""
        if 'task_family' not in self.df.columns:
            logger.warning("No task_family column found for category analysis")
            return
        
        # Extract categories from task_family (format: nyuctf_{category})
        self.df['category'] = self.df['task_family'].str.replace('nyuctf_', '')
        
        # Detailed category statistics
        category_stats = []
        
        for category in self.df['category'].unique():
            cat_df = self.df[self.df['category'] == category]
            
            # Time statistics
            time_stats = {
                'Category': category,
                'Count': len(cat_df),
                'Mean_Minutes': round(cat_df['human_minutes'].mean(), 2),
                'Median_Minutes': round(cat_df['human_minutes'].median(), 2),
                'Min_Minutes': round(cat_df['human_minutes'].min(), 2),
                'Max_Minutes': round(cat_df['human_minutes'].max(), 2),
                'Std_Minutes': round(cat_df['human_minutes'].std(), 2),
                'Total_Hours': round(cat_df['human_minutes'].sum() / 60, 2)
            }
            
            # Calculate percentiles for better understanding of distribution
            for percentile in [25, 75, 90]:
                time_stats[f'P{percentile}_Minutes'] = round(
                    cat_df['human_minutes'].quantile(percentile/100), 2
                )
            
            category_stats.append(time_stats)
        
        category_df = pd.DataFrame(category_stats)
        category_df = category_df.sort_values('Count', ascending=False)
        category_df.to_csv(self.output_dir / 'category_performance_analysis.csv', index=False)
        logger.info("Saved category performance analysis")
    
    def _generate_challenge_type_analysis(self, task_metadata: Dict[str, Dict[str, Any]]) -> None:
        """Analyze challenge types (static vs dynamic, Docker support, etc.)."""
        type_stats = []
        docker_support = {'with_docker': 0, 'without_docker': 0}
        challenge_types = defaultdict(int)
        
        for task_id, metadata in task_metadata.items():
            challenge_type = metadata.get('challenge_type', 'static')
            has_docker = metadata.get('has_docker_compose', False)
            category = metadata.get('category', 'unknown')
            
            challenge_types[challenge_type] += 1
            if has_docker:
                docker_support['with_docker'] += 1
            else:
                docker_support['without_docker'] += 1
            
            type_stats.append({
                'task_id': task_id,
                'challenge_type': challenge_type,
                'has_docker_compose': has_docker,
                'category': category
            })
        
        # Challenge type summary
        type_summary = {
            'Total_Challenges': len(task_metadata),
            'Docker_Supported': docker_support['with_docker'],
            'Docker_Percentage': round((docker_support['with_docker'] / len(task_metadata)) * 100, 2),
            'Challenge_Types': dict(challenge_types),
            'Most_Common_Type': max(challenge_types.items(), key=lambda x: x[1])[0] if challenge_types else 'None'
        }
        
        with open(self.output_dir / 'challenge_type_analysis.json', 'w') as f:
            json.dump(type_summary, f, indent=2)
        
        # Detailed breakdown by category
        type_df = pd.DataFrame(type_stats)
        if not type_df.empty:
            type_breakdown = type_df.groupby(['category', 'challenge_type']).size().unstack(fill_value=0)
            type_breakdown.to_csv(self.output_dir / 'challenge_type_breakdown.csv')
        
        logger.info("Saved challenge type analysis")
    
    def _generate_points_analysis(self, task_metadata: Dict[str, Dict[str, Any]]) -> None:
        """Analyze point distributions across challenges."""
        points_data = []
        
        for task_id, metadata in task_metadata.items():
            points = metadata.get('points', 0)
            category = metadata.get('category', 'unknown')
            year = metadata.get('year', 'unknown')
            event = metadata.get('event', 'unknown')
            
            points_data.append({
                'task_id': task_id,
                'points': points,
                'category': category,
                'year': year,
                'event': event
            })
        
        if points_data:
            points_df = pd.DataFrame(points_data)
            
            # Summary statistics
            points_summary = {
                'Average_Points': round(points_df['points'].mean(), 2),
                'Median_Points': int(points_df['points'].median()),
                'Min_Points': int(points_df['points'].min()),
                'Max_Points': int(points_df['points'].max()),
                'Zero_Point_Challenges': int((points_df['points'] == 0).sum())
            }
            
            # Points by category
            points_by_category = points_df.groupby('category')['points'].agg(['mean', 'median', 'count']).round(2)
            points_by_category.to_csv(self.output_dir / 'points_by_category.csv')
            
            with open(self.output_dir / 'points_analysis_summary.json', 'w') as f:
                json.dump(points_summary, f, indent=2)
            
            logger.info("Saved points analysis")
    
    def _generate_custom_plots(self, task_metadata: Dict[str, Dict[str, Any]]) -> None:
        """Generate NYUCTF specific visualizations."""
        # Plot 1: Year distribution
        self._plot_yearly_distribution(task_metadata)
        
        # Plot 2: Category distribution by event type
        self._plot_category_by_event(task_metadata)
        
        # Plot 3: Challenge type distribution
        self._plot_challenge_types(task_metadata)
        
        # Plot 4: Points distribution
        self._plot_points_distribution(task_metadata)
        
        # Plot 5: Timeline of CSAW competitions
        self._plot_competition_timeline(task_metadata)
    
    def _plot_yearly_distribution(self, task_metadata: Dict[str, Dict[str, Any]]) -> None:
        """Create bar chart of challenges per year."""
        years = [metadata.get('year', 'Unknown') for metadata in task_metadata.values()]
        year_counts = Counter(years)
        
        plt.figure(figsize=(10, 6))
        years_sorted, counts = zip(*sorted(year_counts.items()))
        colors = plt.cm.viridis(np.linspace(0, 1, len(years_sorted)))
        
        bars = plt.bar(years_sorted, counts, color=colors)
        plt.xlabel('Competition Year')
        plt.ylabel('Number of Challenges')
        plt.title('NYUCTF Challenge Distribution by Year')
        plt.xticks(rotation=45)
        
        # Add value labels on bars
        for bar, count in zip(bars, counts):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                    str(count), ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'yearly_distribution.png')
        plt.close()
        logger.info("Saved yearly distribution plot")
    
    def _plot_category_by_event(self, task_metadata: Dict[str, Dict[str, Any]]) -> None:
        """Create stacked bar chart of categories by event type."""
        event_category_data = []
        
        for metadata in task_metadata.values():
            event_category_data.append({
                'event': metadata.get('event', 'Unknown'),
                'category': metadata.get('category', 'unknown')
            })
        
        if event_category_data:
            df = pd.DataFrame(event_category_data)
            crosstab = pd.crosstab(df['event'], df['category'])
            
            plt.figure(figsize=(12, 6))
            crosstab.plot(kind='bar', stacked=True, colormap='Set3', ax=plt.gca())
            plt.xlabel('Event Type')
            plt.ylabel('Number of Challenges')
            plt.title('Challenge Categories by Event Type')
            plt.legend(title='Category', bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(self.output_dir / 'category_by_event.png')
            plt.close()
            logger.info("Saved category by event plot")
    
    def _plot_challenge_types(self, task_metadata: Dict[str, Dict[str, Any]]) -> None:
        """Create pie chart of challenge types and Docker support."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Challenge types
        challenge_types = [metadata.get('challenge_type', 'static') for metadata in task_metadata.values()]
        type_counts = Counter(challenge_types)
        
        ax1.pie(type_counts.values(), labels=type_counts.keys(), autopct='%1.1f%%', startangle=90)
        ax1.set_title('Challenge Type Distribution')
        
        # Docker support
        docker_support = [metadata.get('has_docker_compose', False) for metadata in task_metadata.values()]
        docker_counts = Counter(['Docker Support' if d else 'No Docker' for d in docker_support])
        
        colors = ['lightgreen', 'lightcoral']
        ax2.pie(docker_counts.values(), labels=docker_counts.keys(), autopct='%1.1f%%', 
               startangle=90, colors=colors)
        ax2.set_title('Docker Compose Support')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'challenge_types.png')
        plt.close()
        logger.info("Saved challenge types plot")
    
    def _plot_points_distribution(self, task_metadata: Dict[str, Dict[str, Any]]) -> None:
        """Create histogram of points distribution."""
        points = [metadata.get('points', 0) for metadata in task_metadata.values()]
        
        plt.figure(figsize=(10, 6))
        plt.hist(points, bins=20, edgecolor='black', alpha=0.7)
        plt.xlabel('Points')
        plt.ylabel('Number of Challenges')
        plt.title('Points Distribution Across NYUCTF Challenges')
        plt.grid(True, alpha=0.3)
        
        # Add statistics
        mean_points = np.mean(points)
        plt.axvline(mean_points, color='red', linestyle='--', label=f'Mean: {mean_points:.1f}')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'points_distribution.png')
        plt.close()
        logger.info("Saved points distribution plot")
    
    def _plot_competition_timeline(self, task_metadata: Dict[str, Dict[str, Any]]) -> None:
        """Create timeline showing challenge count evolution."""
        timeline_data = []
        
        for metadata in task_metadata.values():
            year = metadata.get('year', 'Unknown')
            event = metadata.get('event', 'Unknown')
            
            if year != 'Unknown':
                timeline_data.append({
                    'year': int(year),
                    'event': event
                })
        
        if timeline_data:
            df = pd.DataFrame(timeline_data)
            timeline_summary = df.groupby(['year', 'event']).size().unstack(fill_value=0)
            
            plt.figure(figsize=(12, 6))
            timeline_summary.plot(kind='bar', stacked=True, colormap='Set2')
            plt.xlabel('Year')
            plt.ylabel('Number of Challenges')
            plt.title('CSAW CTF Challenge Timeline')
            plt.legend(title='Event Type')
            plt.xticks(rotation=45)
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(self.output_dir / 'competition_timeline.png')
            plt.close()
            logger.info("Saved competition timeline plot")