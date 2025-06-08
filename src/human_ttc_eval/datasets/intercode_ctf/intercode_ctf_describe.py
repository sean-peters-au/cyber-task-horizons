"""
InterCode-CTF dataset describer.

Generates summary statistics and visualizations specific to the InterCode-CTF dataset,
building on the standard analyses provided by the base Describe class.
"""

import json
import logging
from typing import Dict, Any
from collections import Counter
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from human_ttc_eval.core.describe import Describe
from human_ttc_eval.core.registry import register_describer

logger = logging.getLogger(__name__)


@register_describer("intercode-ctf")
class InterCodeCTFDescribe(Describe):
    """
    InterCode-CTF specific implementation of the Describe class.
    
    Adds custom analyses for CTF challenge characteristics, category distributions,
    tag analysis, and source URL patterns beyond the standard METR analyses.
    """
    
    @property
    def dataset_name(self) -> str:
        """Returns the dataset identifier."""
        return "intercode-ctf"
    
    def generate_custom_analysis(self) -> None:
        """
        Generate InterCode-CTF specific analyses.
        
        This includes:
        - CTF category distribution analysis
        - Tag frequency analysis
        - Source URL/platform analysis
        - Setup command complexity analysis
        - Challenge difficulty estimation based on timing
        """
        if self.df is None or self.df.empty:
            logger.warning("No data loaded for custom InterCode-CTF analysis")
            return
        
        logger.info("Generating custom InterCode-CTF analyses...")
        
        # Load task metadata from tasks.jsonl for detailed analysis
        task_metadata = self._load_task_metadata()
        
        if task_metadata:
            self._generate_category_analysis()
            self._generate_tag_analysis(task_metadata)
            self._generate_source_analysis(task_metadata)
            self._generate_setup_complexity_analysis(task_metadata)
            self._generate_difficulty_distribution(task_metadata)
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
    
    def _generate_category_analysis(self) -> None:
        """Generate detailed analysis of CTF categories."""
        if 'task_family' not in self.df.columns:
            logger.warning("No task_family column found for category analysis")
            return
        
        # Extract categories from task_family (format: intercode-ctf_{category})
        self.df['category'] = self.df['task_family'].str.replace('intercode-ctf_', '')
        
        # Category distribution
        category_counts = self.df['category'].value_counts()
        
        # Category statistics
        category_stats = []
        for category in category_counts.index:
            cat_df = self.df[self.df['category'] == category]
            stats = {
                'Category': category,
                'Count': len(cat_df),
                'Percentage': round((len(cat_df) / len(self.df)) * 100, 2),
                'Mean_Time_Minutes': round(cat_df['human_minutes'].mean(), 2) if 'human_minutes' in cat_df.columns else None,
                'Median_Time_Minutes': round(cat_df['human_minutes'].median(), 2) if 'human_minutes' in cat_df.columns else None
            }
            category_stats.append(stats)
        
        category_df = pd.DataFrame(category_stats)
        category_df.to_csv(self.output_dir / 'category_analysis.csv', index=False)
        logger.info("Saved CTF category analysis")
    
    def _generate_tag_analysis(self, task_metadata: Dict[str, Dict[str, Any]]) -> None:
        """Generate analysis of challenge tags."""
        all_tags = []
        tag_combinations = []
        
        for task_id, metadata in task_metadata.items():
            tags = metadata.get('tags', [])
            if isinstance(tags, list) and tags:
                all_tags.extend(tags)
                tag_combinations.append(tuple(sorted(tags)))
        
        if not all_tags:
            logger.warning("No tags found in task metadata")
            return
        
        # Tag frequency analysis
        tag_counts = Counter(all_tags)
        
        # Create DataFrame for top tags
        top_n = 30
        tag_data = []
        total_tasks = len(task_metadata)
        
        for tag, count in tag_counts.most_common(top_n):
            percentage = (count / total_tasks) * 100
            tag_data.append({
                'Tag': tag,
                'Count': count,
                'Percentage_of_Tasks': round(percentage, 2)
            })
        
        tag_df = pd.DataFrame(tag_data)
        tag_df.to_csv(self.output_dir / 'tag_frequency_analysis.csv', index=False)
        
        # Tag combination analysis
        combo_counts = Counter(tag_combinations)
        combo_data = []
        for combo, count in combo_counts.most_common(20):
            combo_data.append({
                'Tag_Combination': ' + '.join(combo),
                'Count': count
            })
        
        if combo_data:
            combo_df = pd.DataFrame(combo_data)
            combo_df.to_csv(self.output_dir / 'tag_combinations.csv', index=False)
        
        logger.info("Saved tag analysis")
    
    def _generate_source_analysis(self, task_metadata: Dict[str, Dict[str, Any]]) -> None:
        """Generate analysis of challenge sources/platforms."""
        sources = []
        
        for metadata in task_metadata.values():
            source_url = metadata.get('source_url', '')
            if source_url:
                # Extract domain/platform from URL
                if 'github.com' in source_url:
                    platform = 'GitHub'
                elif 'ctftime.org' in source_url:
                    platform = 'CTFtime'
                elif 'picoctf' in source_url.lower():
                    platform = 'picoCTF'
                elif 'hackthebox' in source_url.lower():
                    platform = 'HackTheBox'
                else:
                    platform = 'Other'
                sources.append(platform)
        
        if sources:
            source_counts = Counter(sources)
            source_data = []
            for platform, count in source_counts.most_common():
                source_data.append({
                    'Platform': platform,
                    'Count': count,
                    'Percentage': round((count / len(sources)) * 100, 2)
                })
            
            source_df = pd.DataFrame(source_data)
            source_df.to_csv(self.output_dir / 'source_platform_analysis.csv', index=False)
            logger.info("Saved source platform analysis")
    
    def _generate_setup_complexity_analysis(self, task_metadata: Dict[str, Dict[str, Any]]) -> None:
        """Analyze complexity of setup commands."""
        setup_stats = []
        
        for task_id, metadata in task_metadata.items():
            setup_commands = metadata.get('setup_commands', [])
            if isinstance(setup_commands, list):
                num_commands = len(setup_commands)
                total_length = sum(len(cmd) for cmd in setup_commands if isinstance(cmd, str))
                
                setup_stats.append({
                    'task_id': task_id,
                    'num_setup_commands': num_commands,
                    'total_command_length': total_length,
                    'has_docker': any('docker' in str(cmd).lower() for cmd in setup_commands),
                    'has_network': any(keyword in str(cmd).lower() 
                                     for cmd in setup_commands 
                                     for keyword in ['nc', 'netcat', 'curl', 'wget', 'ssh'])
                })
        
        if setup_stats:
            setup_df = pd.DataFrame(setup_stats)
            
            # Summary statistics
            setup_summary = {
                'Average_Setup_Commands': round(setup_df['num_setup_commands'].mean(), 2),
                'Max_Setup_Commands': int(setup_df['num_setup_commands'].max()),
                'Tasks_With_Docker': int(setup_df['has_docker'].sum()),
                'Tasks_With_Network': int(setup_df['has_network'].sum()),
                'Percentage_Docker': round((setup_df['has_docker'].sum() / len(setup_df)) * 100, 2),
                'Percentage_Network': round((setup_df['has_network'].sum() / len(setup_df)) * 100, 2)
            }
            
            with open(self.output_dir / 'setup_complexity_summary.json', 'w') as f:
                json.dump(setup_summary, f, indent=2)
            
            logger.info("Saved setup complexity analysis")
    
    def _generate_difficulty_distribution(self, task_metadata: Dict[str, Dict[str, Any]]) -> None:
        """Estimate difficulty distribution based on timing."""
        if 'human_minutes' not in self.df.columns:
            logger.warning("No timing data available for difficulty analysis")
            return
        
        # Define difficulty buckets based on time
        def categorize_difficulty(minutes):
            if minutes < 2:
                return 'Very Easy'
            elif minutes < 5:
                return 'Easy'
            elif minutes < 10:
                return 'Medium'
            elif minutes < 20:
                return 'Hard'
            else:
                return 'Very Hard'
        
        self.df['estimated_difficulty'] = self.df['human_minutes'].apply(categorize_difficulty)
        
        difficulty_counts = self.df['estimated_difficulty'].value_counts()
        difficulty_order = ['Very Easy', 'Easy', 'Medium', 'Hard', 'Very Hard']
        
        difficulty_data = []
        for difficulty in difficulty_order:
            if difficulty in difficulty_counts:
                count = difficulty_counts[difficulty]
                percentage = (count / len(self.df)) * 100
                difficulty_data.append({
                    'Difficulty': difficulty,
                    'Count': count,
                    'Percentage': round(percentage, 2)
                })
        
        difficulty_df = pd.DataFrame(difficulty_data)
        difficulty_df.to_csv(self.output_dir / 'difficulty_distribution.csv', index=False)
        logger.info("Saved difficulty distribution analysis")
    
    def _generate_custom_plots(self, task_metadata: Dict[str, Dict[str, Any]]) -> None:
        """Generate InterCode-CTF specific visualizations."""
        # Plot 1: Category distribution pie chart
        self._plot_category_distribution()
        
        # Plot 2: Tag frequency bar chart
        self._plot_tag_frequency(task_metadata)
        
        # Plot 3: Time distribution by category
        self._plot_time_by_category()
        
        # Plot 4: Difficulty distribution
        self._plot_difficulty_distribution()
    
    def _plot_category_distribution(self) -> None:
        """Create pie chart of CTF categories."""
        if 'category' not in self.df.columns:
            return
        
        category_counts = self.df['category'].value_counts()
        
        plt.figure(figsize=(10, 8))
        colors = plt.cm.Set3(range(len(category_counts)))
        plt.pie(category_counts.values, labels=category_counts.index, autopct='%1.1f%%', 
                colors=colors, startangle=90)
        plt.title('InterCode-CTF Challenge Category Distribution')
        plt.axis('equal')
        plt.tight_layout()
        plt.savefig(self.output_dir / 'category_distribution_pie.png')
        plt.close()
        logger.info("Saved category distribution pie chart")
    
    def _plot_tag_frequency(self, task_metadata: Dict[str, Dict[str, Any]]) -> None:
        """Create bar chart of most common tags."""
        all_tags = []
        for metadata in task_metadata.values():
            tags = metadata.get('tags', [])
            if isinstance(tags, list):
                all_tags.extend(tags)
        
        if not all_tags:
            return
        
        tag_counts = Counter(all_tags)
        top_tags = tag_counts.most_common(20)
        
        if not top_tags:
            return
        
        tags, counts = zip(*top_tags)
        
        plt.figure(figsize=(12, 6))
        plt.bar(tags, counts, color='skyblue')
        plt.xticks(rotation=45, ha='right')
        plt.xlabel('Tag')
        plt.ylabel('Number of Challenges')
        plt.title('Top 20 Most Common CTF Challenge Tags')
        plt.tight_layout()
        plt.savefig(self.output_dir / 'tag_frequency_chart.png')
        plt.close()
        logger.info("Saved tag frequency chart")
    
    def _plot_time_by_category(self) -> None:
        """Create box plot of completion times by category."""
        if 'category' not in self.df.columns or 'human_minutes' not in self.df.columns:
            return
        
        # Filter out categories with too few samples
        category_counts = self.df['category'].value_counts()
        valid_categories = category_counts[category_counts >= 3].index.tolist()
        
        plot_df = self.df[self.df['category'].isin(valid_categories)]
        
        if plot_df.empty:
            return
        
        plt.figure(figsize=(10, 6))
        sns.boxplot(data=plot_df, x='category', y='human_minutes')
        plt.xlabel('CTF Category')
        plt.ylabel('Completion Time (minutes)')
        plt.title('Challenge Completion Time Distribution by Category')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(self.output_dir / 'time_by_category_boxplot.png')
        plt.close()
        logger.info("Saved time by category box plot")
    
    def _plot_difficulty_distribution(self) -> None:
        """Create bar chart of estimated difficulty distribution."""
        if 'estimated_difficulty' not in self.df.columns:
            return
        
        difficulty_order = ['Very Easy', 'Easy', 'Medium', 'Hard', 'Very Hard']
        difficulty_counts = self.df['estimated_difficulty'].value_counts()
        
        # Ensure all difficulties are represented
        plot_data = []
        for difficulty in difficulty_order:
            count = difficulty_counts.get(difficulty, 0)
            plot_data.append({'Difficulty': difficulty, 'Count': count})
        
        plot_df = pd.DataFrame(plot_data)
        
        plt.figure(figsize=(10, 6))
        colors = ['#2ecc71', '#3498db', '#f39c12', '#e74c3c', '#9b59b6']
        plt.bar(plot_df['Difficulty'], plot_df['Count'], color=colors)
        plt.xlabel('Estimated Difficulty')
        plt.ylabel('Number of Challenges')
        plt.title('Challenge Difficulty Distribution (Based on Completion Time)')
        plt.tight_layout()
        plt.savefig(self.output_dir / 'difficulty_distribution_chart.png')
        plt.close()
        logger.info("Saved difficulty distribution chart") 