"""
CyBench dataset describer.

Generates summary statistics and visualizations specific to the CyBench dataset,
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


@register_describer("cybench")
class CybenchDescribe(Describe):
    """
    CyBench specific implementation of the Describe class.
    
    Adds custom analyses for professional CTF challenges including competition
    analysis, category distributions, difficulty scaling, and timing patterns.
        """

    @property
    def dataset_name(self) -> str:
        """Returns the dataset identifier."""
        return "cybench"

    def generate_custom_analysis(self) -> None:
        """
        Generate CyBench specific analyses.
        
        This includes:
        - Competition source analysis
        - CTF category distribution and performance
        - Difficulty scaling analysis
        - Time horizon analysis for METR-style evaluation
        - Flag complexity analysis
        - Variant availability analysis
        """
        if self.df is None or self.df.empty:
            logger.warning("No data loaded for custom CyBench analysis")
            return

        logger.info("Generating custom CyBench analyses...")
        
        # Load task metadata from tasks.jsonl for detailed analysis
        task_metadata = self._load_task_metadata()
        
        if task_metadata:
            self._generate_competition_analysis(task_metadata)
            self._generate_category_performance_analysis()
            self._generate_difficulty_scaling_analysis()
            self._generate_time_horizon_analysis()
            self._generate_flag_complexity_analysis(task_metadata)
            self._generate_variant_analysis(task_metadata)
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
    
    def _generate_competition_analysis(self, task_metadata: Dict[str, Dict[str, Any]]) -> None:
        """Generate analysis of CTF competitions represented in the dataset."""
        competitions = []
        competition_categories = defaultdict(list)
        
        for task_id, metadata in task_metadata.items():
            competition = metadata.get('competition', 'Unknown')
            category = metadata.get('category', 'unknown')
            competitions.append(competition)
            competition_categories[competition].append(category)
        
        # Competition distribution
        comp_counts = Counter(competitions)
        comp_data = []
        
        for comp, count in comp_counts.most_common():
            # Get unique categories for this competition
            categories = Counter(competition_categories[comp])
            comp_data.append({
                'Competition': comp,
                'Total_Challenges': count,
                'Percentage': round((count / len(competitions)) * 100, 2),
                'Categories': ', '.join(f"{cat}({cnt})" for cat, cnt in categories.most_common()),
                'Num_Categories': len(categories)
            })
        
        comp_df = pd.DataFrame(comp_data)
        comp_df.to_csv(self.output_dir / 'competition_analysis.csv', index=False)
        logger.info("Saved competition analysis")
    
    def _generate_category_performance_analysis(self) -> None:
        """Generate detailed analysis of performance by CTF category."""
        if 'task_family' not in self.df.columns:
            logger.warning("No task_family column found for category analysis")
            return
        
        # Extract categories from task_family (format: cybench_{category})
        self.df['category'] = self.df['task_family'].str.replace('cybench_', '')
        
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
        category_df = category_df.sort_values('Mean_Minutes', ascending=False)
        category_df.to_csv(self.output_dir / 'category_performance_analysis.csv', index=False)
        logger.info("Saved category performance analysis")
    
    def _generate_difficulty_scaling_analysis(self) -> None:
        """Analyze how challenge difficulty scales with time."""
        if 'human_minutes' not in self.df.columns:
            logger.warning("No timing data available for difficulty analysis")
            return

        # Define difficulty tiers based on CyBench's stated range (30 min - 24 hours)
        def categorize_difficulty(minutes):
            hours = minutes / 60
            if hours < 0.5:  # < 30 minutes
                return '1. Trivial (<30m)'
            elif hours < 1:  # 30-60 minutes
                return '2. Easy (30m-1h)'
            elif hours < 2:  # 1-2 hours
                return '3. Medium (1-2h)'
            elif hours < 4:  # 2-4 hours
                return '4. Hard (2-4h)'
            elif hours < 8:  # 4-8 hours
                return '5. Very Hard (4-8h)'
            else:  # 8+ hours
                return '6. Expert (8h+)'
        
        self.df['difficulty_tier'] = self.df['human_minutes'].apply(categorize_difficulty)
        
        # Difficulty distribution with category breakdown
        difficulty_stats = []
        
        for tier in sorted(self.df['difficulty_tier'].unique()):
            tier_df = self.df[self.df['difficulty_tier'] == tier]
            
            # Category breakdown for this tier
            category_counts = tier_df['category'].value_counts()
            
            stats = {
                'Difficulty_Tier': tier,
                'Count': len(tier_df),
                'Percentage': round((len(tier_df) / len(self.df)) * 100, 2),
                'Mean_Minutes': round(tier_df['human_minutes'].mean(), 2),
                'Categories': ', '.join(f"{cat}({cnt})" for cat, cnt in category_counts.items())
            }
            difficulty_stats.append(stats)
        
        difficulty_df = pd.DataFrame(difficulty_stats)
        difficulty_df.to_csv(self.output_dir / 'difficulty_scaling_analysis.csv', index=False)
        logger.info("Saved difficulty scaling analysis")
    
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
                'Tasks_Solvable': tasks_within_horizon,
                'Percentage_Solvable': round(percentage, 2),
                'Categories': ', '.join(f"{cat}({cnt})" for cat, cnt in category_counts.items())
            })
        
        horizon_df = pd.DataFrame(horizon_stats)
        horizon_df.to_csv(self.output_dir / 'time_horizon_analysis.csv', index=False)
        logger.info("Saved time horizon analysis")
    
    def _generate_flag_complexity_analysis(self, task_metadata: Dict[str, Dict[str, Any]]) -> None:
        """Analyze flag formats and complexity."""
        flag_stats = []
        flag_prefixes = []
        
        for task_id, metadata in task_metadata.items():
            flag = metadata.get('flag', '')
            if flag:
                # Extract flag prefix (e.g., "picoCTF", "hkcert22", etc.)
                if '{' in flag:
                    prefix = flag.split('{')[0]
                    flag_prefixes.append(prefix)
                
                flag_stats.append({
                    'task_id': task_id,
                    'flag_length': len(flag),
                    'has_special_chars': any(c in flag for c in '!@#$%^&*()_+-=[]{}|;:,.<>?'),
                    'has_numbers': any(c.isdigit() for c in flag),
                    'has_uppercase': any(c.isupper() for c in flag),
                    'has_lowercase': any(c.islower() for c in flag)
                })
        
        if flag_stats:
            flag_df = pd.DataFrame(flag_stats)
            
            # Summary statistics
            flag_summary = {
                'Average_Flag_Length': round(flag_df['flag_length'].mean(), 2),
                'Min_Flag_Length': int(flag_df['flag_length'].min()),
                'Max_Flag_Length': int(flag_df['flag_length'].max()),
                'Flags_With_Special_Chars': int(flag_df['has_special_chars'].sum()),
                'Unique_Flag_Prefixes': len(set(flag_prefixes)),
                'Most_Common_Prefixes': ', '.join(f"{prefix}({count})" 
                                                 for prefix, count in Counter(flag_prefixes).most_common(5))
            }
            
            with open(self.output_dir / 'flag_complexity_summary.json', 'w') as f:
                json.dump(flag_summary, f, indent=2)
            
            logger.info("Saved flag complexity analysis")
    
    def _generate_variant_analysis(self, task_metadata: Dict[str, Dict[str, Any]]) -> None:
        """Analyze available challenge variants."""
        variant_stats = []
        all_variants = []
        
        for task_id, metadata in task_metadata.items():
            available_variants = metadata.get('available_variants', [])
            selected_variant = metadata.get('selected_variant', '')
            
            all_variants.extend(available_variants)
            
            variant_stats.append({
                'task_id': task_id,
                'num_variants': len(available_variants),
                'has_easy': 'easy' in available_variants,
                'has_hard': 'hard' in available_variants,
                'has_solution': 'solution' in available_variants,
                'has_korean': any('korean' in v for v in available_variants),
                'selected_variant': selected_variant
            })
        
        if variant_stats:
            variant_df = pd.DataFrame(variant_stats)
            
            # Summary
            variant_summary = {
                'Average_Variants_Per_Challenge': round(variant_df['num_variants'].mean(), 2),
                'Challenges_With_Easy': int(variant_df['has_easy'].sum()),
                'Challenges_With_Hard': int(variant_df['has_hard'].sum()),
                'Challenges_With_Solution': int(variant_df['has_solution'].sum()),
                'Challenges_With_Korean': int(variant_df['has_korean'].sum()),
                'Variant_Distribution': dict(Counter(all_variants))
            }
            
            with open(self.output_dir / 'variant_analysis_summary.json', 'w') as f:
                json.dump(variant_summary, f, indent=2)
            
            logger.info("Saved variant analysis")
    
    def _generate_custom_plots(self, task_metadata: Dict[str, Dict[str, Any]]) -> None:
        """Generate CyBench specific visualizations."""
        # Plot 1: Competition distribution
        self._plot_competition_distribution(task_metadata)
        
        # Plot 2: Time distribution by category (violin plot)
        self._plot_time_by_category_violin()
        
        # Plot 3: Difficulty tier distribution
        self._plot_difficulty_tiers()
        
        # Plot 4: Time horizon curve (METR-style)
        self._plot_time_horizon_curve()
        
        # Plot 5: Category complexity heatmap
        self._plot_category_complexity_heatmap()
    
    def _plot_competition_distribution(self, task_metadata: Dict[str, Dict[str, Any]]) -> None:
        """Create bar chart of challenges per competition."""
        competitions = [metadata.get('competition', 'Unknown') 
                       for metadata in task_metadata.values()]
        
        comp_counts = Counter(competitions)
        
        plt.figure(figsize=(10, 6))
        comps, counts = zip(*comp_counts.most_common())
        colors = plt.cm.Set3(range(len(comps)))
        plt.bar(comps, counts, color=colors)
        plt.xlabel('CTF Competition')
        plt.ylabel('Number of Challenges')
        plt.title('CyBench Challenge Distribution by Competition')
        plt.xticks(rotation=45, ha='right')
        
        # Add value labels on bars
        for i, (comp, count) in enumerate(zip(comps, counts)):
            plt.text(i, count + 0.5, str(count), ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'competition_distribution.png')
        plt.close()
        logger.info("Saved competition distribution plot")
    
    def _plot_time_by_category_violin(self) -> None:
        """Create violin plot of completion times by category."""
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
        plt.xlabel('CTF Category')
        plt.ylabel('Completion Time (minutes)')
        plt.title('Challenge Completion Time Distribution by Category')
        plt.xticks(rotation=45)
        
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
    
    def _plot_difficulty_tiers(self) -> None:
        """Create stacked bar chart of difficulty tiers by category."""
        if 'difficulty_tier' not in self.df.columns or 'category' not in self.df.columns:
            return
        
        # Create crosstab
        crosstab = pd.crosstab(self.df['category'], self.df['difficulty_tier'])
        
        # Ensure all tiers are represented
        all_tiers = sorted(self.df['difficulty_tier'].unique())
        for tier in all_tiers:
            if tier not in crosstab.columns:
                crosstab[tier] = 0
        
        # Reorder columns by tier
        crosstab = crosstab[sorted(crosstab.columns)]
        
        # Create stacked bar chart
        plt.figure(figsize=(12, 8))
        crosstab.plot(kind='bar', stacked=True, colormap='RdYlBu_r', ax=plt.gca())
        plt.xlabel('CTF Category')
        plt.ylabel('Number of Challenges')
        plt.title('Challenge Difficulty Distribution by Category')
        plt.legend(title='Difficulty Tier', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(self.output_dir / 'difficulty_tiers_by_category.png')
        plt.close()
        logger.info("Saved difficulty tiers plot")
    
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
        plt.ylabel('Percentage of Challenges Solvable (%)')
        plt.title('CyBench Human Performance Horizon Curve')
        plt.grid(True, alpha=0.3)
        
        # Add reference lines
        plt.axvline(x=0.5, color='gray', linestyle='--', alpha=0.5, label='30 min')
        plt.axvline(x=1, color='gray', linestyle='--', alpha=0.5, label='1 hour')
        plt.axvline(x=4, color='gray', linestyle='--', alpha=0.5, label='4 hours')
        plt.axvline(x=24, color='gray', linestyle='--', alpha=0.5, label='24 hours')
        
        # Add percentage markers
        for pct in [25, 50, 75, 90]:
            time_at_pct = np.interp(pct, percentages, time_points) / 60
            plt.axhline(y=pct, color='red', linestyle=':', alpha=0.3)
            plt.text(0.02, pct + 1, f'{pct}% @ {time_at_pct:.1f}h', fontsize=9)
        
        plt.xlim(0.01, max_time / 60 * 1.1)
        plt.ylim(0, 105)
        plt.tight_layout()
        plt.savefig(self.output_dir / 'time_horizon_curve.png')
        plt.close()
        logger.info("Saved time horizon curve")
    
    def _plot_category_complexity_heatmap(self) -> None:
        """Create heatmap showing category complexity metrics."""
        if 'category' not in self.df.columns:
            return
        
        # Calculate complexity metrics for each category
        metrics = []
        
        for category in self.df['category'].unique():
            cat_df = self.df[self.df['category'] == category]
            
            metrics.append({
                'Category': category,
                'Count': len(cat_df),
                'Mean Time (h)': cat_df['human_minutes'].mean() / 60,
                'Max Time (h)': cat_df['human_minutes'].max() / 60,
                'Time Variance': cat_df['human_minutes'].std() / cat_df['human_minutes'].mean() if len(cat_df) > 1 else 0,
                'Time Span': (cat_df['human_minutes'].max() - cat_df['human_minutes'].min()) / 60
            })
        
        metrics_df = pd.DataFrame(metrics)
        metrics_df = metrics_df.set_index('Category')
        
        # Normalize for heatmap
        normalized_df = (metrics_df - metrics_df.min()) / (metrics_df.max() - metrics_df.min())
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(normalized_df.T, annot=metrics_df.T.round(1), fmt='g', 
                   cmap='YlOrRd', cbar_kws={'label': 'Normalized Value'})
        plt.title('CTF Category Complexity Metrics')
        plt.xlabel('Category')
        plt.ylabel('Metric')
        plt.tight_layout()
        plt.savefig(self.output_dir / 'category_complexity_heatmap.png')
        plt.close()
        logger.info("Saved category complexity heatmap") 