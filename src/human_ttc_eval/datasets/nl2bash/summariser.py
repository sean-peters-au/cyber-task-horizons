"""Summarise NL2Bash dataset results, generating stats and plots."""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import logging
from typing import Optional
from collections import Counter

from ...core.base_summariser import BaseSummariser
from ...core.registry import register_summariser

logger = logging.getLogger(__name__)

@register_summariser("nl2bash")
class NL2BashSummariser(BaseSummariser):
    """Summarises NL2Bash dataset results, generating stats and plots."""

    def __init__(self, jsonl_file_path: Path, output_dir: Path):
        super().__init__(jsonl_file_path, output_dir)
        logger.info(f"NL2BashSummariser initialized. Input JSONL: {self.jsonl_file_path}, Output Dir: {self.output_dir}")

    def _generate_nl2bash_specific_stats(self, df: pd.DataFrame) -> None:
        """Generate NL2Bash-specific statistics."""
        output_path = self.output_dir / "nl2bash_specific_stats.csv"
        if df.empty:
            logger.warning("DataFrame is empty. Skipping NL2Bash-specific stats generation.")
            return

        total_tasks = len(df)
        
        stats = {}

        # Complexity analysis
        if 'complexity_score' in df.columns and not df['complexity_score'].isnull().all():
            stats.update({
                "Complexity Score - Min": df['complexity_score'].min(),
                "Complexity Score - Max": df['complexity_score'].max(),
                "Complexity Score - Mean": df['complexity_score'].mean(),
                "Complexity Score - Median": df['complexity_score'].median(),
                "Complexity Score - StdDev": df['complexity_score'].std(),
            })

        # Feature analysis
        if 'has_pipes' in df.columns:
            stats["Commands with Pipes"] = df['has_pipes'].sum()
            stats["Pipe Percentage"] = (df['has_pipes'].sum() / total_tasks * 100)
        
        if 'has_redirects' in df.columns:
            stats["Commands with Redirects"] = df['has_redirects'].sum()
            stats["Redirect Percentage"] = (df['has_redirects'].sum() / total_tasks * 100)
        
        if 'has_subcommands' in df.columns:
            stats["Commands with Subcommands"] = df['has_subcommands'].sum()
            stats["Subcommand Percentage"] = (df['has_subcommands'].sum() / total_tasks * 100)

        # Word count analysis
        if 'word_count' in df.columns and not df['word_count'].isnull().all():
            stats.update({
                "Word Count - Min": df['word_count'].min(),
                "Word Count - Max": df['word_count'].max(),
                "Word Count - Mean": df['word_count'].mean(),
                "Word Count - Median": df['word_count'].median(),
                "Word Count - StdDev": df['word_count'].std(),
            })

        # Timing source analysis
        if 'timing_source' in df.columns:
            timing_counts = df['timing_source'].value_counts().to_dict()
            for source, count in timing_counts.items():
                stats[f"Timing Source - {source.title()}"] = count
                stats[f"Timing Source - {source.title()} (%)"] = (count / total_tasks * 100)

        if stats:  # Only save if we have stats
            stats_df = pd.DataFrame.from_dict(stats, orient='index', columns=['Value'])
            stats_df.to_csv(output_path)
            logger.info(f"NL2Bash-specific statistics saved to {output_path}")

    def _generate_utility_analysis(self, df: pd.DataFrame) -> None:
        """Generate analysis of utility usage."""
        output_path = self.output_dir / "utility_usage_stats.csv"
        
        if df.empty or 'utilities_used' not in df.columns:
            logger.warning("DataFrame is empty or missing 'utilities_used'. Skipping utility analysis.")
            return

        # Extract all utilities
        all_utilities = []
        for utilities_list in df['utilities_used'].dropna():
            if isinstance(utilities_list, list):
                all_utilities.extend(utilities_list)
            elif isinstance(utilities_list, str):
                # Handle case where utilities might be stored as string
                try:
                    utilities = eval(utilities_list)  # Be careful with eval
                    if isinstance(utilities, list):
                        all_utilities.extend(utilities)
                except:
                    logger.warning(f"Could not parse utilities: {utilities_list}")

        if not all_utilities:
            logger.warning("No utilities found in dataset.")
            return

        utility_counts = Counter(all_utilities)
        
        utility_stats = []
        total_tasks = len(df)
        for utility, count in utility_counts.most_common(20):  # Top 20 utilities
            percentage = (count / total_tasks) * 100
            utility_stats.append({
                'Utility': utility,
                'Count': count,
                'Percentage_of_Tasks': percentage
            })
        
        utility_df = pd.DataFrame(utility_stats)
        utility_df.to_csv(output_path, index=False)
        logger.info(f"Utility usage statistics saved to {output_path}")

    def summarise(self) -> None:
        """Generate comprehensive summary statistics for NL2Bash dataset."""
        # Call parent to generate standard METR format analysis
        super().summarise()
        
        if self.df is None or self.df.empty:
            logger.error("No data loaded by BaseSummariser. Exiting NL2BashSummariser.summarise.")
            return
        
        # Add NL2Bash-specific analysis
        self._generate_nl2bash_specific_stats(self.df)
        self._generate_utility_analysis(self.df)

    def save_plots(self) -> None:
        """Generate NL2Bash-specific plots in addition to standard ones."""
        # Call parent to generate standard METR format plots
        super().save_plots()
        
        if self.df is None or self.df.empty:
            logger.error("No data loaded by BaseSummariser. Exiting NL2BashSummariser.save_plots.")
            return

        # NL2Bash-specific plots
        
        # Histogram of complexity scores
        if 'complexity_score' in self.df.columns:
            self._plot_histogram(self.df, 'complexity_score', 'Distribution of Complexity Scores', 
                               self.output_dir / "hist_complexity_scores.png")

        # Histogram of word counts
        if 'word_count' in self.df.columns:
            self._plot_histogram(self.df, 'word_count', 'Distribution of Command Word Counts', 
                               self.output_dir / "hist_word_counts.png")

        # Scatter plot: Complexity vs Completion Time
        if 'complexity_score' in self.df.columns and 'human_minutes' in self.df.columns:
            hue_col = 'timing_source' if 'timing_source' in self.df.columns else None
            self._plot_scatter(self.df, 'complexity_score', 'human_minutes', hue_col, 
                             'Complexity Score vs Completion Time', 
                             self.output_dir / "scatter_complexity_vs_time.png")

        # Scatter plot: Word Count vs Completion Time
        if 'word_count' in self.df.columns and 'human_minutes' in self.df.columns:
            hue_col = 'has_pipes' if 'has_pipes' in self.df.columns else None
            self._plot_scatter(self.df, 'word_count', 'human_minutes', hue_col, 
                             'Word Count vs Completion Time', 
                             self.output_dir / "scatter_wordcount_vs_time.png")

        # Feature comparison plots
        if all(col in self.df.columns for col in ['has_pipes', 'has_redirects', 'has_subcommands', 'human_minutes']):
            features = ['has_pipes', 'has_redirects', 'has_subcommands']
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            
            for i, feature in enumerate(features):
                feature_data = []
                labels = []
                for has_feature in [False, True]:
                    subset = self.df[self.df[feature] == has_feature]['human_minutes'].dropna()
                    if not subset.empty:
                        feature_data.append(subset)
                        labels.append(f"With {feature.replace('has_', '')}" if has_feature else f"Without {feature.replace('has_', '')}")
                
                if len(feature_data) == 2:
                    axes[i].boxplot(feature_data, labels=labels)
                    axes[i].set_title(f'Completion Time by {feature.replace("has_", "").title()}')
                    axes[i].set_ylabel('Completion Time (minutes)')
                    
            plt.tight_layout()
            plt.savefig(self.output_dir / "box_completion_time_by_features.png")
            plt.close()
            logger.info(f"Feature comparison plot saved to {self.output_dir / 'box_completion_time_by_features.png'}")

        # Timing source comparison (if available)
        if 'timing_source' in self.df.columns and 'human_minutes' in self.df.columns:
            timing_sources = self.df['timing_source'].unique()
            if len(timing_sources) > 1:
                plt.figure(figsize=(10, 6))
                sns.boxplot(x='timing_source', y='human_minutes', data=self.df)
                plt.title('Completion Time Distribution by Timing Source')
                plt.xlabel('Timing Source')
                plt.ylabel('Completion Time (minutes)')
                plt.tight_layout()
                plt.savefig(self.output_dir / "box_completion_time_by_timing_source.png")
                plt.close()
                logger.info(f"Timing source comparison plot saved to {self.output_dir / 'box_completion_time_by_timing_source.png'}")

        logger.info(f"NL2Bash dataset plots saved to {self.output_dir}") 