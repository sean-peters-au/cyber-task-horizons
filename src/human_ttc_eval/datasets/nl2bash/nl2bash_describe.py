"""
NL2Bash dataset describer.

Generates summary statistics and visualizations specific to the NL2Bash dataset,
building on the standard analyses provided by the base Describe class.
"""

import json
import logging
from pathlib import Path
from typing import List, Dict, Any
from collections import Counter
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import textwrap

from human_ttc_eval.core.describe import Describe
from human_ttc_eval.core.registry import register_describer

logger = logging.getLogger(__name__)


@register_describer("nl2bash")
class NL2BashDescribe(Describe):
    """
    NL2Bash-specific implementation of the Describe class.
    
    Adds custom analyses for bash command characteristics, utility usage,
    and complexity distributions beyond the standard METR analyses.
    """
    
    @property
    def dataset_name(self) -> str:
        """Returns the dataset identifier."""
        return "nl2bash"
    
    def generate_custom_analysis(self) -> None:
        """
        Generate NL2Bash-specific analyses.
        
        This includes:
        - Utility usage statistics
        - Feature prevalence (pipes, redirects, etc.)
        - Complexity distribution details
        - Timing source analysis
        - Review table generation
        """
        if self.df is None or self.df.empty:
            logger.warning("No data loaded for custom NL2Bash analysis")
            return
        
        logger.info("Generating custom NL2Bash analyses...")
        
        # Load task metadata from tasks.jsonl for detailed analysis
        task_metadata = self._load_task_metadata()
        
        if task_metadata:
            self._generate_utility_analysis(task_metadata)
            self._generate_feature_analysis(task_metadata)
            self._generate_complexity_analysis(task_metadata)
            self._generate_timing_source_analysis(task_metadata)
            self._generate_review_table(task_metadata)
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
    
    def _generate_utility_analysis(self, task_metadata: Dict[str, Dict[str, Any]]) -> None:
        """Generate analysis of bash utility usage."""
        all_utilities = []
        
        for task_id, metadata in task_metadata.items():
            utilities = metadata.get('utilities_used', [])
            if isinstance(utilities, list):
                all_utilities.extend(utilities)
        
        if not all_utilities:
            logger.warning("No utilities found in task metadata")
            return
        
        utility_counts = Counter(all_utilities)
        
        # Create DataFrame for top utilities
        top_n = 30
        utility_data = []
        total_tasks = len(task_metadata)
        
        for utility, count in utility_counts.most_common(top_n):
            percentage = (count / total_tasks) * 100
            utility_data.append({
                'Utility': utility,
                'Count': count,
                'Percentage_of_Tasks': round(percentage, 2)
            })
        
        utility_df = pd.DataFrame(utility_data)
        utility_df.to_csv(self.output_dir / 'utility_usage_stats.csv', index=False)
        logger.info(f"Saved utility usage statistics for top {top_n} utilities")
    
    def _generate_feature_analysis(self, task_metadata: Dict[str, Dict[str, Any]]) -> None:
        """Generate analysis of bash command features."""
        feature_counts = {
            'has_pipes': 0,
            'has_redirects': 0,
            'has_subcommands': 0,
            'is_simple': 0  # No pipes, redirects, or subcommands
        }
        
        total_tasks = len(task_metadata)
        
        for task_id, metadata in task_metadata.items():
            has_pipes = metadata.get('has_pipes', False)
            has_redirects = metadata.get('has_redirects', False)
            has_subcommands = metadata.get('has_subcommands', False)
            
            if has_pipes:
                feature_counts['has_pipes'] += 1
            if has_redirects:
                feature_counts['has_redirects'] += 1
            if has_subcommands:
                feature_counts['has_subcommands'] += 1
            if not (has_pipes or has_redirects or has_subcommands):
                feature_counts['is_simple'] += 1
        
        # Create feature analysis DataFrame
        feature_data = []
        for feature, count in feature_counts.items():
            percentage = (count / total_tasks) * 100
            feature_data.append({
                'Feature': feature.replace('_', ' ').title(),
                'Count': count,
                'Percentage': round(percentage, 2)
            })
        
        feature_df = pd.DataFrame(feature_data)
        feature_df.to_csv(self.output_dir / 'feature_analysis.csv', index=False)
        logger.info("Saved bash command feature analysis")
    
    def _generate_complexity_analysis(self, task_metadata: Dict[str, Dict[str, Any]]) -> None:
        """Generate detailed complexity distribution analysis."""
        complexity_scores = []
        complexity_by_category = {}
        
        for task_id, metadata in task_metadata.items():
            score = metadata.get('complexity_score', 0.0)
            category = metadata.get('complexity_category', 'unknown')
            complexity_scores.append(score)
            
            if category not in complexity_by_category:
                complexity_by_category[category] = []
            complexity_by_category[category].append(score)
        
        if not complexity_scores:
            logger.warning("No complexity scores found")
            return
        
        # Overall complexity statistics
        complexity_stats = {
            'Mean_Complexity': round(pd.Series(complexity_scores).mean(), 2),
            'Median_Complexity': round(pd.Series(complexity_scores).median(), 2),
            'Std_Complexity': round(pd.Series(complexity_scores).std(), 2),
            'Min_Complexity': round(min(complexity_scores), 2),
            'Max_Complexity': round(max(complexity_scores), 2)
        }
        
        # Category distribution
        category_counts = Counter(metadata.get('complexity_category', 'unknown') 
                                 for metadata in task_metadata.values())
        
        # Save complexity analysis
        with open(self.output_dir / 'complexity_analysis.json', 'w') as f:
            json.dump({
                'overall_stats': complexity_stats,
                'category_distribution': dict(category_counts),
                'scores_by_category': {
                    cat: {
                        'mean': round(pd.Series(scores).mean(), 2),
                        'median': round(pd.Series(scores).median(), 2),
                        'count': len(scores)
                    }
                    for cat, scores in complexity_by_category.items()
                }
            }, f, indent=2)
        
        logger.info("Saved complexity analysis")
    
    def _generate_timing_source_analysis(self, task_metadata: Dict[str, Dict[str, Any]]) -> None:
        """Analyze timing estimation sources if multiple are present."""
        timing_sources = Counter(metadata.get('timing_source', 'unknown') 
                                for metadata in task_metadata.values())
        
        if len(timing_sources) > 1:
            # Compare timing estimates by source
            timing_by_source = {}
            
            for run in self.runs:
                if hasattr(run, 'human_source'):
                    source = 'llm' if 'llm' in run.human_source else 'heuristic'
                    if source not in timing_by_source:
                        timing_by_source[source] = []
                    timing_by_source[source].append(run.human_minutes)
            
            if timing_by_source:
                timing_comparison = {}
                for source, times in timing_by_source.items():
                    timing_comparison[source] = {
                        'count': len(times),
                        'mean_minutes': round(pd.Series(times).mean(), 2),
                        'median_minutes': round(pd.Series(times).median(), 2)
                    }
                
                with open(self.output_dir / 'timing_source_comparison.json', 'w') as f:
                    json.dump(timing_comparison, f, indent=2)
                
                logger.info("Saved timing source comparison")
    
    def _generate_review_table(self, task_metadata: Dict[str, Dict[str, Any]]) -> None:
        """Generate a PDF table of tasks for human review."""
        review_data = []
        for task_id, metadata in task_metadata.items():
            run_data = self.df[self.df['task_id'] == task_id]
            if not run_data.empty:
                human_minutes = run_data.iloc[0].get('human_minutes', 0)
                estimated_seconds = human_minutes * 60
                
                review_data.append({
                    'NL Description': metadata.get('nl_description', ''),
                    'Bash Command': metadata.get('bash_command', ''),
                    'Time (s)': f"{estimated_seconds:.1f}"
                })

        if not review_data:
            logger.warning("No data to generate review table.")
            return

        review_df = pd.DataFrame(review_data)
        
        # Sort by time estimate
        review_df['Time (s)'] = pd.to_numeric(review_df['Time (s)'])
        review_df = review_df.sort_values(by='Time (s)', ascending=True).reset_index(drop=True)

        # Wrap text for better display in PDF
        wrapped_df = review_df.copy()
        
        # Escape special characters for matplotlib to avoid parsing errors
        for col in ['NL Description', 'Bash Command']:
            wrapped_df[col] = wrapped_df[col].astype(str).str.replace('$', r'\$', regex=False)
            wrapped_df[col] = wrapped_df[col].str.replace('{', r'\{', regex=False)
            wrapped_df[col] = wrapped_df[col].str.replace('}', r'\}', regex=False)
            wrapped_df[col] = wrapped_df[col].str.replace('_', r'\_', regex=False)
            wrapped_df[col] = wrapped_df[col].str.replace('#', r'\#', regex=False)
            wrapped_df[col] = wrapped_df[col].str.replace('%', r'\%', regex=False)
            wrapped_df[col] = wrapped_df[col].str.replace('&', r'\&', regex=False)

        wrapped_df['NL Description'] = wrapped_df['NL Description'].apply(lambda x: '\n'.join(textwrap.wrap(x, width=60)))
        wrapped_df['Bash Command'] = wrapped_df['Bash Command'].apply(lambda x: '\n'.join(textwrap.wrap(x, width=50)))
        
        # Create PDF with matplotlib table
        fig, ax = plt.subplots(figsize=(16, 2 + len(wrapped_df) * 0.8)) # Dynamic height
        ax.axis('tight')
        ax.axis('off')
        
        table = ax.table(cellText=wrapped_df.values, colLabels=wrapped_df.columns, loc='center', cellLoc='left')
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 2.5) # Adjust scale for vertical spacing
        
        # Make header bold
        for (row, col), cell in table.get_celld().items():
            if row == 0:
                cell.set_text_props(weight='bold')

        pdf_path = self.output_dir / 'tasks_review.pdf'
        plt.savefig(pdf_path, bbox_inches='tight', pad_inches=0.1)
        plt.close(fig)
        
        logger.info(f"Saved task review table to {pdf_path}")
    
    def _generate_custom_plots(self, task_metadata: Dict[str, Dict[str, Any]]) -> None:
        """Generate NL2Bash-specific visualizations."""
        # Plot 1: Utility usage bar chart
        self._plot_utility_usage(task_metadata)
        
        # Plot 2: Feature combination heatmap
        self._plot_feature_combinations(task_metadata)
        
        # Plot 3: Complexity score distribution by category
        self._plot_complexity_by_category(task_metadata)
        
        # Plot 4: Time vs complexity scatter plot
        self._plot_time_vs_complexity()
    
    def _plot_utility_usage(self, task_metadata: Dict[str, Dict[str, Any]]) -> None:
        """Create bar chart of most common utilities."""
        all_utilities = []
        for metadata in task_metadata.values():
            utilities = metadata.get('utilities_used', [])
            if isinstance(utilities, list):
                all_utilities.extend(utilities)
        
        if not all_utilities:
            return
        
        utility_counts = Counter(all_utilities)
        top_utilities = utility_counts.most_common(20)
        
        utilities, counts = zip(*top_utilities)
        
        plt.figure(figsize=(12, 6))
        plt.bar(utilities, counts)
        plt.xticks(rotation=45, ha='right')
        plt.xlabel('Bash Utility')
        plt.ylabel('Number of Tasks')
        plt.title('Top 20 Most Used Bash Utilities')
        plt.tight_layout()
        plt.savefig(self.output_dir / 'utility_usage_chart.png')
        plt.close()
        logger.info("Saved utility usage chart")
    
    def _plot_feature_combinations(self, task_metadata: Dict[str, Dict[str, Any]]) -> None:
        """Create heatmap showing combinations of features."""
        feature_combinations = Counter()
        
        for metadata in task_metadata.values():
            features = []
            if metadata.get('has_pipes', False):
                features.append('Pipes')
            if metadata.get('has_redirects', False):
                features.append('Redirects')
            if metadata.get('has_subcommands', False):
                features.append('Subcommands')
            
            if not features:
                features = ['None']
            
            feature_combinations[tuple(sorted(features))] += 1
        
        # Create matrix for heatmap
        feature_names = ['None', 'Pipes', 'Redirects', 'Subcommands']
        matrix = [[0] * len(feature_names) for _ in range(8)]  # 2^3 combinations
        
        combination_labels = []
        for i, (features, count) in enumerate(sorted(feature_combinations.items())):
            combination_labels.append(' + '.join(features))
            for j, feature in enumerate(feature_names):
                if feature in features:
                    matrix[i][j] = count
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(matrix[:len(combination_labels)], 
                   xticklabels=feature_names,
                   yticklabels=combination_labels,
                   annot=True, fmt='d', cmap='YlOrRd')
        plt.title('Bash Command Feature Combinations')
        plt.tight_layout()
        plt.savefig(self.output_dir / 'feature_combinations_heatmap.png')
        plt.close()
        logger.info("Saved feature combinations heatmap")
    
    def _plot_complexity_by_category(self, task_metadata: Dict[str, Dict[str, Any]]) -> None:
        """Create box plot of complexity scores by category."""
        complexity_data = []
        
        for metadata in task_metadata.values():
            complexity_data.append({
                'category': metadata.get('complexity_category', 'unknown'),
                'score': metadata.get('complexity_score', 0.0)
            })
        
        if not complexity_data:
            return
        
        complexity_df = pd.DataFrame(complexity_data)
        
        plt.figure(figsize=(10, 6))
        category_order = ['very_simple', 'simple', 'medium', 'complex', 'very_complex']
        sns.boxplot(data=complexity_df, x='category', y='score', 
                   order=[c for c in category_order if c in complexity_df['category'].unique()])
        plt.xlabel('Complexity Category')
        plt.ylabel('Complexity Score')
        plt.title('Complexity Score Distribution by Category')
        plt.tight_layout()
        plt.savefig(self.output_dir / 'complexity_by_category.png')
        plt.close()
        logger.info("Saved complexity by category plot")
    
    def _plot_time_vs_complexity(self) -> None:
        """Create scatter plot of completion time vs complexity score."""
        if self.df is None or self.df.empty:
            return
        
        # Merge with task metadata to get complexity scores
        tasks_file = self.input_files[0].parent / f"{self.dataset_name}_tasks.jsonl"
        if not tasks_file.exists():
            return
        
        # Load complexity scores
        complexity_by_task = {}
        try:
            with open(tasks_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        task = json.loads(line)
                        if 'task_id' in task and 'dataset_task_metadata' in task:
                            metadata = task['dataset_task_metadata']
                            complexity_by_task[task['task_id']] = metadata.get('complexity_score', 0.0)
        except Exception as e:
            logger.warning(f"Could not load complexity scores: {e}")
            return
        
        # Add complexity scores to dataframe
        plot_data = []
        for _, row in self.df.iterrows():
            task_id = row.get('task_id')
            if task_id in complexity_by_task:
                plot_data.append({
                    'complexity_score': complexity_by_task[task_id],
                    'human_minutes': row.get('human_minutes', 0),
                    'task_family': row.get('task_family', 'unknown')
                })
        
        if not plot_data:
            return
        
        plot_df = pd.DataFrame(plot_data)
        
        plt.figure(figsize=(10, 6))
        sns.scatterplot(data=plot_df, x='complexity_score', y='human_minutes', 
                       hue='task_family', alpha=0.6)
        plt.xlabel('Complexity Score')
        plt.ylabel('Estimated Time (minutes)')
        plt.title('Task Completion Time vs Complexity Score')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(self.output_dir / 'time_vs_complexity_scatter.png')
        plt.close()
        logger.info("Saved time vs complexity scatter plot") 