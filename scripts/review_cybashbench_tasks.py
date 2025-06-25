#!/usr/bin/env python3
"""
Interactive review tool for CyBashBench tasks.
Modern CLI interface for reviewing and updating task time estimates.
"""

import json
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional
import datetime
import random
import argparse
from collections import Counter

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.layout import Layout
from rich.prompt import Prompt

class CyBashBenchReviewer:
    def __init__(self, input_file: str, rereview=False, time_range: str = None):
        self.input_file = Path(input_file)
        self.rereview = rereview
        self.time_range = time_range
        self.tasks = self._load_tasks()
        self.review_results = []
        self.time_ranges = self._categorize_by_time()
        self.console = Console()
            
    def _load_tasks(self) -> List[Dict[str, Any]]:
        """Load tasks from JSONL file."""
        tasks = []
        with open(self.input_file, 'r') as f:
            for line in f:
                if line.strip():
                    tasks.append(json.loads(line))
        return tasks
    
    def _categorize_by_time(self) -> Dict[str, List[Dict[str, Any]]]:
        """Categorize tasks by time ranges for contextual examples."""
        ranges = {
            'very_fast': [],  # < 3 seconds (0.05 min)
            'fast': [],       # 3-6 seconds (0.05-0.1 min)  
            'medium': [],     # 6-12 seconds (0.1-0.2 min)
            'slow': [],       # 12-24 seconds (0.2-0.4 min)
            'very_slow': []   # > 24 seconds (0.4+ min)
        }
        
        for task in self.tasks:
            time_mins = task.get('human_minutes', 0)
            metadata = task.get('dataset_task_metadata', {})
            
            if time_mins < 0.05:
                range_key = 'very_fast'
            elif time_mins < 0.1:
                range_key = 'fast'
            elif time_mins < 0.2:
                range_key = 'medium'
            elif time_mins < 0.4:
                range_key = 'slow'
            else:
                range_key = 'very_slow'
            
            # Get target/expected answer
            task_type = metadata.get('task_type', 'nl2bash')
            if task_type == 'nl2bash-blanks':
                target = metadata.get('expected_fill', '')
            elif task_type == 'nl2bash-prefixed':
                target = metadata.get('expected_completion', '')
            elif task_type == 'mcq':
                target = metadata.get('correct_answer', '')
            elif task_type == 'single-char':
                target = metadata.get('expected_completion', '')
            else:
                target = metadata.get('bash_command', '')
            
            ranges[range_key].append({
                'desc': metadata.get('nl_description', ''),
                'target': target,
                'time_mins': time_mins,
                'time_secs': time_mins * 60,
                'type': task_type,
                'source': metadata.get('timing_source', '')
            })
        
        return ranges
    
    def _get_time_category(self, time_mins: float) -> str:
        """Get the time category for a given time in minutes."""
        if time_mins < 0.05:
            return 'very_fast'
        elif time_mins < 0.1:
            return 'fast'
        elif time_mins < 0.2:
            return 'medium'
        elif time_mins < 0.4:
            return 'slow'
        else:
            return 'very_slow'
    
    def _create_distribution_plot(self) -> str:
        """Create a simple ASCII histogram of time distribution."""
        # Collect all times in seconds
        times = [task.get('human_minutes', 0) * 60 for task in self.tasks]
        
        # Create buckets
        buckets = [0, 1, 2, 3, 5, 10, 15, 20, 30, 60]
        bucket_counts = [0] * (len(buckets) - 1)
        bucket_labels = []
        
        for i in range(len(buckets) - 1):
            start, end = buckets[i], buckets[i + 1]
            count = sum(1 for t in times if start <= t < end)
            bucket_counts[i] = count
            bucket_labels.append(f"{start}-{end}s")
        
        # Handle overflow
        overflow = sum(1 for t in times if t >= buckets[-1])
        if overflow > 0:
            bucket_counts.append(overflow)
            bucket_labels.append(f"{buckets[-1]}s+")
        
        # Create ASCII plot
        max_count = max(bucket_counts) if bucket_counts else 1
        plot_lines = []
        
        for label, count in zip(bucket_labels, bucket_counts):
            bar_length = int((count / max_count) * 30) if max_count > 0 else 0
            bar = '█' * bar_length
            plot_lines.append(f"{label:>6} │{bar:<30} {count:>3}")
        
        return '\n'.join(plot_lines)
    
    def _get_time_range_examples(self, current_time_mins: float) -> List[Dict[str, Any]]:
        """Get examples from the same time range."""
        range_key = self._get_time_category(current_time_mins)
        
        examples = self.time_ranges[range_key]
        if len(examples) < 2:
            return []
        
        # Get 3 random examples, prioritizing shorter descriptions
        suitable_examples = [ex for ex in examples if len(ex['desc']) < 100]
        selected = random.sample(suitable_examples, min(3, len(suitable_examples)))
        return selected
    
    def _format_task_rich(self, task: Dict[str, Any], task_num: int, total_tasks: int, original_pos: int, total_original: int) -> None:
        """Format task using Rich with responsive layout based on terminal height."""
        metadata = task.get('dataset_task_metadata', {})
        task_type = metadata.get('task_type', 'nl2bash')
        nl = metadata.get('nl_description', '')
        time_mins = task.get('human_minutes', 0)
        time_secs = time_mins * 60
        timing_source = metadata.get('timing_source', 'unknown')
        
        # Get terminal height and determine layout
        terminal_height = self.console.size.height
        
        if terminal_height < 15:
            # Ultra compact for very small terminals
            header = f"[bold blue]TASK {task_num}/{total_tasks}[/] [cyan]{task['task_id'].split('/')[-1]}[/] [green]{time_secs:.1f}s[/]"
            self.console.print(f"\n{header}")
            self.console.print(f"[dim]ID:[/] [cyan]{task['task_id']}[/]")
            self.console.print(f"[bold]{nl}[/]")
            
            # Show task-specific info
            if task_type == 'nl2bash-blanks':
                expected = metadata.get('expected_fill', '')
                self.console.print(f"[dim]Fill:[/] [green]{expected}[/]")
            elif task_type in ['nl2bash-prefixed', 'nl2bash-prefix']:
                prefix = metadata.get('command_prefix', '')
                completion = metadata.get('expected_completion', '')
                self.console.print(f"[dim]Complete:[/] [cyan]{prefix}[/cyan][yellow]___[/yellow]")
                self.console.print(f"[dim]Answer:[/] [green]{prefix}{completion}[/]")
            elif task_type == 'mcq':
                expected = metadata.get('correct_answer', '')
                self.console.print(f"[dim]Answer:[/] [green]{expected}[/]")
            elif task_type == 'single-char':
                expected = metadata.get('expected_completion', '')
                self.console.print(f"[dim]Complete:[/] [green]{expected}[/]")
            else:
                expected = metadata.get('bash_command', '')
                self.console.print(f"[dim]Expected:[/] [green]{expected}[/]")
            self.console.print("[green]ENTER[/]=Accept [red]D[/]=Delete [yellow]Number[/]=Update [magenta]Q[/]=Quit")
            
        elif terminal_height < 25:
            # Medium layout with some examples
            layout = Layout()
            layout.split_row(
                Layout(name="task", ratio=3),
                Layout(name="examples", ratio=1)
            )
            
            # Task content
            header_text = f"[bold blue]TASK {task_num}/{total_tasks}[/] [cyan]{task['task_id']}[/] [yellow]{task_type.upper()}[/] [green]{time_secs:.1f}s[/]"
            
            # Format task based on type
            if task_type == 'nl2bash':
                expected = metadata.get('bash_command', '')
                task_content = f"{header_text}\n\n[bold cyan]ID:[/] {task['task_id']}\n[bold]{nl}[/]\n[dim]→ Convert to bash command[/]\n[dim]Expected:[/] [green]{expected}[/]"
            elif task_type in ['nl2bash-prefixed', 'nl2bash-prefix']:
                prefix = metadata.get('command_prefix', '')
                completion = metadata.get('expected_completion', '')
                full_command = f"{prefix}{completion}"
                task_content = f"{header_text}\n\n[bold cyan]ID:[/] {task['task_id']}\n[bold]{nl}[/]\n[bold white]Complete this:[/] [cyan]{prefix}[/cyan][yellow]___|[/yellow]\n[dim]Full answer:[/] [green]{full_command}[/]"
            elif task_type == 'nl2bash-blanks':
                template = metadata.get('template', '')
                expected_fill = metadata.get('expected_fill', '')
                blank_template = template.replace('___', '[red]___[/red]')
                task_content = f"{header_text}\n\n[bold cyan]ID:[/] {task['task_id']}\n[bold]{nl}[/]\n[dim]Fill:[/] {blank_template}\n[dim]Answer:[/] [green]{expected_fill}[/]"
            elif task_type == 'contextual':
                context = metadata.get('context', '')
                expected = metadata.get('bash_command', '')
                task_content = f"{header_text}\n\n[bold cyan]ID:[/] {task['task_id']}\n[dim]Context:[/] {context}\n[bold]{nl}[/]\n[dim]Expected:[/] [green]{expected}[/]"
            elif task_type == 'multi-step':
                prev_cmd = metadata.get('previous_command', '')
                prev_out = metadata.get('previous_output', '')
                expected = metadata.get('bash_command', '')
                task_content = f"{header_text}\n\n[bold cyan]ID:[/] {task['task_id']}\n[dim]Previous:[/] [cyan]{prev_cmd}[/] → [yellow]{prev_out}[/]\n[bold]{nl}[/]\n[dim]Expected:[/] [green]{expected}[/]"
            elif task_type == 'mcq':
                choices = metadata.get('choices', [])
                correct_answer = metadata.get('correct_answer', '')
                choices_text = '\n'.join([f"{chr(65+i)}. {choice}" for i, choice in enumerate(choices)])
                task_content = f"{header_text}\n\n[bold cyan]ID:[/] {task['task_id']}\n[bold]{nl}[/]\n\n{choices_text}\n\n[dim]Answer:[/] [green]{correct_answer}[/]"
            elif task_type == 'single-char':
                prefix = metadata.get('command_prefix', '')
                expected_completion = metadata.get('expected_completion', '')
                task_content = f"{header_text}\n\n[bold cyan]ID:[/] {task['task_id']}\n[bold]{nl}[/]\n[dim]Complete:[/] [cyan]{prefix}[/]\n[dim]Answer:[/] [green]{expected_completion}[/]"
            else:
                task_content = f"{header_text}\n\n[bold cyan]ID:[/] {task['task_id']}\n[bold]{nl}[/]"
            
            # Expected answer is already included in task-specific formatting above
            
            layout["task"].update(Panel(task_content, style="white"))
            
            # Examples
            examples = self._get_time_range_examples(time_mins)
            if examples:
                if time_mins < 0.05:
                    range_desc = '< 3s'
                elif time_mins < 0.1:
                    range_desc = '3-6s'
                elif time_mins < 0.2:
                    range_desc = '6-12s'
                elif time_mins < 0.4:
                    range_desc = '12-24s'
                else:
                    range_desc = '> 24s'
                
                examples_content = f"[cyan]Similar ({range_desc}):[/]\n\n"
                for ex in examples[:3]:
                    desc = ex['desc'][:30] + "..." if len(ex['desc']) > 30 else ex['desc']
                    target = ex['target'][:15] + "..." if len(ex['target']) > 15 else ex['target']
                    examples_content += f"[yellow]{ex['time_secs']:.1f}s[/] {desc}\n[dim]→ {target}[/]\n\n"
            else:
                examples_content = "[dim]No similar examples[/]"
            
            layout["examples"].update(Panel(examples_content, title="Examples", style="cyan"))
            
            self.console.print(layout)
            self.console.print("\n[green]ENTER[/]=Accept [red]D[/]=Delete [yellow]Number[/]=Update(seconds) [magenta]Q[/]=Quit")
            
        else:
            # Full layout for tall terminals
            layout = Layout()
            layout.split_column(
                Layout(name="header", size=3),
                Layout(name="main"),
                Layout(name="footer", size=3)
            )
            
            layout["main"].split_row(
                Layout(name="task", ratio=2),
                Layout(name="examples", ratio=1)
            )
            
            # Header
            header_text = f"[bold blue]TASK REVIEW[/] [magenta]{task_num}/{total_tasks}[/] | [dim]Original: {original_pos}/{total_original}[/]\n"
            header_text += f"[cyan]{task['task_id']}[/]"
            layout["header"].update(Panel(header_text, style="blue"))
            
            # Task content
            task_content = f"[bold cyan]ID:[/] {task['task_id']}\n"
            task_content += f"[yellow]Type:[/] {task_type.upper()} | [yellow]Category:[/] {metadata.get('security_category', 'N/A')}\n"
            task_content += f"[yellow]Source:[/] {timing_source} | [green]Current: {time_secs:.1f}s[/] [dim]({time_mins:.4f}min)[/]\n\n"
            
            # Format task based on type (full version)
            if task_type == 'nl2bash':
                expected = metadata.get('bash_command', '')
                task_content += f"[bold white]Task:[/] {nl}\n\n[dim]→ Convert to bash command[/]\n[bold white]Expected:[/] [green]{expected}[/]"
            elif task_type in ['nl2bash-prefixed', 'nl2bash-prefix']:
                prefix = metadata.get('command_prefix', '')
                completion = metadata.get('expected_completion', '')
                full_command = f"{prefix}{completion}"
                task_content += f"[bold white]Task:[/] {nl}\n[bold white]Complete this:[/] [cyan]{prefix}[/cyan][yellow]___|[/yellow]\n[bold white]Full answer:[/] [green]{full_command}[/]"
            elif task_type == 'nl2bash-blanks':
                template = metadata.get('template', '')
                expected_fill = metadata.get('expected_fill', '')
                blank_template = template.replace('___', '[red]___[/red]')
                task_content += f"[bold white]Task:[/] {nl}\n[bold white]Fill blank:[/] {blank_template}\n[bold white]Answer:[/] [green]{expected_fill}[/]"
            elif task_type == 'contextual':
                context = metadata.get('context', '')
                expected = metadata.get('bash_command', '')
                task_content += f"[yellow]Context:[/] {context}\n[bold white]Task:[/] {nl}\n[bold white]Expected:[/] [green]{expected}[/]"
            elif task_type == 'multi-step':
                prev_cmd = metadata.get('previous_command', '')
                prev_out = metadata.get('previous_output', '')
                expected = metadata.get('bash_command', '')
                task_content += f"[bold white]Previous:[/] [cyan]{prev_cmd}[/]\n[bold white]Output:[/] [yellow]{prev_out}[/]\n[bold white]Next:[/] {nl}\n[bold white]Expected:[/] [green]{expected}[/]"
            elif task_type == 'mcq':
                choices = metadata.get('choices', [])
                correct_answer = metadata.get('correct_answer', '')
                choices_text = '\n'.join([f"{chr(65+i)}. {choice}" for i, choice in enumerate(choices)])
                task_content += f"[bold white]Question:[/] {nl}\n\n{choices_text}\n\n[bold white]Answer:[/] [green]{correct_answer}[/]"
            elif task_type == 'single-char':
                prefix = metadata.get('command_prefix', '')
                expected_completion = metadata.get('expected_completion', '')
                task_content += f"[bold white]Task:[/] {nl}\n[bold white]Complete:[/] [cyan]{prefix}[/]\n[bold white]Answer:[/] [green]{expected_completion}[/]"
            
            # Expected answer is already included in task-specific formatting above
            
            layout["task"].update(Panel(task_content, title="Task Content", style="white"))
            
            # Examples panel
            examples = self._get_time_range_examples(time_mins)
            if examples:
                if time_mins < 0.05:
                    range_desc = '< 3s'
                elif time_mins < 0.1:
                    range_desc = '3-6s'
                elif time_mins < 0.2:
                    range_desc = '6-12s'
                elif time_mins < 0.4:
                    range_desc = '12-24s'
                else:
                    range_desc = '> 24s'
                
                examples_content = f"[cyan]Similar timing ({range_desc}):[/]\n\n"
                
                for ex in examples:
                    desc = ex['desc'][:45] + "..." if len(ex['desc']) > 45 else ex['desc']
                    target = ex['target'][:20] + "..." if len(ex['target']) > 20 else ex['target']
                    examples_content += f"[yellow]{ex['time_secs']:.1f}s[/] {desc}\n[dim]→ {target}[/]\n\n"
            else:
                examples_content = "[dim]No similar examples available[/]"
            
            layout["examples"].update(Panel(examples_content, title="Similar Tasks", style="cyan"))
            
            # Footer with controls
            footer_content = "[bold green]ENTER[/] Accept | [bold red]D[/] Delete | [bold yellow]Number[/] Update (seconds) | [bold magenta]Q[/] Quit | [bold cyan]H[/] Help"
            layout["footer"].update(Panel(footer_content, title="Controls", style="blue"))
            
            self.console.print(layout)
    
    def _print_summary_stats(self):
        """Print summary statistics at the start."""
        timing_sources = Counter()
        for task in self.tasks:
            source = task.get('dataset_task_metadata', {}).get('timing_source', 'unknown')
            timing_sources[source] += 1
        
        # Create summary table
        table = Table(title="CyBashBench Task Review - Summary")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")
        
        table.add_row("Total tasks", str(len(self.tasks)))
        table.add_row("", "")  # Spacer
        
        for source, count in timing_sources.items():
            table.add_row(f"• {source}", str(count))
        
        self.console.print(table)
        
        # Distribution plot
        plot_panel = Panel(self._create_distribution_plot(), title="Time Distribution", style="blue")
        self.console.print(plot_panel)
    
    def _save_review_result(self, task: Dict[str, Any], action: str, new_time: Optional[float] = None):
        """Save review result."""
        result = {
            'task_id': task['task_id'],
            'original_time_minutes': task.get('human_minutes', 0),
            'original_time_seconds': task.get('human_minutes', 0) * 60,
            'action': action,
            'reviewed_at': datetime.datetime.now().isoformat()
        }
        
        if action == 'updated' and new_time is not None:
            result['new_time_minutes'] = new_time
            result['new_time_seconds'] = new_time * 60
        
        self.review_results.append(result)
    
    def _update_task_timing_source(self, task: Dict[str, Any], action: str) -> Dict[str, Any]:
        """Update timing_source field based on review action."""
        updated_task = task.copy()
        updated_task['dataset_task_metadata'] = task['dataset_task_metadata'].copy()
        
        current_source = task.get('dataset_task_metadata', {}).get('timing_source', '')
        
        if current_source != 'human_calibration' and action in ['accepted', 'updated']:
            updated_task['dataset_task_metadata']['timing_source'] = 'human_reviewed'
        
        return updated_task
    
    def review_tasks(self):
        """Main review loop."""
        self._print_summary_stats()
        
        # Filter tasks that need review
        tasks_to_review = []
        skipped_count = 0
        time_filtered_count = 0
        
        for task in self.tasks:
            timing_source = task.get('dataset_task_metadata', {}).get('timing_source', '')
            time_mins = task.get('human_minutes', 0)
            time_category = self._get_time_category(time_mins)
            
            # Skip if not in rereview mode and already reviewed
            if not self.rereview and timing_source in ['human_reviewed', 'human_calibration']:
                skipped_count += 1
                continue
                
            # If time range specified, filter by time category
            if self.time_range and time_category != self.time_range:
                time_filtered_count += 1
                continue
                
            tasks_to_review.append(task)
        
        # Show filtering status messages
        if skipped_count > 0 and not self.rereview:
            self.console.print(f"\n[cyan]ℹ Skipping[/] [yellow]{skipped_count}[/] tasks that are already human-reviewed or human-calibrated")
        elif self.rereview:
            if self.time_range:
                time_range_desc = {
                    'very_fast': '< 3s',
                    'fast': '3-6s', 
                    'medium': '6-12s',
                    'slow': '12-24s',
                    'very_slow': '> 24s'
                }[self.time_range]
                self.console.print(f"\n[yellow]ℹ Re-review mode:[/] Including {self.time_range} tasks ({time_range_desc}) only")
            else:
                self.console.print(f"\n[yellow]ℹ Re-review mode:[/] Including all tasks (even previously reviewed)")
                
        if time_filtered_count > 0:
            self.console.print(f"[cyan]ℹ Time filtered:[/] [yellow]{time_filtered_count}[/] tasks outside selected time range")
        
        if not tasks_to_review:
            self.console.print("\n[green]✓ All tasks have been reviewed![/] No tasks need review.")
            return
        
        self.console.print(f"\n[green]Reviewing[/] [yellow]{len(tasks_to_review)}[/] tasks that need review")
        
        for i, task in enumerate(tasks_to_review):
            task_num = i + 1
            total_to_review = len(tasks_to_review)
            original_position = self.tasks.index(task) + 1
            total_tasks = len(self.tasks)
            
            self._format_task_rich(task, task_num, total_to_review, original_position, total_tasks)
            user_input = Prompt.ask("\n[bold]Action[/]", default="")
            
            if user_input == '':
                self._save_review_result(task, 'accepted')
                self.console.print("[green]✓ Accepted[/]")
                    
            elif user_input.upper() == 'D':
                self._save_review_result(task, 'deleted')
                self.console.print("[red]✗ Task marked for deletion[/]")
                    
            elif user_input.upper() == 'Q':
                self.console.print("\n[yellow]Saving progress and exiting...[/]")
                self._save_updated_tasks()
                return
                
            else:
                try:
                    new_time_seconds = float(user_input)
                    new_time_minutes = new_time_seconds / 60
                    self._save_review_result(task, 'updated', new_time_minutes)
                    self.console.print(f"[green]✓ Time updated to[/] [bold]{new_time_seconds}s[/] [dim]({new_time_minutes:.4f} minutes)[/]")
                except ValueError:
                    self.console.print(f"[red]Invalid input: '{user_input}'. Please try again.[/]")
                    continue
        
        self._save_updated_tasks()
    
    def _save_updated_tasks(self):
        """Save the updated task list based on review results."""
        actions = {result['task_id']: result for result in self.review_results}
        
        # Create backup of original file
        backup_file = self.input_file.with_suffix(self.input_file.suffix + '.backup')
        import shutil
        shutil.copy2(self.input_file, backup_file)
        
        # Write updated tasks directly to the input file
        with open(self.input_file, 'w') as f:
            for task in self.tasks:
                task_id = task['task_id']
                if task_id in actions:
                    action = actions[task_id]
                    if action['action'] == 'deleted':
                        continue
                    
                    updated_task = self._update_task_timing_source(task, action['action'])
                    
                    if action['action'] == 'updated':
                        updated_task['human_minutes'] = action['new_time_minutes']
                    
                    f.write(json.dumps(updated_task) + '\n')
                else:
                    f.write(json.dumps(task) + '\n')
        
        # Print summary
        accepted = sum(1 for r in self.review_results if r['action'] == 'accepted')
        updated = sum(1 for r in self.review_results if r['action'] == 'updated')
        deleted = sum(1 for r in self.review_results if r['action'] == 'deleted')
        
        summary_table = Table(title="Review Summary")
        summary_table.add_column("Action", style="cyan")
        summary_table.add_column("Count", style="green")
        
        summary_table.add_row("Accepted", str(accepted))
        summary_table.add_row("Updated", str(updated))
        summary_table.add_row("Deleted", str(deleted))
        summary_table.add_row("Total reviewed", str(len(self.review_results)))
        
        self.console.print(summary_table)
        self.console.print(f"\n[cyan]Files saved:[/]\n• Updated tasks: {self.input_file}\n• Backup: {backup_file}")

def main():
    parser = argparse.ArgumentParser(description="Interactive CyBashBench task reviewer")
    parser.add_argument("input_file", nargs="?", 
                       default="/Users/speters/git/personal/human-ttc-eval/data/keep/cybashbench/cybashbench_tasks.jsonl",
                       help="Input JSONL file with tasks")
    parser.add_argument("--rereview", nargs="?", const=True, 
                       choices=['very_fast', 'fast', 'medium', 'slow', 'very_slow'],
                       help="Include previously reviewed tasks for re-review. Optionally specify time range: very_fast (<3s), fast (3-6s), medium (6-12s), slow (12-24s), very_slow (>24s)")
    
    args = parser.parse_args()
    
    if not Path(args.input_file).exists():
        print(f"Error: Input file '{args.input_file}' not found")
        sys.exit(1)
    
    # Handle rereview argument - can be True (boolean) or a time range string
    rereview_enabled = bool(args.rereview)
    time_range = args.rereview if isinstance(args.rereview, str) else None
    
    reviewer = CyBashBenchReviewer(args.input_file, rereview_enabled, time_range)
    
    try:
        reviewer.review_tasks()
    except KeyboardInterrupt:
        reviewer.console.print("\n[yellow]Review interrupted. Progress saved.[/]")
        reviewer._save_updated_tasks()

if __name__ == "__main__":
    main()