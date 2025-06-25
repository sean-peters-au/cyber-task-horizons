#!/usr/bin/env python3
"""
Interactive review tool for CyBashBench tasks.
Allows viewing tasks as they would be presented, with time estimates,
and updating/deleting tasks.
"""

import json
import os
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import datetime
import random

# ANSI color codes
class Colors:
    RED = '\033[91m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    MAGENTA = '\033[95m'
    CYAN = '\033[96m'
    WHITE = '\033[97m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'
    
    @staticmethod
    def colorize(text: str, color: str) -> str:
        return f"{color}{text}{Colors.END}"

class CyBashBenchReviewer:
    def __init__(self, input_file: str, output_file: str):
        self.input_file = Path(input_file)
        self.output_file = Path(output_file)
        self.tasks = self._load_tasks()
        self.review_results = []
        self.time_ranges = self._categorize_by_time()
        
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
            
            ranges[range_key].append({
                'desc': task.get('dataset_task_metadata', {}).get('nl_description', ''),
                'time_mins': time_mins,
                'time_secs': time_mins * 60,
                'type': task.get('dataset_task_metadata', {}).get('task_type', ''),
                'source': task.get('dataset_task_metadata', {}).get('timing_source', '')
            })
        
        return ranges
    
    def _get_time_range_examples(self, current_time_mins: float) -> str:
        """Get examples from the same time range."""
        # Determine range
        if current_time_mins < 0.05:
            range_key = 'very_fast'
            range_desc = "< 3 seconds"
        elif current_time_mins < 0.1:
            range_key = 'fast'
            range_desc = "3-6 seconds"
        elif current_time_mins < 0.2:
            range_key = 'medium'
            range_desc = "6-12 seconds"
        elif current_time_mins < 0.4:
            range_key = 'slow'
            range_desc = "12-24 seconds"
        else:
            range_key = 'very_slow'
            range_desc = "> 24 seconds"
        
        examples = self.time_ranges[range_key]
        if len(examples) < 2:
            return f"Similar timing range ({range_desc}): No other examples available\n"
        
        # Get 3 random examples, excluding very long descriptions
        suitable_examples = [ex for ex in examples if len(ex['desc']) < 80]
        selected = random.sample(suitable_examples, min(3, len(suitable_examples)))
        
        example_strings = []
        for ex in selected:
            desc = ex['desc'][:50] + "..." if len(ex['desc']) > 50 else ex['desc']
            example_strings.append(f"[{ex['time_secs']:.1f}s] {desc}")
        
        examples_text = " • ".join(example_strings)
        return f"Similar timing range ({Colors.colorize(range_desc, Colors.CYAN)}): {examples_text}\n"
    
    def _format_task_prompt(self, task: Dict[str, Any], task_num: int, total_tasks: int) -> str:
        """Format task as it would appear in the benchmark."""
        metadata = task.get('dataset_task_metadata', {})
        task_type = metadata.get('task_type', 'nl2bash')
        nl = metadata.get('nl_description', '')
        time_mins = task.get('human_minutes', 0)
        time_secs = time_mins * 60
        timing_source = metadata.get('timing_source', 'unknown')
        
        # Header with progress
        header = f"\n{Colors.colorize('=' * 80, Colors.BLUE)}\n"
        header += f"{Colors.colorize('TASK', Colors.BOLD)} {Colors.colorize(f'{task_num}/{total_tasks}', Colors.MAGENTA)} "
        header += f"| {Colors.colorize(task['task_id'], Colors.CYAN)}\n"
        header += f"{Colors.colorize('=' * 80, Colors.BLUE)}\n"
        
        # Task info
        info = f"Type: {Colors.colorize(task_type.upper(), Colors.YELLOW)} | "
        info += f"Category: {Colors.colorize(metadata.get('security_category', 'N/A'), Colors.YELLOW)} | "
        info += f"Source: {Colors.colorize(timing_source, Colors.YELLOW)}\n"
        info += f"Current estimate: {Colors.colorize(f'{time_secs:.1f} seconds', Colors.GREEN)} "
        info += f"({Colors.colorize(f'{time_mins:.4f} minutes', Colors.GREEN)})\n\n"
        
        # Similar timing examples
        examples = self._get_time_range_examples(time_mins)
        examples_section = f"{Colors.colorize('Similar timing examples:', Colors.UNDERLINE)}\n{examples}\n"
        
        # Task content
        content_header = f"{Colors.colorize('TASK CONTENT:', Colors.BOLD + Colors.WHITE)}\n"
        content_header += f"{Colors.colorize('-' * 40, Colors.WHITE)}\n"
        
        # Format based on task type
        task_content = ""
        if task_type == 'nl2bash':
            task_content = f"Task: {nl}\n\n[Expected to convert this to a bash command]"
            
        elif task_type == 'nl2bash-prefixed':
            prefix = metadata.get('command_prefix', '')
            task_content = f"{nl}\nComplete the command: `{Colors.colorize(prefix, Colors.CYAN)}`"
            
        elif task_type == 'nl2bash-blanks':
            template = metadata.get('template', '')
            blank_template = template.replace('___', Colors.colorize('___', Colors.RED))
            task_content = f"{nl}\nFill in the blank: `{blank_template}`"
            
        elif task_type == 'contextual':
            context = metadata.get('context', '')
            task_content = f"Context: {Colors.colorize(context, Colors.YELLOW)}\nTask: {nl}"
            
        elif task_type == 'multi-step':
            prev_cmd = metadata.get('previous_command', '')
            prev_out = metadata.get('previous_output', '')
            task_content = f"Previous command: `{Colors.colorize(prev_cmd, Colors.CYAN)}`\n"
            task_content += f"Output: `{Colors.colorize(prev_out, Colors.YELLOW)}`\n"
            task_content += f"Next task: {nl}"
            
        elif task_type == 'mcq':
            choices = metadata.get('choices', [])
            choices_text = '\n'.join([f"{chr(65+i)}. {choice}" for i, choice in enumerate(choices)])
            task_content = f"{nl}\n\n{choices_text}\n\nAnswer:"
            
        elif task_type == 'single-char':
            prefix = metadata.get('command_prefix', '')
            task_content = f"{nl}\nComplete: `{Colors.colorize(prefix, Colors.CYAN)}`"
        
        # Expected answer
        answer_section = f"\n{Colors.colorize('-' * 40, Colors.WHITE)}\n"
        answer_section += f"{Colors.colorize('EXPECTED ANSWER:', Colors.BOLD + Colors.WHITE)}\n"
        
        if task_type == 'nl2bash-blanks':
            answer_section += f"Fill: {Colors.colorize(metadata.get('expected_fill', ''), Colors.GREEN)}"
        elif task_type == 'nl2bash-prefixed':
            answer_section += f"Completion: {Colors.colorize(metadata.get('expected_completion', ''), Colors.GREEN)}"
        elif task_type == 'mcq':
            answer_section += f"Correct: {Colors.colorize(metadata.get('correct_answer', ''), Colors.GREEN)}"
        elif task_type == 'single-char':
            answer_section += f"Completion: {Colors.colorize(metadata.get('expected_completion', ''), Colors.GREEN)}"
        else:
            answer_section += f"Command: {Colors.colorize(metadata.get('bash_command', ''), Colors.GREEN)}"
        
        return header + info + examples_section + content_header + task_content + answer_section
    
    def _save_review_result(self, task: Dict[str, Any], action: str, 
                           new_time: Optional[float] = None):
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
    
    def _print_summary_stats(self):
        """Print summary statistics at the start."""
        timing_sources = {}
        for task in self.tasks:
            source = task.get('dataset_task_metadata', {}).get('timing_source', 'unknown')
            timing_sources[source] = timing_sources.get(source, 0) + 1
        
        print(f"\n{Colors.colorize('CYBASHBENCH TASK REVIEW', Colors.BOLD + Colors.WHITE)}")
        print(f"{Colors.colorize('=' * 50, Colors.BLUE)}")
        print(f"Total tasks: {Colors.colorize(str(len(self.tasks)), Colors.GREEN)}")
        print(f"Timing sources:")
        for source, count in timing_sources.items():
            print(f"  • {Colors.colorize(source, Colors.YELLOW)}: {Colors.colorize(str(count), Colors.GREEN)} tasks")
        
        print(f"\n{Colors.colorize('Time distribution:', Colors.UNDERLINE)}")
        for range_name, tasks in self.time_ranges.items():
            range_labels = {
                'very_fast': '< 3s',
                'fast': '3-6s', 
                'medium': '6-12s',
                'slow': '12-24s',
                'very_slow': '> 24s'
            }
            print(f"  • {Colors.colorize(range_labels[range_name], Colors.CYAN)}: {Colors.colorize(str(len(tasks)), Colors.GREEN)} tasks")
    
    def _print_help(self):
        """Print help information."""
        help_box = f"""
{Colors.colorize('┌─ CONTROLS ─────────────────────────────────────────────────┐', Colors.BLUE)}
{Colors.colorize('│', Colors.BLUE)} {Colors.colorize('ENTER', Colors.GREEN)}     Accept current time estimate and continue        {Colors.colorize('│', Colors.BLUE)}  
{Colors.colorize('│', Colors.BLUE)} {Colors.colorize('D + ENTER', Colors.RED)}  Delete this task from the dataset               {Colors.colorize('│', Colors.BLUE)}
{Colors.colorize('│', Colors.BLUE)} {Colors.colorize('Number', Colors.YELLOW)}     Update time estimate (in seconds, e.g., "2.5")  {Colors.colorize('│', Colors.BLUE)}
{Colors.colorize('│', Colors.BLUE)} {Colors.colorize('Q + ENTER', Colors.MAGENTA)} Quit and save progress                          {Colors.colorize('│', Colors.BLUE)}
{Colors.colorize('│', Colors.BLUE)} {Colors.colorize('H + ENTER', Colors.CYAN)}  Show this help again                            {Colors.colorize('│', Colors.BLUE)}
{Colors.colorize('└────────────────────────────────────────────────────────────┘', Colors.BLUE)}
        """
        print(help_box)
    
    def _update_task_timing_source(self, task: Dict[str, Any], action: str) -> Dict[str, Any]:
        """Update timing_source field based on review action."""
        # Make a copy to avoid modifying original
        updated_task = task.copy()
        updated_task['dataset_task_metadata'] = task['dataset_task_metadata'].copy()
        
        current_source = task.get('dataset_task_metadata', {}).get('timing_source', '')
        
        # Only update if it's not already human_calibration and action involves human review
        if current_source != 'human_calibration' and action in ['accepted', 'updated']:
            updated_task['dataset_task_metadata']['timing_source'] = 'human_reviewed'
        
        return updated_task
    
    def _save_results(self):
        """Save review results to file."""
        with open(self.output_file, 'w') as f:
            for result in self.review_results:
                f.write(json.dumps(result) + '\n')
    
    def review_tasks(self):
        """Main review loop."""
        self._print_summary_stats()
        self._print_help()
        
        # Filter tasks that need review
        tasks_to_review = []
        skipped_count = 0
        
        for task in self.tasks:
            timing_source = task.get('dataset_task_metadata', {}).get('timing_source', '')
            if timing_source in ['human_reviewed', 'human_calibration']:
                skipped_count += 1
            else:
                tasks_to_review.append(task)
        
        if skipped_count > 0:
            print(f"\n{Colors.colorize('ℹ Skipping', Colors.CYAN)} {Colors.colorize(str(skipped_count), Colors.YELLOW)} tasks that are already human-reviewed or human-calibrated")
        
        if not tasks_to_review:
            print(f"\n{Colors.colorize('✓ All tasks have been reviewed!', Colors.GREEN)} No tasks need review.")
            return
        
        print(f"\n{Colors.colorize('Reviewing', Colors.GREEN)} {Colors.colorize(str(len(tasks_to_review)), Colors.YELLOW)} tasks that need review\n")
        
        for i, task in enumerate(tasks_to_review):
            task_num = i + 1
            total_to_review = len(tasks_to_review)
            
            # Find original position in full task list
            original_position = self.tasks.index(task) + 1
            total_tasks = len(self.tasks)
            
            # Show both the review progress and original position
            print(f"\n{Colors.colorize('REVIEW PROGRESS:', Colors.BOLD)} {task_num}/{total_to_review} | {Colors.colorize('ORIGINAL POSITION:', Colors.BOLD)} {original_position}/{total_tasks}")
            print(self._format_task_prompt(task, task_num, total_to_review))
            
            while True:
                user_input = input(f"\n{Colors.colorize('Action:', Colors.BOLD)} ").strip()
                
                if user_input == '':
                    # Accept and continue
                    self._save_review_result(task, 'accepted')
                    print(f"{Colors.colorize('✓ Accepted', Colors.GREEN)}")
                    break
                    
                elif user_input.upper() == 'D':
                    # Delete task
                    self._save_review_result(task, 'deleted')
                    print(f"{Colors.colorize('✗ Task marked for deletion', Colors.RED)}")
                    break
                    
                elif user_input.upper() == 'Q':
                    # Quit
                    print(f"\n{Colors.colorize('Saving progress and exiting...', Colors.YELLOW)}")
                    self._save_results()
                    self._save_updated_tasks()
                    return
                    
                elif user_input.upper() == 'H':
                    # Show help
                    self._print_help()
                    continue
                    
                else:
                    # Try to parse as number (seconds)
                    try:
                        new_time_seconds = float(user_input)
                        new_time_minutes = new_time_seconds / 60
                        self._save_review_result(task, 'updated', new_time_minutes)
                        print(f"{Colors.colorize('✓ Time updated to', Colors.GREEN)} "
                              f"{Colors.colorize(f'{new_time_seconds}s', Colors.BOLD)} "
                              f"({new_time_minutes:.4f} minutes)")
                        break
                    except ValueError:
                        print(f"{Colors.colorize(f'Invalid input: \"{user_input}\". Please try again.', Colors.RED)}")
        
        # Save when done
        print(f"\n{Colors.colorize('Review complete! Saving results...', Colors.GREEN)}")
        self._save_results()
        self._save_updated_tasks()
    
    def _save_updated_tasks(self):
        """Save the updated task list based on review results."""
        # Create a mapping of actions by task_id
        actions = {}
        for result in self.review_results:
            actions[result['task_id']] = result
        
        # Write updated tasks
        updated_file = self.input_file.parent / "cybashbench_tasks_reviewed.jsonl"
        with open(updated_file, 'w') as f:
            for task in self.tasks:
                task_id = task['task_id']
                if task_id in actions:
                    action = actions[task_id]
                    if action['action'] == 'deleted':
                        continue  # Skip deleted tasks
                    
                    # Update timing source for accepted/updated tasks
                    updated_task = self._update_task_timing_source(task, action['action'])
                    
                    if action['action'] == 'updated':
                        # Update time
                        updated_task['human_minutes'] = action['new_time_minutes']
                    
                    f.write(json.dumps(updated_task) + '\n')
                else:
                    # Task not reviewed yet, save as-is
                    f.write(json.dumps(task) + '\n')
        
        print(f"\n{Colors.colorize('FILES SAVED:', Colors.BOLD)}")
        print(f"  • Updated tasks: {Colors.colorize(str(updated_file), Colors.CYAN)}")
        print(f"  • Review log: {Colors.colorize(str(self.output_file), Colors.CYAN)}")
        
        # Print summary
        accepted = sum(1 for r in self.review_results if r['action'] == 'accepted')
        updated = sum(1 for r in self.review_results if r['action'] == 'updated')
        deleted = sum(1 for r in self.review_results if r['action'] == 'deleted')
        
        print(f"\n{Colors.colorize('REVIEW SUMMARY:', Colors.BOLD)}")
        print(f"  • {Colors.colorize('Accepted:', Colors.GREEN)} {accepted}")
        print(f"  • {Colors.colorize('Updated:', Colors.YELLOW)} {updated}")
        print(f"  • {Colors.colorize('Deleted:', Colors.RED)} {deleted}")
        print(f"  • {Colors.colorize('Total reviewed:', Colors.MAGENTA)} {len(self.review_results)}")

def main():
    if len(sys.argv) > 1:
        input_file = sys.argv[1]
    else:
        input_file = "/Users/speters/git/personal/human-ttc-eval/data/keep/cybashbench/cybashbench_tasks.jsonl"
    
    output_file = "review.jsonl"
    
    if not Path(input_file).exists():
        print(f"{Colors.colorize('Error:', Colors.RED)} Input file '{input_file}' not found")
        sys.exit(1)
    
    reviewer = CyBashBenchReviewer(input_file, output_file)
    reviewer.review_tasks()

if __name__ == "__main__":
    main()