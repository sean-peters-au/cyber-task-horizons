"""
Transform benchmark results to METR's all_runs.jsonl format.

Simple approach: join our AI results with human baseline data by task path.
"""

import json
import yaml
from pathlib import Path
from typing import List, Dict, Any
import logging

logger = logging.getLogger(__name__)


def transform_benchmark_results(
    results_dir: Path,
    human_baselines_file: Path,
    models_registry_file: Path,
    output_dir: Path
) -> Dict[str, Path]:
    """
    Transform benchmark results to METR format.
    
    Args:
        results_dir: Directory with our benchmark JSON files
        human_baselines_file: Path to cybench_human_runs.jsonl
        models_registry_file: Path to models.json
        output_dir: Where to save output files
        
    Returns:
        Dict with paths to generated files
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load human baselines
    logger.info(f"Loading human baselines from {human_baselines_file}")
    human_baselines = {}
    with open(human_baselines_file, 'r') as f:
        for line in f:
            if line.strip():
                data = json.loads(line)
                task_path = data.get('_raw_task_path', '')
                human_minutes = data.get('human_minutes', 0)
                if task_path and human_minutes > 0:
                    human_baselines[task_path] = human_minutes
    
    logger.info(f"Loaded {len(human_baselines)} human baseline tasks")
    
    # Also load NL2Bash human baselines
    nl2bash_baselines_file = Path("data/processed/nl2bash/all_tasks.jsonl")
    if nl2bash_baselines_file.exists():
        logger.info(f"Loading NL2Bash baselines from {nl2bash_baselines_file}")
        with open(nl2bash_baselines_file, 'r') as f:
            for line in f:
                if line.strip():
                    data = json.loads(line)
                    task_id = data.get('task_id', '')
                    human_minutes = data.get('human_minutes', 0)
                    if task_id and human_minutes > 0:
                        human_baselines[task_id] = human_minutes
        logger.info(f"Total human baselines: {len(human_baselines)}")
    else:
        logger.warning(f"NL2Bash baselines file not found: {nl2bash_baselines_file}")
    
    # Load models registry for release dates
    with open(models_registry_file, 'r') as f:
        models_data = json.load(f)
    
    model_dates = {}
    for model in models_data.get('models', []):
        full_name = model.get('full_name', '')
        release_date = model.get('release_date', '')
        if full_name and release_date:
            model_dates[full_name] = release_date
    
    logger.info(f"Loaded {len(model_dates)} model release dates")
    
    # Process benchmark result files
    runs = []
    result_files = list(results_dir.glob("**/*.json"))
    logger.info(f"Found {len(result_files)} result files")
    
    for result_file in result_files:
        try:
            with open(result_file, 'r') as f:
                result = json.load(f)
            
            # Skip failed results
            if not result.get('success', True):
                continue
                
            task_results = result.get('task_results', [])
            if not task_results:
                continue
            
            model_name = result.get('model_name', '')
            dataset_name = result.get('dataset_name', '')
            
            # Transform each task result to METR format
            for task_result in task_results:
                task_path = task_result.get('task_path', '')
                task_name = task_result.get('task_name', '') or task_result.get('task_id', '')
                success = task_result.get('success', False)
                
                # Join with human baseline - try both task_path (CyBench) and task_id (NL2Bash)
                human_minutes = human_baselines.get(task_path)
                if human_minutes is None and task_name:
                    human_minutes = human_baselines.get(task_name)
                    
                if human_minutes is None:
                    logger.debug(f"No human baseline for task: {task_name} (path: {task_path})")
                    continue
                
                # Create METR run entry
                run = {
                    'agent': model_name,
                    'task_id': task_name,
                    'score_binarized': 1 if success else 0,
                    'human_minutes': human_minutes,
                    'task_source': dataset_name,
                    'invsqrt_task_weight': 1.0,  # Equal weighting
                    'equal_task_weight': 1.0,
                    'weight': 1.0,
                }
                
                runs.append(run)
                
        except Exception as e:
            logger.warning(f"Failed to process {result_file}: {e}")
    
    logger.info(f"Generated {len(runs)} runs")
    
    # Save all_runs.jsonl
    runs_file = output_dir / "all_runs.jsonl"
    with open(runs_file, 'w') as f:
        for run in runs:
            f.write(json.dumps(run) + '\n')
    
    # Create release_dates.yaml
    agents = set(run['agent'] for run in runs)
    release_dates = {
        "date": {
            agent: model_dates.get(agent, "2024-01-01") 
            for agent in agents
        }
    }
    
    dates_file = output_dir / "release_dates.yaml"
    with open(dates_file, 'w') as f:
        yaml.dump(release_dates, f)
    
    logger.info(f"Saved {len(runs)} runs to {runs_file}")
    logger.info(f"Saved release dates for {len(agents)} agents to {dates_file}")
    
    return {
        'runs_file': runs_file,
        'release_dates_file': dates_file
    } 