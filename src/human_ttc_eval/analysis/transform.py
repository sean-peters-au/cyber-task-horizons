"""
Transform benchmark results to METR's all_runs.jsonl format.

Joins our AI benchmark results with human baseline data by task.
"""

import json
import yaml
from pathlib import Path
from typing import Dict, List, Any
import logging

from ..core.run import Run
from ..core.task import Task
from .. import config

logger = logging.getLogger(__name__)


def transform_benchmark_results(output_dir: Path) -> Dict[str, Path]:
    """
    Transform all benchmark results to METR format.
    
    Reads from standard locations:
    - Benchmark results: config.RESULTS_DIR / "benchmarks"
    - Human baselines: config.DATA_DIR / "processed" / <dataset> / <dataset>_tasks.jsonl
    - Models registry: src/human_ttc_eval/models.json
    
    Args:
        output_dir: Where to save output files
        
    Returns:
        Dict with paths to generated files
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load models registry
    models_registry_file = Path(__file__).parent.parent / "models.json"
    model_details_map = _load_models_registry(models_registry_file)
    
    # Load all human baselines from processed data
    human_baselines = _load_all_human_baselines()
    logger.info(f"Loaded {len(human_baselines)} total human baseline tasks")
    
    # Process all benchmark results
    runs = []
    benchmarks_dir = config.RESULTS_DIR / "benchmarks"
    
    for dataset_dir in benchmarks_dir.iterdir():
        if not dataset_dir.is_dir():
            continue
            
        dataset_name = dataset_dir.name
        logger.info(f"Processing {dataset_name} benchmark results...")
        
        for result_file in dataset_dir.glob("**/*.json"):
            try:
                runs.extend(_process_benchmark_file(
                    result_file, 
                    human_baselines, 
                    model_details_map,
                    dataset_name
                ))
            except Exception as e:
                logger.warning(f"Failed to process {result_file}: {e}", exc_info=True)
    
    logger.info(f"Generated {len(runs)} runs for METR format")
    
    # Convert to Run objects to calculate proper weights
    run_objects = []
    for run_dict in runs:
        # Create a minimal Run object with required fields
        run_obj = Run(
            task_id=run_dict['task_id'],
            task_family=run_dict.get('task_family', run_dict['task_source']),
            run_id=run_dict.get('run_id', f"{run_dict['model']}_{run_dict['task_id']}"),
            alias=run_dict['alias'],
            model=run_dict['model'],
            score_binarized=run_dict['score_binarized'],
            human_minutes=run_dict['human_minutes']
        )
        run_objects.append(run_obj)
    
    # Calculate proper weights
    Run.calculate_weights(run_objects)
    
    # Update our run dicts with calculated weights
    for run_dict, run_obj in zip(runs, run_objects):
        run_dict['equal_task_weight'] = run_obj.equal_task_weight
        run_dict['invsqrt_task_weight'] = run_obj.invsqrt_task_weight
        # METR expects 'weight' - use equal_task_weight by default
        run_dict['weight'] = run_obj.equal_task_weight
    
    # Save runs
    runs_file = output_dir / "all_runs.jsonl"
    with open(runs_file, 'w') as f:
        for run in runs:
            f.write(json.dumps(run) + '\n')
    
    # Create release_dates.yaml
    unique_agents = set(r['agent'] for r in runs)
    release_dates_data = _create_release_dates(unique_agents, model_details_map)
    
    dates_file = output_dir / "release_dates.yaml"
    with open(dates_file, 'w') as f:
        yaml.dump(release_dates_data, f)
    
    logger.info(f"Saved {len(runs)} runs to {runs_file}")
    logger.info(f"Saved release dates for {len(unique_agents)} agents to {dates_file}")
    
    return {
        'runs_file': runs_file,
        'release_dates_file': dates_file
    }


def _load_models_registry(models_registry_file: Path) -> Dict[str, Dict[str, Any]]:
    """Load and parse the models registry."""
    model_details_map = {}
    
    try:
        with open(models_registry_file, 'r') as f:
            models_data = json.load(f)
        
        models_list = models_data.get('models', [])
        if not isinstance(models_list, list):
            logger.error(f"Models registry format error: 'models' should be a list")
            return model_details_map
            
        for model_entry in models_list:
            full_name = model_entry.get('full_name')
            if full_name:
                model_details_map[full_name] = {
                    'release_date': model_entry.get('release_date', '2024-01-01'),
                    'display_name': model_entry.get('display_name', full_name),
                    'metr_id': model_entry.get('metr_id', model_entry.get('display_name', full_name))
                }
                
    except Exception as e:
        logger.error(f"Error loading models registry: {e}")
        
    return model_details_map


def _load_all_human_baselines() -> Dict[str, float]:
    """Load human baselines from all processed datasets."""
    human_baselines = {}
    processed_dir = config.DATA_DIR / "processed"
    
    for dataset_dir in processed_dir.iterdir():
        if not dataset_dir.is_dir():
            continue
            
        # Try loading tasks file
        tasks_file = dataset_dir / f"{dataset_dir.name}_tasks.jsonl"
        if tasks_file.exists():
            logger.info(f"Loading tasks from {tasks_file}")
            tasks = Task.load_jsonl(tasks_file)
            
            for task in tasks:
                # Store by task_id
                if task.human_minutes > 0:
                    human_baselines[task.task_id] = task.human_minutes
                    
                # Also store by any dataset-specific identifiers
                if 'task_path' in task.dataset_task_metadata:
                    task_path = task.dataset_task_metadata['task_path']
                    if task_path:
                        human_baselines[task_path] = task.human_minutes
    
    return human_baselines


def _process_benchmark_file(
    result_file: Path,
    human_baselines: Dict[str, float],
    model_details_map: Dict[str, Dict[str, Any]],
    dataset_name: str
) -> List[Dict[str, Any]]:
    """Process a single benchmark result file."""
    runs = []
    
    with open(result_file, 'r') as f:
        result = json.load(f)
    
    if not result.get('success', True):
        logger.debug(f"Skipping failed result file: {result_file}")
        return runs
        
    # Check for both 'runs' and 'task_results' fields for compatibility
    task_results = result.get('runs') or result.get('task_results', [])
    if not task_results:
        logger.debug(f"No runs/task_results in {result_file}")
        return runs
    
    model_full_name = result.get('model_name', '')
    model_info = model_details_map.get(model_full_name, {})
    
    # METR expects 'agent' field for logistic regression
    agent_name_for_metr = model_info.get('metr_id', model_full_name)
    # 'alias' is for display in plots
    alias_for_plotting = model_info.get('display_name', model_full_name)
    
    for task_result in task_results:
        # For runs that are already in METR format, just update agent/alias
        if 'score_binarized' in task_result and 'human_minutes' in task_result:
            # This is already a properly formatted run
            run = task_result.copy()
            run['agent'] = agent_name_for_metr
            run['alias'] = alias_for_plotting
            run['model'] = model_full_name
            run['task_source'] = dataset_name
            
            # METR expects 'weight' field - use equal_task_weight by default
            # This is what METR uses for the 'weighting' parameter
            if 'weight' not in run:
                run['weight'] = run.get('equal_task_weight', 1.0)
                
            runs.append(run)
        else:
            # Legacy format - convert to METR format
            task_id = task_result.get('task_id') or task_result.get('task_name', '')
            task_path = task_result.get('task_path', '')
            
            # Look up human baseline
            human_minutes = human_baselines.get(task_id)
            if human_minutes is None and task_path:
                human_minutes = human_baselines.get(task_path)
                
            if human_minutes is None:
                logger.debug(f"No human baseline for task: {task_id} (path: {task_path})")
                continue
            
            success = task_result.get('success', False)
            
            # Create METR-format run
            # Note: We include both our schema fields and METR's expected 'weight' field
            run = {
                'agent': agent_name_for_metr,
                'alias': alias_for_plotting,
                'model': model_full_name,
                'task_id': task_id,
                'score_binarized': 1 if success else 0,
                'human_minutes': human_minutes,
                'task_source': dataset_name,
                'invsqrt_task_weight': 1.0,  # Our schema
                'equal_task_weight': 1.0,     # Our schema
                'weight': 1.0,  # METR's expected field for weighting parameter
            }
            runs.append(run)
    
    return runs


def _create_release_dates(
    unique_agents: set,
    model_details_map: Dict[str, Dict[str, Any]]
) -> Dict[str, Dict[str, str]]:
    """Create release dates mapping for METR format."""
    release_dates = {}
    
    for agent in unique_agents:
        # Find the model entry that maps to this agent
        release_date = "2024-01-01"  # default
        
        for full_name, details in model_details_map.items():
            if details.get('metr_id') == agent:
                release_date = details.get('release_date', "2024-01-01")
                break
        
        release_dates[agent] = release_date
    
    return {"date": release_dates} 