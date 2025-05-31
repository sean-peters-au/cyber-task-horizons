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
    
    logger.info(f"Loading human baselines from {human_baselines_file}")
    human_baselines = {}
    with open(human_baselines_file, 'r') as f:
        for line in f:
            if line.strip():
                data = json.loads(line)
                task_path = data.get('_raw_task_path', '') # CyBench specific
                human_minutes = data.get('human_minutes', 0)
                if task_path and human_minutes > 0:
                    human_baselines[task_path] = human_minutes
    logger.info(f"Loaded {len(human_baselines)} human baseline tasks from CyBench file")
    
    nl2bash_baselines_file = Path("data/processed/nl2bash/all_tasks.jsonl")
    if nl2bash_baselines_file.exists():
        logger.info(f"Loading NL2Bash baselines from {nl2bash_baselines_file}")
        with open(nl2bash_baselines_file, 'r') as f:
            for line in f:
                if line.strip():
                    data = json.loads(line)
                    task_id = data.get('task_id', '') # NL2Bash uses task_id directly
                    human_minutes = data.get('human_minutes', 0)
                    if task_id and human_minutes > 0:
                        human_baselines[task_id] = human_minutes
        logger.info(f"Total human baselines after NL2Bash: {len(human_baselines)}")
    else:
        logger.warning(f"NL2Bash baselines file not found: {nl2bash_baselines_file}")
    
    # Load models registry
    model_details_map = {}
    try:
        with open(models_registry_file, 'r') as f:
            models_data_root = json.load(f)
        # Expecting a dict with a top-level key like "models"
        models_list = models_data_root.get('models', []) 
        if not isinstance(models_list, list):
            logger.error(f"Models registry ({models_registry_file}) is not in the expected format: root key 'models' should contain a list.")
            models_list = [] # Prevent further errors
            
        for model_entry in models_list:
            full_name = model_entry.get('full_name')
            if full_name:
                model_details_map[full_name] = {
                    'release_date': model_entry.get('release_date', '2024-01-01'),
                    'display_name': model_entry.get('display_name', full_name),
                    'metr_id': model_entry.get('metr_id', model_entry.get('display_name', full_name))
                }
    except Exception as e:
        logger.error(f"Error loading or parsing models registry {models_registry_file}: {e}. Proceeding with limited model info.")

    logger.info(f"Loaded details for {len(model_details_map)} models from registry.")
    
    runs = []
    result_files = list(results_dir.glob("**/*.json"))
    logger.info(f"Found {len(result_files)} result files")
    
    for result_file in result_files:
        try:
            with open(result_file, 'r') as f:
                result = json.load(f)
            
            if not result.get('success', True):
                logger.debug(f"Skipping failed result file: {result_file}")
                continue
                
            task_results = result.get('task_results', [])
            if not task_results:
                logger.debug(f"No task_results in {result_file}")
                continue
            
            model_full_name = result.get('model_name', '') # This is the key for model_details_map
            dataset_name = result.get('dataset_name', '')
            
            model_info = model_details_map.get(model_full_name, {})
            # agent_name_for_metr should match what logistic_fits expects as 'agent'
            agent_name_for_metr = model_info.get('metr_id', model_full_name) 
            # alias_for_plotting is for the histogram plot's 'alias' field
            alias_for_plotting = model_info.get('display_name', model_full_name)

            for task_result in task_results:
                task_path = task_result.get('task_path', '') # CyBench way
                task_name = task_result.get('task_name', '') or task_result.get('task_id', '') # CyBench or NL2Bash task_id
                
                human_minutes = human_baselines.get(task_path) # Try CyBench path first
                if human_minutes is None and task_name: # Then try task_id (NL2Bash etc)
                    human_minutes = human_baselines.get(task_name)
                    
                if human_minutes is None:
                    logger.debug(f"No human baseline for task: {task_name} (path: {task_path}) in file {result_file}")
                    continue
                
                success = task_result.get('success', False)
                run = {
                    'agent': agent_name_for_metr, # Used by logistic regression, becomes 'agent' in logistic_fits
                    'alias': alias_for_plotting,  # Expected by individual_histograms for display
                    'model': model_full_name,     # Original full name for reference
                    'task_id': task_name,
                    'score_binarized': 1 if success else 0,
                    'human_minutes': human_minutes,
                    'task_source': dataset_name,
                    'invsqrt_task_weight': 1.0, 
                    'equal_task_weight': 1.0,
                    'weight': 1.0, # Default weight, METR uses this as 'weighting' param
                }
                runs.append(run)
        except Exception as e:
            logger.warning(f"Failed to process {result_file}: {e}", exc_info=True)
    
    logger.info(f"Generated {len(runs)} runs for METR format.")
    
    # Pre-save check for agent field
    for i, run_entry in enumerate(runs):
        if not isinstance(run_entry.get('agent'), str):
            logger.warning(f"Run at index {i} has a non-string agent field: {run_entry.get('agent')} (type: {type(run_entry.get('agent'))}). Full record: {run_entry}")
        if not run_entry.get('agent'): # Check for empty string agent names
             logger.warning(f"Run at index {i} has an empty string agent field. Full record: {run_entry}")

    runs_file = output_dir / "all_runs.jsonl"
    with open(runs_file, 'w') as f:
        for run_entry in runs:
            f.write(json.dumps(run_entry) + '\n')
    
    # Create release_dates.yaml using the agent_name_for_metr (metr_id or display_name)
    unique_metr_agents = set(r['agent'] for r in runs)
    release_dates_data = {
        "date": {
            metr_agent_name: model_details_map.get(next((fn for fn, dt in model_details_map.items() if dt.get('metr_id', dt.get('display_name', fn)) == metr_agent_name), metr_agent_name), {}).get('release_date', "2024-01-01")
            for metr_agent_name in unique_metr_agents
        }
    }
    
    dates_file = output_dir / "release_dates.yaml"
    with open(dates_file, 'w') as f:
        yaml.dump(release_dates_data, f)
    
    logger.info(f"Saved {len(runs)} runs to {runs_file}")
    logger.info(f"Saved release dates for {len(unique_metr_agents)} METR agents to {dates_file}")
    
    return {
        'runs_file': runs_file,
        'release_dates_file': dates_file
    } 