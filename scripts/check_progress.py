import argparse
from pathlib import Path

def check_progress(datasets_str: str, models_str: str):
    """
    Checks the progress of benchmark evaluations and reports completion status.

    Args:
        datasets_str (str): A space-separated string of dataset names.
        models_str (str): A space-separated string of model identifiers.
    """
    datasets = [d.strip() for d in datasets_str.split(' ') if d.strip()]
    models = [m.strip() for m in models_str.split(' ') if m.strip()]
    
    base_results_dir = Path("results/benchmarks")
    
    total_benchmarks = len(datasets) * len(models)
    completed_benchmarks = 0
    missing_benchmarks = []
    
    print("--- Benchmark Progress Report ---")
    print(f"Checking {total_benchmarks} total benchmarks ({len(datasets)} datasets x {len(models)} models)")
    
    for dataset in datasets:
        for model in models:
            model_path_str = model.replace('/', '_')
            dataset_dir = base_results_dir / dataset
            
            found = False
            if dataset_dir.is_dir():
                # Check for any directory that starts with the model name,
                # accounting for timestamps appended by the benchmark script.
                found = any(item.is_dir() and item.name.startswith(model_path_str) for item in dataset_dir.iterdir())
            
            if found:
                completed_benchmarks += 1
            else:
                missing_benchmarks.append((dataset, model))

    percentage_complete = (completed_benchmarks / total_benchmarks) * 100 if total_benchmarks > 0 else 0
    
    print(f"\nCompletion: {completed_benchmarks} / {total_benchmarks} ({percentage_complete:.2f}%)\n")
    
    if missing_benchmarks:
        print("Missing Benchmarks:")
        # Group missing benchmarks by dataset for readability
        missing_by_dataset = {}
        for dataset, model in missing_benchmarks:
            if dataset not in missing_by_dataset:
                missing_by_dataset[dataset] = []
            missing_by_dataset[dataset].append(model)
        
        for dataset, models in sorted(missing_by_dataset.items()):
            print(f"\n  Dataset: {dataset}")
            for model in sorted(models):
                print(f"    - {model}")
    else:
        print("🎉 All benchmarks are complete!")
    print("\n---------------------------------")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Check benchmark progress for the Human TTC Eval project.")
    parser.add_argument("--datasets", required=True, help="A space-separated string of dataset names.")
    parser.add_argument("--models", required=True, help="A space-separated string of model identifiers.")
    
    args = parser.parse_args()
    
    check_progress(args.datasets, args.models) 