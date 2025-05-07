from abc import ABC, abstractmethod
from pathlib import Path
import pandas as pd # Requires pandas to be installed
from typing import Optional
import json # Added import for json

class BaseSummariser(ABC):
    """
    Abstract base class for dataset summarisers.
    Each summariser will take a path to an all_runs.jsonl file (or equivalent)
    and produce summary statistics and plots.
    """

    def __init__(self, jsonl_file_path: Path, output_dir: Path):
        self.jsonl_file_path = jsonl_file_path
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.df: Optional[pd.DataFrame] = None # Initialize df as None

    def load_data(self) -> None:
        """Loads the JSONL file into a pandas DataFrame."""
        records = []
        with open(self.jsonl_file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError as e:
                        # Replace with logging
                        print(f"Skipping malformed JSON line: {line.strip()} - Error: {e}") 
        
        if not records:
            print(f"No valid records found in {self.jsonl_file_path}. DataFrame will be empty.") # Replace with logging
            self.df = pd.DataFrame()
        else:
            self.df = pd.DataFrame(records)
            print(f"Loaded {len(self.df)} records from {self.jsonl_file_path}.") # Replace with logging
            # Basic type coercion, can be overridden by subclasses
            numeric_cols = ['score_binarized', 'score_cont', 'human_minutes', 'human_score', 'command_count']
            for col in numeric_cols:
                if col in self.df.columns:
                    self.df[col] = pd.to_numeric(self.df[col], errors='coerce')

    @abstractmethod
    def summarise(self) -> None:
        """
        Generates summary statistics from the loaded DataFrame.
        Subclasses should implement this to save specific stats (e.g., CSVs).
        """
        if self.df is None or self.df.empty:
            print("DataFrame not loaded or empty. Skipping summary generation.") # Replace with logging
            return
        pass

    def save_plots(self) -> None:
        """
        Optional: Generates and saves plots from the loaded DataFrame.
        Subclasses can implement this to create dataset-specific visualizations.
        By default, does nothing.
        """
        if self.df is None or self.df.empty:
            print("DataFrame not loaded or empty. Skipping plot generation.") # Replace with logging
            return
        pass

# Example of how this might be used by a CLI later:
# if __name__ == '__main__':
#     # This part would be in cli.py or a top-level script
#     summariser_name = sys.argv[1] # e.g., "kypo"
#     jsonl_path = Path(sys.argv[2])
#     output_dir_path = Path(sys.argv[3])
#
#     # Using a registry (to be defined in core/registry.py)
#     # summariser_class = get_summariser(summariser_name)
#     # summariser_instance = summariser_class(jsonl_file_path=jsonl_path, output_dir=output_dir_path)
#     # summariser_instance.load_data()
#     # summariser_instance.summarise()
#     # summariser_instance.save_plots() # Optional 