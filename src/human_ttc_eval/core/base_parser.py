from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Dict, Any
import json

class BaseParser(ABC):
    """
    Abstract base class for dataset parsers.
    Each parser will transform a specific raw dataset format into
    a list of run dictionaries adhering to the all_runs.jsonl schema.
    """

    def __init__(self, input_dir: Path, output_file: Path):
        self.input_dir = input_dir
        self.output_file = output_file
        # Potentially add common CLI args like min/max duration here

    @abstractmethod
    def parse(self) -> List[Dict[str, Any]]:
        """
        Parses the raw dataset from input_dir and returns a list of
        dictionaries, each representing a run.
        """
        pass

    @property
    @abstractmethod
    def dataset_name(self) -> str:
        """
        Returns a slug-like name for the dataset (e.g., "kypo", "hcast_debug").
        """
        pass

    def write_jsonl(self, runs: List[Dict[str, Any]]) -> None:
        """
        Writes the list of run dictionaries to the specified output_file
        in JSONL format.
        """
        self.output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.output_file, 'w', encoding='utf-8') as f:
            for run_data in runs:
                # Filter out None values before writing, similar to existing parser
                filtered_run_data = {k: v for k, v in run_data.items() if v is not None}
                f.write(json.dumps(filtered_run_data) + '\n')
        print(f"Successfully wrote {len(runs)} records to {self.output_file}") # Replace with logging

# Example of how this might be used by a CLI later:
# if __name__ == '__main__':
#     # This part would be in cli.py or a top-level script
#     parser_name = sys.argv[1] # e.g., "kypo"
#     input_path = Path(sys.argv[2])
#     output_path = Path(sys.argv[3])
#
#     # Using a registry (to be defined in core/registry.py)
#     # parser_class = get_parser(parser_name)
#     # parser_instance = parser_class(input_dir=input_path, output_file=output_path)
#     # runs = parser_instance.parse()
#     # if runs:
#     #     parser_instance.write_jsonl(runs) 