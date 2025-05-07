import json
import logging
import re
from dataclasses import dataclass, fields, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

from ...core.base_parser import BaseParser
from ...core.registry import register_parser
from ...core.utils import slugify

# Configure logging for this module - might want a more centralized setup later
logger = logging.getLogger(__name__)

# --- Constants specific to KYPO parsing ---
# These were moved from the old parse_datasets.py
MIN_SESSION_DURATION_SECONDS = 5 * 60  # 5 minutes
MAX_SESSION_DURATION_SECONDS = 3 * 60 * 60  # 3 hours
KYPO_TIME_LIMIT_MINUTES = 120.0

SUCCESS_COMMAND_SUBSTRINGS = [
    "check_flag.sh",
    "cat /root/flag.txt",
    "cat flag.txt",
    "cat flag",
    "get-flag",
    "submit-flag",
]
SUCCESS_LAST_COMMAND_TOKENS = [
    "echo FLAG",
    "submit_flag",
]

# --- Data Structures ---
# These are largely the same as in the original parse_datasets.py
@dataclass
class CommandLog:
    timestamp_str: str
    command_str: str
    timestamp: Optional[datetime] = None

    def __post_init__(self):
        try:
            ts_to_parse = self.timestamp_str
            if not ts_to_parse:
                self.timestamp = None
                logger.warning("Empty timestamp string found. Command will be skipped.")
                return

            if ts_to_parse.endswith('Z'):
                normalized_ts_str = ts_to_parse[:-1] + ".000000+00:00"
                parts = ts_to_parse[:-1].split('.')
                if len(parts) == 2:
                    integer_part, fractional_part = parts
                    fractional_part = fractional_part.ljust(6, '0')[:6]
                    normalized_ts_str = f"{integer_part}.{fractional_part}+00:00"
                else:
                    normalized_ts_str = f"{parts[0]}.000000+00:00"
                ts_to_parse = normalized_ts_str
            
            match_subsecond = re.match(r'^(.*\.)(\d{1,5})([Z+-].*)$', ts_to_parse)
            if match_subsecond:
                prefix, frac, suffix = match_subsecond.groups()
                frac = frac.ljust(6, '0')
                ts_to_parse = f"{prefix}{frac}{suffix}"
            
            if 'Z' in ts_to_parse and not ts_to_parse.endswith("+00:00"):
                 ts_to_parse = ts_to_parse.replace('Z', '+00:00')

            dt_obj = datetime.fromisoformat(ts_to_parse)
            
            if dt_obj.tzinfo is None:
                dt_obj = dt_obj.replace(tzinfo=timezone.utc)
            else:
                dt_obj = dt_obj.astimezone(timezone.utc)
            self.timestamp = dt_obj
            
        except (ValueError, TypeError) as e:
            ts_str_for_log = ts_to_parse if 'ts_to_parse' in locals() else self.timestamp_str
            logger.warning(f'Could not parse timestamp "{self.timestamp_str}" (normalized to "{ts_str_for_log}"): {e}. Command will skip.')
            self.timestamp = None

@dataclass
class HumanKypoRun: # This mirrors the structure from the original script
    task_family: Optional[str] = None
    task_id: Optional[str] = None
    run_id: Optional[str] = None
    alias: str = "Human (KYPO)"
    model: str = "human"
    score_binarized: Optional[int] = None
    score_cont: Optional[float] = None
    fatal_error_from: Optional[str] = None
    human_minutes: Optional[float] = None
    human_score: Optional[float] = None
    human_source: str = "kypo_cybersecurity_logs_v4"
    task_source: str = "kypo_cybersecurity_dataset"
    generation_cost: float = 0.0
    human_cost: Optional[float] = None
    time_limit: Optional[float] = KYPO_TIME_LIMIT_MINUTES
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    task_version: Optional[str] = "1.0"
    equal_task_weight: Optional[float] = None
    invsqrt_task_weight: Optional[float] = None
    command_count: int = 0
    _raw_training_name: Optional[str] = None
    _raw_session_group: Optional[str] = None
    _raw_sandbox_id: Optional[str] = None
    _raw_file_path: Optional[str] = None

# --- Helper Functions specific to KYPO parsing ---
def extract_sandbox_id_from_filename(filename: str) -> Optional[str]:
    if filename.endswith("-useractions.json") and filename.startswith("sandbox-"):
        parts = filename[len("sandbox-"):-len("-useractions.json")]
        if parts.isalnum():
            return parts
    return None

def _parse_single_session_log(log_file_path: Path, raw_training_name: str, raw_session_group: str, data_root_dir: Path) -> HumanKypoRun:
    raw_sandbox_id = extract_sandbox_id_from_filename(log_file_path.name)
    
    run_result = HumanKypoRun(
        _raw_training_name=raw_training_name,
        _raw_session_group=raw_session_group,
        _raw_sandbox_id=raw_sandbox_id if raw_sandbox_id else "UNKNOWN_ID",
        _raw_file_path=str(log_file_path.relative_to(data_root_dir))
    )

    if not raw_sandbox_id:
        msg = f"Could not extract valid sandbox_id from filename: {log_file_path.name}"
        logger.warning(msg)
        run_result.fatal_error_from = msg
        return run_result

    run_result.task_family = slugify(raw_training_name)
    run_result.task_id = f"{run_result.task_family}/main"
    run_result.run_id = f"human_kypo_{run_result.task_family}_main_sandbox{raw_sandbox_id}"

    try:
        raw_log_entries_data = []
        with open(log_file_path, 'r', encoding='utf-8') as f:
            for line_number, line in enumerate(f, 1):
                line = line.strip()
                if not line: continue
                try:
                    raw_log_entries_data.append(json.loads(line))
                except json.JSONDecodeError as e_line:
                    logger.warning(f"JSONDecodeError on line {line_number} in {log_file_path}: {e_line}. Skipping line.")

        if not raw_log_entries_data:
            run_result.fatal_error_from = "No valid JSON objects found (expected JSONL)."
            logger.warning(f"{run_result.fatal_error_from} - {log_file_path}")
            return run_result

        command_logs: List[CommandLog] = []
        for i, entry in enumerate(raw_log_entries_data):
            if not isinstance(entry, dict):
                logger.warning(f"Entry #{i} in {log_file_path.name} is not a dict, skipping.")
                continue
            
            ts_str = entry.get("timestamp_str")
            cmd_str = entry.get("cmd")

            if ts_str is None or cmd_str is None:
                logger.warning(f"Entry #{i} in {log_file_path.name} missing 'timestamp_str' or 'cmd', skipping.")
                continue
            
            cmd_log_entry = CommandLog(timestamp_str=ts_str, command_str=cmd_str)
            if cmd_log_entry.timestamp:
                command_logs.append(cmd_log_entry)
            else:
                logger.debug(f"Skipping command due to unparsable timestamp in {log_file_path.name}: {ts_str}")

        if not command_logs:
            run_result.fatal_error_from = "No valid command logs with parsable timestamps."
            logger.warning(f"{run_result.fatal_error_from} - {log_file_path}")
            return run_result
            
        command_logs.sort(key=lambda log: log.timestamp)
        run_result.command_count = len(command_logs)

        min_ts = command_logs[0].timestamp
        max_ts = command_logs[-1].timestamp
        
        run_result.started_at = min_ts.isoformat()
        run_result.completed_at = max_ts.isoformat()
        
        duration_seconds = (max_ts - min_ts).total_seconds()
        run_result.human_minutes = duration_seconds / 60.0

        is_successful_bool = False
        for cmd_log in command_logs:
            command_lower = cmd_log.command_str.lower()
            if any(substring.lower() in command_lower for substring in SUCCESS_COMMAND_SUBSTRINGS):
                is_successful_bool = True
                logger.info(f"Success (substring): '{cmd_log.command_str}' in {run_result._raw_file_path}")
                break
            if re.search(r"flag\{[a-zA-Z0-9_@.-]+\}", cmd_log.command_str, re.IGNORECASE):
                is_successful_bool = True
                logger.info(f"Success (FLAG{{...}} pattern): '{cmd_log.command_str}' in {run_result._raw_file_path}")
                break

        if not is_successful_bool and command_logs:
            last_command_text_lower = command_logs[-1].command_str.lower()
            if any(token.lower() in last_command_text_lower for token in SUCCESS_LAST_COMMAND_TOKENS):
                is_successful_bool = True
                logger.info(f"Success (last command token): '{command_logs[-1].command_str}' in {run_result._raw_file_path}")
        
        run_result.score_binarized = 1 if is_successful_bool else 0
        run_result.score_cont = float(run_result.score_binarized)
        run_result.human_score = run_result.score_cont
        
    except Exception as e: # Broader catch, was FileNotFoundError, json.JSONDecodeError before
        run_result.fatal_error_from = f"Processing error: {str(e)}"
        logger.error(f"Error processing {log_file_path}: {e}", exc_info=True)
        
    return run_result

@register_parser("kypo")
class KypoParser(BaseParser):
    """Parses KYPO cyber range logs into the common all_runs.jsonl format."""

    @property
    def dataset_name(self) -> str:
        return "kypo"

    def __init__(self, input_dir: Path, output_file: Path, data_root_for_relative_paths: Optional[Path] = None):
        super().__init__(input_dir, output_file)
        # DATA_ROOT_DIR from old script was used for Path.relative_to()
        # The parser needs this to construct _raw_file_path correctly.
        # It should be the original DATA_ROOT_DIR ('data/cybersecurity_dataset_v4')
        self.data_root_for_relative_paths = data_root_for_relative_paths if data_root_for_relative_paths else self.input_dir
        logger.info(f"KypoParser initialized. Input: {self.input_dir}, Output: {self.output_file}, Data Root for Relative: {self.data_root_for_relative_paths}")


    def parse(self) -> List[Dict[str, Any]]:
        """Main parsing logic adapted from parse_datasets.py's main().
        Uses self.input_dir (which is the DATA_ROOT_DIR from the old script)."""
        if not self.input_dir.exists() or not self.input_dir.is_dir():
            logger.error(f"Data root directory '{self.input_dir}' not found or not a directory.")
            return []

        all_parsed_runs_objects: List[HumanKypoRun] = []
        
        # input_dir here corresponds to the old DATA_ROOT_DIR
        json_log_files = list(self.input_dir.rglob("sandbox-*-useractions.json"))
        logger.info(f"Found {len(json_log_files)} potential session log files in '{self.input_dir}'.")

        if not json_log_files:
            logger.warning(f"No session log files found in {self.input_dir}. Returning empty list.")
            return []

        for log_file_path in json_log_files:
            try:
                # Path.relative_to needs a base. If input_dir is deep, this might be wrong.
                # The old script used DATA_ROOT_DIR. Here, self.input_dir serves that role
                # for finding files, but _raw_file_path needs to be relative to the dataset's true root.
                relative_path = log_file_path.relative_to(self.data_root_for_relative_paths)
                relative_path_parts = relative_path.parts
                
                raw_training_name = relative_path_parts[0]
                raw_session_group = "N/A" # Default
                
                if len(relative_path_parts) > 2: # e.g. training_name/session_group/sandbox-file.json
                    raw_session_group = relative_path_parts[1]
                elif len(relative_path_parts) < 2: # Should be at least training_name/sandbox-file.json
                    logger.warning(f"Unexpected file path structure for {log_file_path} relative to {self.data_root_for_relative_paths}. Skipping.")
                    error_run = HumanKypoRun(
                        _raw_file_path=str(log_file_path), # Store absolute if relative fails
                        fatal_error_from=f"Path processing error: Unexpected structure '{relative_path}'"
                    )
                    all_parsed_runs_objects.append(error_run)
                    continue
                    
                logger.debug(f"Processing: {relative_path} (Training: {raw_training_name}, Group: {raw_session_group})")
                # Pass self.data_root_for_relative_paths so _parse_single_session_log can make _raw_file_path relative to it
                session_data_obj = _parse_single_session_log(log_file_path, raw_training_name, raw_session_group, self.data_root_for_relative_paths)
                all_parsed_runs_objects.append(session_data_obj)
            except Exception as e:
                logger.error(f"Critical error processing path for {log_file_path}: {e}", exc_info=True)
                error_run = HumanKypoRun(
                    _raw_file_path=str(log_file_path),
                    fatal_error_from=f"Critical path processing error: {str(e)}"
                )
                all_parsed_runs_objects.append(error_run)

        # Filter sessions based on duration criteria
        final_runs_for_output: List[Dict[str, Any]] = []
        for run_obj in all_parsed_runs_objects:
            # Convert to dict, removing None and internal fields, similar to original script's final write loop
            output_dict = {}
            for field_info in fields(HumanKypoRun): # Iterate over fields of HumanKypoRun dataclass
                if not field_info.name.startswith('_'): # Exclude internal fields
                    value = getattr(run_obj, field_info.name)
                    # We will filter Nones in the BaseParser.write_jsonl, but fatal_error_from might need special handling
                    # For now, include all non-underscore fields.
                    output_dict[field_info.name] = value

            if run_obj.fatal_error_from:
                logger.warning(f"Run from {run_obj._raw_file_path or 'unknown file'} has fatal error: {run_obj.fatal_error_from}")
                final_runs_for_output.append(output_dict) # Still include, error is noted
            elif run_obj.human_minutes is None:
                output_dict["fatal_error_from"] = (output_dict.get("fatal_error_from", "") + " Duration (human_minutes) could not be calculated.").strip()
                final_runs_for_output.append(output_dict)
                logger.warning(f"Session for raw file {run_obj._raw_file_path} had None human_minutes without explicit prior fatal_error_from.")
            elif MIN_SESSION_DURATION_SECONDS <= (run_obj.human_minutes * 60) <= MAX_SESSION_DURATION_SECONDS:
                final_runs_for_output.append(output_dict)
            else:
                logger.info(
                    f"Excluding session from {run_obj._raw_file_path} due to duration: "
                    f"{run_obj.human_minutes * 60:.2f}s. Not within "
                    f"[{MIN_SESSION_DURATION_SECONDS}s - {MAX_SESSION_DURATION_SECONDS}s] range."
                )
                # These are simply excluded, not marked with fatal_error

        logger.info(f"Total sessions processed: {len(all_parsed_runs_objects)}. "
                     f"Sessions to be returned for JSONL: {len(final_runs_for_output)}.")
        
        return final_runs_for_output

# To make this runnable for testing (though CLI will be the primary way):
# if __name__ == '__main__':
#     # Example: python -m human_ttc_eval.datasets.kypo.parser <path_to_data_root> <output_jsonl_file>
#     import sys
#     if len(sys.argv) < 4:
#         print("Usage: python -m human_ttc_eval.datasets.kypo.parser <data_root_dir> <output_jsonl_file> <data_root_for_relative_paths>")
#         print("Example: python -m human_ttc_eval.datasets.kypo.parser data/cybersecurity_dataset_v4 data/kypo_parsed.jsonl data/cybersecurity_dataset_v4")
#         sys.exit(1)
    
#     DATA_ROOT = Path(sys.argv[1])
#     OUTPUT_FILE = Path(sys.argv[2])
#     DATA_ROOT_RELATIVE = Path(sys.argv[3])

#     # Setup basic logging for standalone script run
#     logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(module)s - %(message)s')
    
#     parser = KypoParser(input_dir=DATA_ROOT, output_file=OUTPUT_FILE, data_root_for_relative_paths=DATA_ROOT_RELATIVE)
#     runs_data = parser.parse()
#     if runs_data:
#         parser.write_jsonl(runs_data) # BaseParser provides this
#     else:
#         logger.info("No data was parsed or returned.") 