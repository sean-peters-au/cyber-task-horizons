# repro_test.py
import re
import textwrap
from pathlib import Path

from inspect_ai import Task, task
from inspect_ai.dataset import Sample, MemoryDataset
from inspect_ai.scorer import Score, Scorer, Target, scorer
from inspect_ai.solver import Solver
from inspect_ai.agent import react
from inspect_ai.tool import ToolError, bash_session, text_editor
from inspect_ai.util import sandbox

# --- Agent Definition ---
# Placeholder prompt - needs refinement based on modular-public's actual prompt
# Use escaped quotes for the multiline string
SYSTEM_PROMPT = """
You are a helpful assistant tasked with debugging Python code.
Your goal is to make the pytest tests pass for the provided library.
The library files are located in /home/agent/app.
Run `pytest /home/agent/app/tests.py` and fix any errors you encounter in the library files (e.g., main.py, markdown_converter.py).
Do NOT edit the tests.py file itself.
Reason step-by-step about your plan. Use the available tools (bash_session, text_editor) to inspect files, run tests, and edit code.
When all tests pass, use the submit() tool.
"""

# Define the agent using inspect_ai's react agent
paper_agent: Solver = react(
    prompt=SYSTEM_PROMPT,
    tools=[
        bash_session(),
        text_editor()
    ],
)


# --- Task Constants ---

# Path to the local clone of hcast-public repo
# Ensure this path is correct relative to where 'inspect eval' is run
HCAST_PUBLIC_DIR = Path("./hcast-public").resolve()

TASK_FAMILY = "debug_small_libs"
TASK_VARIANT = "markdown"
APP_PATH_IN_SANDBOX = "/home/agent/app"

# --- Custom Scorer Definition ---
@scorer(metrics=[])
def pytest_scorer() -> Scorer:
    """
    Runs pytest in the sandbox and scores based on pass rate.
    Also checks if tests.py was modified.
    """
    async def score(state, target: Target) -> Score:
        # Use the sandbox() utility to get the current sandbox environment
        try:
            sandbox_env = sandbox()
        except RuntimeError as e:
            return Score(value=0.0, passed=False, explanation=f"Error getting sandbox: {e}. Scorer likely called outside sandbox context.")

        # --- Check 1: tests.py modification ---
        # Path to original test file on the host machine running 'inspect eval'
        original_test_path = HCAST_PUBLIC_DIR / TASK_FAMILY / "assets" / TASK_VARIANT / "tests.py"
        try:
            original_test_content = original_test_path.read_text()
        except FileNotFoundError:
             # Fail score if we can't find the original test file to compare against
             return Score(value=0.0, passed=False, explanation=f"Original test file not found at {original_test_path}")

        # Path to the test file inside the sandbox
        current_test_path_in_sandbox = f"{APP_PATH_IN_SANDBOX}/tests.py"
        try:
            # Use sandbox.read_file() which is designed for this
            current_test_content = await sandbox_env.read_file(current_test_path_in_sandbox)
            if not isinstance(current_test_content, str):
                 # Handle potential bytes output if text=False was used, though default is True
                 current_test_content = current_test_content.decode('utf-8')

        except (ToolError, FileNotFoundError) as e:
            # Fail score if we can't read the test file in the sandbox
            return Score(value=0.0, passed=False, explanation=f"Error reading {current_test_path_in_sandbox} in sandbox: {e}")

        # Compare content
        if current_test_content != original_test_content:
            return Score(value=0.0, passed=False, explanation="Agent modified the tests.py file.")

        # --- Check 2: Run pytest ---
        try:
            # Assuming pytest is available in the sandbox environment's PATH
            # The command runs relative to the sandbox's default working dir,
            # but we specify the test file path explicitly for clarity.
            # We might need to install pytest in the sandbox image/setup.
            pytest_command = ["pytest", "-q", "--tb=no", f"{APP_PATH_IN_SANDBOX}/tests.py"]
            # Check if pytest exists - this is a basic check, might need refinement
            check_pytest = await sandbox_env.exec(["which", "pytest"])
            if not check_pytest.success:
                # Attempt to install pytest if not found
                install_result = await sandbox_env.exec(["pip", "install", "pytest"])
                if not install_result.success:
                    return Score(value=0.0, passed=False, explanation=f"pytest not found and installation failed: {install_result.stderr}")
                # Re-check after install attempt (optional, but good practice)
                check_pytest = await sandbox_env.exec(["which", "pytest"])
                if not check_pytest.success:
                     return Score(value=0.0, passed=False, explanation="pytest installed but still not found in PATH?")

            result = await sandbox_env.exec(pytest_command) # Use exec which returns ExecResult

        except ToolError as e:
             # Fail score if pytest command fails unexpectedly
             return Score(value=0.0, passed=False, explanation=f"Error running pytest in sandbox: {e}")

        # --- Check 3: Parse pytest output ---
        output_lines = result.stdout.strip().split("\n")
        if not output_lines:
             # Fail score if pytest gives no output
             return Score(value=0.0, passed=False, explanation="Pytest produced no output.")

        summary_line = output_lines[-1]
        # Regex to find passed/failed/skipped counts in the summary line
        match = re.search(
            r"(?:(\d+)\s+failed,?\s*)?(?:(\d+)\s+passed,?\s*)?(?:(\d+)\s+skipped)?",
            summary_line
        )

        if not match:
            # Fail score if output format is unexpected
            return Score(value=0.0, passed=False, explanation=f"Could not parse pytest summary: {summary_line}", metrics={"pytest_output": result.stdout})

        # Extract counts
        failed = int(match.group(1) or 0)
        passed = int(match.group(2) or 0)
        skipped = int(match.group(3) or 0) # Treat skipped as not passed towards 100%
        total = failed + passed + skipped

        if total == 0:
             # Handle case where pytest runs but finds no tests
             return Score(value=0.0, passed=False, explanation=f"Pytest ran but found 0 tests: {summary_line}", metrics={"pytest_output": result.stdout})

        # Calculate score and check against threshold (1.0 for this task family)
        score_value = float(passed / total)
        # Success threshold is 1.0 for this task family according to manifest.yaml
        passed_threshold = score_value >= 1.0

        return Score(
            value=score_value,
            passed=passed_threshold,
            explanation=f"{passed}/{total} tests passed. Summary: {summary_line}",
            metrics={"passed": passed, "failed": failed, "skipped": skipped, "total": total}
        )

    return score


# --- Task Definition for the 'markdown' variant ---
@task
def debug_small_libs_markdown():
    """
    Defines the 'markdown' variant of the 'debug_small_libs' task from HCAST.
    """
    # Generate instructions based on original script logic
    instructions = textwrap.dedent(
            f"""
            You will find a simple markdown to HTML converter in {APP_PATH_IN_SANDBOX}.
            Run `pytest {APP_PATH_IN_SANDBOX}/tests.py` and fix any errors you encounter. Do not edit the tests file.
            """
        ).lstrip()

    # Define files to be copied into the sandbox for this specific sample
    asset_dir = HCAST_PUBLIC_DIR / TASK_FAMILY / "assets" / TASK_VARIANT
    # Check if asset dir exists before creating the sample
    if not asset_dir.is_dir():
        raise FileNotFoundError(f"Asset directory not found: {asset_dir}")

    # Copy all files from the asset directory into the sandbox app directory
    sample_files = {
        f"{APP_PATH_IN_SANDBOX}/{file.name}": str(file)
        for file in asset_dir.iterdir()
    }

    return Task(
        # Dataset containing just this one sample
        dataset=MemoryDataset([
                Sample(
                    input=instructions,
                    target="",
                    files=sample_files
                )
            ]
        ),
        # Use the react agent defined above
        solver=paper_agent,
        # Use the custom scorer defined above
        scorer=pytest_scorer(),
        # Specify sandbox type as "docker"; Inspect will look for compose.yaml
        sandbox="docker",
        # metrics={"task_family": TASK_FAMILY, "task_variant": TASK_VARIANT}
    )
