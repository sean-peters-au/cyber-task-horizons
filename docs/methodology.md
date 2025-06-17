# Methodology

## METR Time Horizon Analysis Adaptation

### Core Concept

METR's foundational insight is that task difficulty can be meaningfully measured by expert human completion time. By plotting AI success rates against increasing time budgets, we can visualize capability progression and identify critical threshold crossings.

For cybersecurity tasks, this approach is particularly valuable because:
- **Speed matters operationally**: Exploit development time directly impacts attack feasibility
- **Expert baselines exist**: Competition data provides real-world timing benchmarks
- **Scaling implications**: Capability doubling times reveal when AI might exceed human experts

### Horizon Curve Construction

1. **Task Difficulty Calibration**: Use human expert completion times to assign difficulty scores
2. **AI Evaluation**: Run models on the same tasks.
3. **Success Rate Calculation**: Determine percentage of tasks solved at each capability level
4. **Curve Fitting**: Apply METR's logistic regression to identify 50% success thresholds

The resulting curves show AI capability as a function of task complexity, directly comparable across models and time periods.

## Human Baseline Collection Strategies

### Dataset-Specific Approaches

Different datasets require different strategies for obtaining human timing baselines. Our approach balances practicality with accuracy while being transparent about limitations.

#### CyBashBench (0.6s-15s): AI-Assisted with Author Calibration
- **Challenge**: No existing empirical data for atomic cybersecurity bash commands
- **Approach**: AI-generated estimates with extensive author review and calibration
- **Process**: Initial heuristic scoring, then iterative refinement with Claude 4 Opus
- **Author Involvement**: High - personally reviewed and calibrated all estimates
- **Limitation**: No empirical validation with actual cybersecurity experts

#### NL2Bash (4s-4min): AI-Assisted with CyBashBench Anchoring  
- **Challenge**: Original corpus lacks timing data for cybersecurity context
- **Approach**: AI estimates using CyBashBench as reference anchor for similar operations
- **Process**: Claude 4 Opus estimation with explicit anchoring instructions
- **Author Involvement**: Limited - spot-checking for reasonableness
- **Limitation**: Less calibrated than CyBashBench; relies on cross-dataset consistency

#### InterCode-CTF (10s-10min): AI-Assisted with Author Review
- **Source**: No reliable baseline (cybench paper cites this as a 3.5-minute average, but lacks methodology)
- **Approach**: AI-assisted estimates based on task complexity analysis
- **Process**: Task-by-task estimation with author review for plausibility
- **Author Involvement**: Medium - reviewed estimates for logical consistency
- **Limitation**: Most uncertain baseline; author expertise limited for advanced tasks

#### NYUCTF (2min-6h): AI-Assisted with CyBench Anchoring
- **Source**: CyBench competition times used as reference points
- **Approach**: AI estimates anchored on CyBench solve times and task descriptions
- **Process**: Cross-reference task complexity with CyBench examples
- **Author Involvement**: Limited - focused on logical deduction rather than domain expertise
- **Limitation**: Tasks often beyond author capability; relies heavily on AI reasoning

#### CyBench (2min-25h): Competition Times with AI Validation  
- **Source**: First-solve metadata from professional CTF competitions
- **Approach**: Used competition times with AI-assisted reasonableness review
- **Process**: Discussed each task with AI to validate timing plausibility
- **Author Involvement**: Medium - reviewed timing patterns for obvious outliers
- **Limitation**: Competition contamination effects; very few adjustments made

### AI-Assisted Annotation Philosophy

This framework was developed as a pragmatic research project with significant time and resource constraints. Conducting proper human timing studies with cybersecurity experts would require substantial funding and coordination that was not available. Despite attempts to contact original dataset authors for access to competition logs and timing data, no responses were received.

Given these constraints, the AI-assisted approach follows these principles:

1. **Transparent Uncertainty**: Clearly label estimation methodology and confidence levels
2. **Anchoring Strategy**: Use higher-quality estimates to calibrate lower-quality ones
3. **Graduated Author Involvement**: Focus personal calibration effort where most impactful
4. **Conservative Interpretation**: Emphasize trends and relative comparisons over absolute claims

## Evaluation Pipeline

### Stage 1: Retrieve
- Fetch raw dataset files, competition data, or external repositories
- No data transformation; preserve original format and metadata
- Document provenance and collection methodology

### Stage 2: Prepare
- Transform raw data into METR-compatible Run objects
- Calculate task-level weights for statistical analysis
- Separate human baselines from task definitions
- Validate schema compliance and data integrity

### Stage 3: Describe  
- Generate dataset-specific statistics and visualizations
- Analyze timing distributions and task family characteristics
- Identify potential outliers or annotation issues
- Create cross-dataset comparison metrics

### Stage 4: Bench
- Evaluate AI models using sandboxed environments (inspect_ai)
- Apply appropriate task formatting and evaluation criteria
- Generate Run objects compatible with human baselines
- Calculate summary statistics and save results

## METR Integration Approach

### Schema Compliance
All Run objects strictly conform to METR's `all_runs.jsonl` format:
- `task_id`, `task_family`, `run_id`, `alias`, `model`
- `score_binarized`, `score_cont`, `human_minutes`
- `human_source`, `task_source`, timing metadata

### Analysis Pipeline
We use METR's logistic regression and plotting code directly:
- Transform our results to METR's expected format
- Apply their curve-fitting methodology without modification
- Generate comparable horizon plots and statistics

### Model Release Date Integration
Maintain compatibility with METR's release date tracking for trend analysis across time periods.

## Evaluation Harness Design

### Task Formatting
- **Chat Models**: Appropriate system messages and user prompts
- **Completion Models**: Few-shot examples with clear formatting
- **Tool-Using Models**: Function calling for bash execution where applicable

### Scoring Approach
- **Deterministic**: Exact flag/output matching where possible
- **LLM-Assisted**: Structured scoring for functional equivalence
- **Hybrid**: Combine automated checks with LLM evaluation for nuanced tasks

### Environment Isolation
- Docker containers for challenge environments
- Network isolation to prevent information leakage
- Resource limits and security constraints
- Randomized flags and credentials per evaluation

## Limitations and Mitigation Strategies

### Known Issues
1. **Annotation Quality**: Highly variable across datasets
2. **Task Isolation**: Real attacks involve multi-day campaigns
3. **Expert Variation**: Individual differences in completion time
4. **Model Contamination**: Training data may include similar tasks
5. **Evaluation Budget Constraints**: API cost limitations restrict iteration budgets, potentially underestimating model capabilities

### Mitigation Approaches
1. **Sensitivity Analysis**: Test results across different annotation assumptions
2. **Multiple Baselines**: Compare against different expert populations where possible
3. **Conservative Interpretation**: Emphasize directional trends over precise thresholds
4. **Open Methodology**: Document all assumptions and limitations transparently

## Future Improvements

### Higher-Quality Baselines
- Controlled human studies with cybersecurity professionals
- Multiple expert annotations per task with inter-rater reliability
- Task-specific validation of AI-generated time estimates

### Expanded Coverage
- Multi-step campaign scenarios spanning longer time horizons
- Defensive cybersecurity tasks (not just offensive)
- Cross-domain transfer assessment

### Enhanced Evaluation
- Process-level assessment (how tasks are solved, not just success/failure)
- Adaptive evaluation that responds to model capabilities
- Integration with red-teaming and safety evaluation frameworks

### Realistic Evaluation Budgets
With proper funding, evaluations should provide cost-equivalent API budgets to human contractor fees rather than the current pragmatic constraints. This would enable:

- **Generous Iteration Budgets**: Allow models sufficient attempts to solve complex challenges rather than the current 30-command limit that may be inadequate for professional-level tasks
- **Multiple Evaluation Runs**: Statistical robustness through repeated evaluations rather than single attempts
- **Extended Time Horizons**: Full exploration of model capabilities on multi-hour challenges without premature timeout
- **Comprehensive Coverage**: Evaluate full datasets rather than selective subsampling due to cost constraints

The current evaluation approach likely underestimates model capabilities, particularly on professional-grade challenges where human experts would naturally iterate, explore dead ends, and refine approaches over extended periods.

The goal is continuous improvement while maintaining scientific rigor and transparent uncertainty quantification.