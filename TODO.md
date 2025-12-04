DONE: https://ukgovernmentbeis.github.io/inspect_evals/evals/cybersecurity/cve_bench/
DONE: Integrate new models into the CyBench benchmark harness
DONE: Amend max_iterations for CyBench and other benchmarks

## misc
TODO: Fix release dates for new models
TODO: NYUCTF concurrency

DONE: Add is_sota flag to models used in horizon analysis (SOTA models from blog post)
DONE: Add missing alias fields for all models that might appear in plots
DONE: Move _create_error_result to base Bench class
DONE: Standardise path construction
DONE: Add docstrings for resummable tasks 
DONE: Metadata for new benchmarks (desc, target_platofrm, app_url, variants, attacker_role)
DONE: 30 min per sample timeout 
DONE: Sort out task JSON fields (e.g some benchmarks have categories, some don't)
TODO: integrate multi-shot / multi-attempt logic
TODO: integrate and test gpt 5.x models with new reasoning/non-reasoning api logic 
TODO: Why are most tasks so high in cached tokens? Is it a anthropic API thing? Inspect community? 


## benchmarks
DONE: Integrate CyberGym benchmark (subset: 10 tasks) https://github.com/sunblaze-ucb/cybergym
TODO: Run both CVE variants ? 
TODO: Reach out to cybergym authors re task sampling

## models
TODO:new open-models (kimi, glm etc) (together vs openrouter)

- Agent scaffolding ?
-  