# Cyber Task Horizons

**[Read the full report here](https://sean-peters-au.github.io/2025/06/18/ai-task-length-horizons-in-offensive-cybersecurity.html)**

Evaluating modern LLMs on **offensive-cybersecurity** tasks with METR-style time-horizon analysis.

<p align="center">
  <img src="data/keep/plots/horizon_plot_p50.png" width="70%" alt="Time-horizon trend"/>
</p>

> **Key result** State-of-the-art models' 50 % success horizon doubles roughly every **5 months**. Today's best models solve **15-minute** CTF tasks half the time.

---

## What's in this repo?

| Folder | Purpose |
|--------|---------|
| `src/human_ttc_eval` | Benchmark harness & dataset adapters |
| `scripts` | Additional utility scripts |
| `data/keep` | dataset artefacts checked into git |
| `published/` | Evaluation logs, results and plots |
| `docs/` | Benchmark and code documentation |

---

## Datasets

| Dataset | Tasks | Time Range | Description |
|---------|-------|-----------|-------------|
| **CyBashBench** | 200 | 1 s – 30 s | High-frequency terminal reflexes (author-created) |
| **NL2Bash** | 162 | 4 s – 4 min | Natural-language → bash (Tellina corpus) |
| **InterCode-CTF** | 100 | 10 s – 10 min | Interactive PicoCTF-style problems |
| **NYUCTF** | 50 | 2 min – 6 h | CSAW challenges 2011-23 (dockerised) |
| **CyBench** | 40 | 2 min – 25 h | Recent pro-level CTF tasks (first-blood timings) |

Human task times are anchored by first-blood logs (CyBench) or estimates cross-checked with AI assistance.

---

## Methodology (1-min version)

1. **Estimate human time** for each task.
2. Run each model once per task inside a locked-down Docker sandbox (tool-call budgets: 15 / 25 / 25).
3. Fit a 2-PL logistic curve (ability vs log₂(time)).
4. Extract P(50) horizons and plot vs model release date.


Full details: [`docs/blog.md`](docs/blog.md) / [`docs/methodology.md`](docs/methodology.md)

---

## Quick start

```bash
# install dependencies
curl -fsSL https://get.uv.dev | bash
uv sync                 

# add API keys (OpenAI, Anthropic, Google)
cp .env.template .env  

# clone third-party repos
make third-party

# For GPT2 and GPT3
make start-local-model-servers

# run fast tier (cheap models, 1 run each)
make repro TIER=1

# generate plots
make plot
```

---

**Feedback welcome!** Open an issue or reach out on GitHub if you have suggestions or spot errors.
