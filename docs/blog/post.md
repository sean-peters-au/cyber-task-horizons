---
layout: post
title: "AI Task Length Horizons in Offensive Cybersecurity"
date: 2025-06-29
scholar: {}
---

_This is a rough research note where the primary objective was my own learning. I am sharing it because I'd love feedback and I thought the results were interesting._

## Introduction

A recent METR paper {% cite metr2024horizons %} showed that the length of software engineering tasks that LLMs could successfully complete appeared to be doubling roughly every seven months. I asked the same question for offensive cybersecurity, a domain with distinct skills and unique AI-safety implications.

Using METR's methodology on five cyber benchmarks, with tasks ranging from 0.5s to 25h in human-expert estimated times, I evaluated many state of the art model releases over the past 5 years. I found:
1. Cyber task horizons are doubling every ~5 months.
2. The best current models solve 15-minute tasks with a 50% success rate.

Below I outline the datasets, IRT-based analysis, results and caveats. All code artifacts, evaluation logs, and results can be found [here](https://github.com/sean-peters-au/cyber-task-horizons).

## Methodology

This work reproduces the methodology of METR's _Measuring AI Ability to Complete Long Tasks
_ paper {% cite metr2024horizons %}. In brief:

1. Annotate each task with estimated expert-human time.
2. Run each model once per task in a sandbox (with a tool-call budget).
3. Fit a 2-PL IRT curve to the success-vs-time dataset for each model.
4. Read off the 50% horizon; plot horizon vs release date.

#### Item Response Theory

Digging deeper into the approach requires starting with Item Response Theory (IRT).

IRT comes from the field of psychometrics and has become a de-facto standard{% cite rasch1980probabilistic %}. It was invented so exam designers could:
- Assign a latent ability to each student that is comparable across very different question sets.
- Assign a latent difficulty to each item that is comparable across very different test-takers.
- Predict success probabilities on unseen student × item pairs.
It is straightforward to see how this approach, can be immediately applied to model benchmarking.

In its simplest form IRT, just comes down to a simple logistic relationship,

$$P(\text{success}) = \frac{1}{1 + e^{-a(\theta - \beta)}}$$

where:
> $a$: Discrimination parameter  
> $\theta$: Model ability level  
> $\beta$: Item difficulty level

METR's key insight was to use human task time length as a proxy for difficulty in this expression. Equipped with this, model task evaluations provide a means to capture probability densities at varying task lengths. With logistic regression, we can calculate these curves. METR found these regressions fit well.

Finally, we can then start plotting probability of success time horizon curves and examine trends e.g. the task horizon lengths at P(50) for different models. This is where we the ~5 month doubling times come from.

I follow the same methodology, but introduce new datasets (see table below); 4 existing datasets and a new constructed dataset, with the goal of covering a wide span of human task lengths in the offensive cybersecurity domain. This wide span of task lengths is important for well-constrained logistic curves for older models through to modern state-of-the-art models.

### Datasets

To span both micro-commands and day-long exploit chains, I combined five benchmarks (Table 1). CyBashBench and NL2Bash cover sub-minute terminal work, while three CTF datasets extend into multi-hour challenges across reversing, binary exploitation, crypto, web, and other "real-world" attack scenarios.

<figure markdown="1">
<figcaption><strong>Table&nbsp;1.</strong> Summary of the five offensive cybersecurity datasets used in this evaluation.</figcaption>

| Dataset | Tasks | Time Range | Source | Description | Scoring | Human Baseline |
|---------|-------|------------|---------|-------------|---------|----------------|
| **CyBashBench** | 200 | 1s-30s | Author-created | Extremely short horizon terminal tasks | Functional equivalence (AI assessor) | Estimated |
| **NL2Bash** | 162 | 4s-4min | [TellinaTool corpus](https://arxiv.org/abs/1802.08979) | Natural language to bash translation | Functional equivalence (AI assessor) | Estimated (AI assistance) |
| **InterCode-CTF** | 100 | 10s-10min | [InterCode framework](https://arxiv.org/abs/2306.14898) | Interactive CTF solving with bash/python | Flag discovery | Estimated (AI assistance) |
| **NYUCTF** | 50 | 2min-6h | [NYU CTF Dataset](https://arxiv.org/abs/2406.05590) | CSAW-CTF challenges (2011-2023), dockerized | Flag discovery | Estimated (AI assistance) |
| **CyBench** | 40 | 2min-25h | [Professional CTFs](https://arxiv.org/abs/2408.08926) | Recent high-level competitions, full docker environments | Flag discovery | Competition data |

</figure>

<figure id="fig-task-dist">
  <img src="{{ '/assets/images/cyber/task_length_distribution.png' | relative_url }}" alt="Task Length Distribution">
  <figcaption><strong>Figure&nbsp;1.</strong> Distribution of task lengths across the five datasets, plotted on a log scale.</figcaption>
</figure>

#### CyBashBench

I constructed this dataset to capture the gap of short horizon tasks. 

Psychometric work on speed tests shows that very short horizon task performance hinges on frequency-driven automaticity: the more often an expert has completed a task, the lower the task time.{% cite shiffrin1977 %}. Thus, tasks in this dataset are limited to those that human cybersecurity experts would plausibly have high frequency exposure to. The lower the task time, the more this principle is qualitatively applied.

In order to get human time estimates, a subset of the tasks were completed by hand for timing anchors. The remaining were then estimated.

Borrowing the core NL → bash idea from NL2Bash ({% cite lin2018nl2bash %}) and the METR's SWAA ({% cite metr2024horizons %}), we mix six minimal formats to cover both recall and recognition:

1. Full translation – natural-language task → entire command.
2. Prefix completion – NL task + opening tokens → finish the line.
3. Fill-in-the-blank – one flag/arg blanked out.
4. Last-step chaining – earlier pipeline steps provided; supply the terminal command.
5. Multiple choice (MCQ) – pick the right command/flag from 4 options.
6. Single-token close – predict exactly one missing token.

Given tasks can be solved by a variety of solutions, answers were graded by `o4-mini-2025-04-16` for functional equivalence.

**NL2Bash**

This dataset includes a large sample of natural language to command line translations. While less cybersecurity specific by simple nature of being sourced from different authors it offers diversity to the _CyBashBench_ dataset. This dataset also includes more sophisticated command targets. Human times were iteratively estimated with AI assistance by manually estimating anchors, prompting the `Claude 4 Opus` to a subset of tasks, reviewing model estimates, and repeating this process across the dataset.

As with _CyBashBench_, tasks can be solved by a variety of solutions, and similarly, answers were graded by `o4-mini-2025-04-16` for functional equivalence.

**Intercode CTF**

This dataset, published in late 2023, contains _capture the flag_ competition tasks from the popular _PicoCTF_ online platform targeting high schoolers and other students{% cite yang2023intercode %}. The problems aim to be introductory and provide an avenue for hands on cybersecurity learning.

An example task in this dataset has the competitor exploit a reused XOR keystream. The intended solution involves sending known plaintext to the server and XOR-ing the returned ciphertext twice, thus recovering the flag.

Models were sandboxed in docker containers with the Intercode CTF provided Dockerfile, and run with the extensible ReAct agent{% cite yao2023react %} made available within `inspect_ai`. In each task the model was provided a budget of 15 tool calls before the assessment was marked as incorrect.

Success was graded by agent submission of the correct flag.

Task estimates were derived as described in `NL2Bash`, with the exception that simpler `CyBench` tasks were used as references in order to approximate task lengths.

**NYU CTF Bench**

_NYU CTF Bench_{% cite shao2024nyuctf %} continues the CTF trend. This dataset includes tasks from the CSAW qualifiers and finals. These competitions are targeted at university students or individuals looking to break into the field. While there is some range, many of the tasks in this dataset are genuinely difficult even for experts.

A representive task requires the participant to reverse-engineer an implementation of RC4 (a stream cipher) and identify an XOR-swap vulnerability, where the pseudorandom generation algorithm contains a subtle bug that causes the keystream to slowly collapse to nearly all 0x00. This provides an attacker an opportunity to "flood" the cipher out eventually reading the plaintext directly.

Models were sandboxed in task specific containers provided in the _NYU CTF Bench_ public repository. The agent scaffold and grading matched the _Intercode CTF_ setup, but with an increased budget of 25 tool calls.

Task estimates were derived as described in `NL2Bash`, with the exception that `CyBench` tasks were used as references in order to approximate task lengths.

**CyBench**

Finally _CyBench_{% cite zhang2024cybench %} contains tasks from a variety of CTF competitions hosted around the world. These tasks can get exceptionally sophisticated. Many tasks taking world class CTF experts many hours – with the hardest task coming in at a 25 hours submission team, by the first team to solve it.

This is the only dataset for which human time task lengths could be estimated with grounded data – the _first blood times_. First blood times represent the time of first successful submission in the originally published competition. While there are some limitations (participants usually compete in teams, and may solve problems in parallel or in sequence), this still provides a useful proxy. The first blood times were reviewed with AI assistance. The vast majority of first blood times stood as the human task length estimate.

A representative task in this dataset requires the participant to reverse-engineer a key-storage API that tracks membership with a Bloom filter built on MurmurHash3; by leveraging a published pair of universal collision inputs, the attacker inserts one value that flips exactly the 47 hash-selected bits, so the service later believes the second (regex-compliant) value is already present and grants admin access, exposing the flag. The first blood time for this task was ~7.5 hours.

Models were sandboxed in a Kali linux container and scaffolded and scored as per _InterCode CTF_ and _NYU CTF Bench_ with a maximum budget of 25 tool calls.

### Models

The evaluated models span 2019 to mid-2025 (Table 2), chosen with the goal of capturing coverage of state of art releases across the period. Some concessions were made in order to reduce the API costs of the project. Additionally, many previously state of the art models are now deprecated by providers and not publicly available.

<figure markdown="1">
<figcaption><strong>Table&nbsp;2.</strong> Models evaluated, spanning from 2019 to mid-2025. Only models marked "State of the Art" were used for trend analysis.</figcaption>

| Release Date | Provider  | Model Name                     | Type         | State of the Art |
|--------------|-----------|--------------------------------|--------------|------------------|
| 2019-11-05   | OpenAI    | gpt2-xl                      | Completion   | Yes              |
| 2020-07-11   | OpenAI    | davinci-002                  | Completion   | Yes              |
| 2022-03-15   | OpenAI    | gpt-3.5-turbo                | Chat         | Yes              |
| 2024-06-20   | Anthropic | claude-3-5-sonnet-20240620   | Chat         | Yes              |
| 2024-10-22   | Anthropic | claude-3-5-haiku-20241022    | Chat         | No               |
| 2024-10-22   | Anthropic | claude-3-5-sonnet-20241022   | Chat         | Yes              |
| 2025-01-31   | OpenAI    | o3-mini-2025-01-31           | Chat         | Yes              |
| 2025-04-16   | OpenAI    | o4-mini-2025-04-16           | Chat         | No               |
| 2025-04-16   | OpenAI    | o3-2025-04-16                | Chat         | Yes              |
| 2025-06-05   | Google    | gemini-2.5-pro-20250605      | Chat         | Yes              |

</figure>

Completion-only models (`gpt2-xl`, `davinci-002`) provide a challenge for modern LLM evaluations. For _CyBashBench_ and _NL2Bash_ these models were wrapped in a few-shot task specific templates that provided tool-use examples so as to direct them towards task completion. For all _capture the flag_ datasets a zero score was imputed (which I believe to a fair assessment).

Finally, `gpt2-xl` was run locally from the HuggingFace checkpoint as it is no longer available via API.

Each model produced one attempt per task; tool calls capped at 15 / 25 / 25 tool calls for InterCode, NYUCTF, and CyBench respectively.

Only models marked as "state of the art" above, were used in horizon trends analysis.

### Results

Figure 2 plots the fitted IRT curves and the resulting 50 percent success horizons. The logistic fits are respectable: McFadden pseudo-R² values sit between 0.25 and 0.40, which is generally considered a good fit for this metric. Two points stand out. First, state-of-the-art horizons double in length roughly every five months. Second, the best 2025 models can solve fifteen-minute offensive-cyber tasks half the time, about a quarter of the horizon METR found for software-engineering tasks. Figure 3 gives the per-model fit diagnostics.

<figure id="fig-p50-horizons">
  <img src="{{ '/assets/images/cyber/horizon_plot_p50.png' | relative_url }}" alt="P(50) Time Horizons">
  <figcaption><strong>Figure&nbsp;2.</strong> P(50) time horizons for state-of-the-art models plotted against their release date. The exponential fit (R²=0.94) suggests a doubling time of approximately 5 months.</figcaption>
</figure>

<figure id="fig-ind-histograms">
  <img src="{{ '/assets/images/cyber/individual_histograms.png' | relative_url }}" alt="Individual Model Histograms">
  <figcaption><strong>Figure&nbsp;3.</strong> Individual IRT curves for each model, showing the probability of success as a function of task length. Newer models consistently demonstrate higher success probabilities across all task lengths.</figcaption>
</figure>

#### 5 Month Doubling Times

A five-month doubling rate implies that today's 15-minute horizon could reach a full week in about four years. This pace compresses the "adaptation buffer", the window in which defenders can prepare before the same capabilities become broadly available, highlighted in Toner 2025 ({% cite toner2025adaptation %}). The slope almost matches the seven-month doubling METR found for software-engineering tasks, which is unsurprising because both curves likely trace the same industry drivers (scaling, algorithmic tuning, RLHF, longer inference computation). Offensive-cyber tasks start from a much shorter horizon, but the underlying improvement rate appears similar. If this empirically observed trend holds, longer-chain intrusion or persistence operations will move from "research demo" to "commodity script" on a short clock, something worth factoring into misuse-risk planning even though deeper policy discussion is beyond this post's scope.

#### 15 Minute Task Horizons

The 15-minute P(50) success horizon on these benchmarks is substantially lower than the ~60-minute horizon reported in METR's software engineering study. While a shorter horizon for cyber tasks is not surprising, the factor-of-four difference is large enough to be noteworthy.

I speculate this gap is due to two kinds of reasons:

**Measurement Factors**

- Our time estimates are anchored by first blood times from elite global CTF teams. These are likely much faster than the times from the professional software developers METR hired to set their baselines, which could account for a large part of the gap.
- Other sources of measurement noise may also contribute, as discussed in the Limitations section.

**Task Factors**
- Offensive cybersecurity often requires long, exploratory chains of reasoning where models appear to underperform.
- Exploit chains are typically brittle. A single incorrect step can cause failure with little feedback, whereas coding tasks offer more opportunities for debugging and self-correction.

### Limitations

The points below concern internal validity of this study

**1. Human task time lengths are AI-assisted estimates**

Instead of genuine human task time lengths (or strong proxies e.g. first blood times), most of the task length times were estimated with AI assistance. Higher quality work would have used real human baselines, however this was not logistically possible for a hobby project. Additionally, estimates would have been significantly improved with a human offensive cybersecurity professional. As a software engineer by trade, I found that as the task length and complexity grew, I needed to lean heavily on AI support for estimation. 

The original ambition was that I could scrape CTF competitions websites for event logs and submission times. Popular open-source CTF hosting software typically has this sort of stuff available on APIs. Unfortunately, most old CTF competition pages are either missing, or inaccessible. I did reach out to the authors behind each of the CTF datasets but did not hear back.

**2. Dataset contamination**

It is a near certainty that many of the tasks in the collection of datasets in this study, would have been included in many of the model pretraining sets. This could inflate absolute success rates and therefore the time horizons trends. Recent analyses show contamination can inflate benchmark accuracy by up to ~14 pp on some tasks while being completely negligible in others{% cite li-etal-2024-open-source %}.

It is however worth noting the public benefit of open datasets like Cybench, NYUCTF, Intercode-CTF, and NL2Bash – they have allowed someone like me to do this work.

**3. Prompt and tool-budget design**

The specific prompts, few-shot templates, and capped tool-call budgets used here constrain model behaviour and could systematically inflate or depress scores.

**4. Single run per task**

Each model attempted every task once to control cost, so stochastic variance in model outputs could shift individual success rates.

**5. AI grader noise**

Automatic functional-equivalence grading relies on an LLM and may occasionally misjudge success or failure, introducing measurement noise.

### Personal Retrospective & Next Steps

I started this project with the goal of learning by building _something_ within the AI safety domain. What follows is an unstructured set of notes on what I found most surprising, interesting, or simply just personally worthwhile to remember.

**Bad evals are easy, good evals are hard**. Tackling and solving for brittle evaluations was easily the bulk of my time. This unsurprisingly included many, many examples of misconfiguration in my own benchmark harness, but extended to existing benchmark tasks too, which were in some cases were entirely unsolvable.

**Deprecating model APIs will close off this methodology to public researchers**. There are a number of models I would have liked to have included (e.g. GPT-4, and early Gemini and Claude models), that appear to have been entirely deprecated to the public. It would be fantastic if labs would or do grant safety researchers access to archived checkpoints.

**Inspect AI**{% cite inspectai2024 %} is a fantastic framework. I should have gotten onto it earlier and I should have gone more all in on it e.g. using .eval files as the primary evaluation artefact. This is a project worth contributing to.

**Models cheat**. There are plenty of existing examples of this being observed, but nevertheless it was surprising when I encountered it myself. Several times I caught models searching the web for CTF writeups, until I more deliberately constrained sandbox network access.

**This can get expensive if self-funded**. I ran all off this out of my own wallet. This greatly limited the breadth of models I decided to benchmark and the depth to which I could benchmark each one. I would have loved to include more tasks from both NYUCTF and CyBench. I would have loved to give them higher token budgets. I would have loved to have done multiple runs of each model on each task.

&nbsp;

I am currently not planning to extend this particular study any further. However I am still very interested in any feedback or suggestions, particularly as there other projects I'd like to work on in this space. Potential ideas include; extending this methodology to scheming-capability horizons, or cybersecurity defense vs offense horizons.