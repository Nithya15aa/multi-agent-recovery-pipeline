# Multi-Agent Failure Recovery Pipeline

An LLM pipeline that catches its own wrong answers and fixes them without you having to do anything.

When the first answer is wrong, it figures out why, looks up similar past failures from memory, and retries from the exact step where things went wrong — not from scratch.

Built with Python and Gemini 2.5 Flash API. Orchestration is custom-built — no frameworks. Evaluated on GSM8K (math word problems), ARC-Challenge (science questions), and BoolQ (yes/no reading comprehension).

---

## The 6 agents and what each one does

**ExecutionAgent**
Takes the problem and produces an answer. Uses step-by-step reasoning and calls a calculator tool for any arithmetic so numbers don't get hallucinated.

**VerificationAgent**
Checks if the answer is right. Uses ground truth when you have it. Falls back to asking the LLM to critique its own output when you don't.

**ClassifierAgent**
If the answer is wrong, figures out what kind of failure it was. Five categories: arithmetic error, misread problem, missing constraint, logic gap, knowledge gap. The category list can split over time as the system learns which failures behave differently from each other.

**StepAttributor**
Runs a second independent attempt at the same problem, then compares the two reasoning chains line by line to find exactly where they first diverged. This tells the recovery agent which step broke — so it doesn't have to redo the whole thing.

**RecoveryAgent**
Builds a recovery prompt using three things: the failure type, the exact divergence step, and similar past failures pulled from memory. Retries from the broken step only.

**PipelineOrchestrator**
Runs the whole loop. Before retrying, checks historical success rates for this failure type — if recovery is unlikely to work, it skips it and saves the compute. Updates memory and success rates after every run.

---

## Three ideas that make it work

**RAG memory**
Every failure gets stored: the question, what went wrong, and whether recovery worked. When a new failure happens, it pulls the 2-3 most similar past cases using TF-IDF cosine similarity and feeds them to the recovery agent as context.

**Bayesian recoverability estimator**
A Beta-Binomial model tracks how often each failure type gets successfully recovered, per domain. Before retrying, it estimates the probability of success. If it's too low, the pipeline skips the retry. This avoids wasting API calls on failures that almost never recover.

**Self-mutating failure taxonomy (SMFO)**
The five failure categories aren't fixed. When a category builds up enough cases where some recover and some don't, it splits into two subcategories using Agglomerative Clustering. The system gets more precise at classifying failures the longer it runs.

---

## Results

| Method | GSM8K | ARC-Challenge | BoolQ |
|---|---|---|---|
| Single pass | 82% | 74% | 79% |
| Majority vote (k=3) | 88% | 78% | 83% |
| This pipeline | **91%** | **83%** | **85%** |

Of all the first attempts that were wrong, how many got fixed:
- GSM8K: 53% recovered
- ARC: 38% recovered
- BoolQ: 30% recovered

---

## Setup

```bash
pip install google-genai datasets scikit-learn numpy matplotlib
```

If `google-genai` doesn't install, try:
```bash
pip install google-generativeai datasets scikit-learn numpy matplotlib
```

---

## Running it

Set your Gemini API key:
```bash
export GEMINI_API_KEY=your_key_here
```

Or pass it directly:
```bash
python run_evaluation.py --domain gsm8k --samples 50 --api-key YOUR_KEY
```

Run on different datasets:
```bash
python run_evaluation.py --domain arc --samples 200
python run_evaluation.py --domain boolq --samples 50
```

Run all ablation variants at once:
```bash
python run_evaluation.py --domain gsm8k --samples 50 --all-ablations
```

---

## Generating figures

After a run, results go into `./results/`. To generate charts:

```bash
python visualize_results.py --results_dir ./results --output_dir ./figures
```

Generates 10 figures as PDF and PNG:
- Main accuracy comparison across methods
- Recovery rate heatmap by failure type and domain
- Recoverability estimator calibration (estimated vs actual)
- Ablation study showing what each component contributes
- RSR convergence as more samples run
- Verification confusion matrices
- Compute vs accuracy tradeoff
- Which reasoning step fails most often
- Failure type distribution per domain
- Bootstrap confidence intervals

Also generates a LaTeX table for the main results.

---

## Folder structure

```
.
├── run_evaluation.py       # Run this to evaluate
├── visualize_results.py    # Run this to generate figures
├── core/
│   ├── __init__.py
│   ├── agents.py           # All 6 agent classes
│   ├── evaluation.py       # EvaluationRunner and ablation configs
│   └── tool_routing.py     # Calculator tool for arithmetic
├── results/                # Created automatically after a run
└── figures/                # Created by visualize_results.py
```

---

## Ablation configs

Each config removes one component so you can see what it contributes:

| Config | What's removed |
|---|---|
| `full_pipeline` | Nothing — full system |
| `A1_no_rag` | RAG memory |
| `A2_random_classify` | Correct classification (random instead) |
| `A3_monolithic` | Multi-agent structure (single prompt instead) |
| `no_step_attribution` | Surgical recovery (full retry instead) |
| `no_recoverability` | Recoverability check (always retries) |
| `A5_no_ground_truth` | Ground truth verification (LLM critique only) |

---

## Why I built this

I wanted to understand what actually goes wrong inside LLM reasoning and build something that fixes it systematically. Every part of this — the Bayesian estimator, the step attribution, the taxonomy splitting — came from a specific failure mode I ran into and had to solve.

The pattern also applies directly to production systems: infrastructure and software that detects its own failure modes and recovers without waking someone up at 3am.
