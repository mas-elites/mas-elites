# Do Agent Societies Develop Intellectual Elites?
### The Hidden Power Laws of Collective Cognition in LLM Multi-Agent Systems

[![Paper](https://img.shields.io/badge/arXiv-2604.02674-b31b1b.svg)](https://arxiv.org/abs/2604.02674)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)


## Overview

This repository contains the official code and data for the paper *"Do Agent Societies Develop Intellectual Elites? The Hidden Power Laws of Collective Cognition in LLM Multi-Agent Systems"*.

We present the first large-scale empirical study of coordination dynamics in LLM-based multi-agent systems. Analyzing over **1.5 million coordination events** across tasks, topologies, and agent scales, we uncover three coupled empirical laws governing collective reasoning, and introduce **Deficit-Triggered Integration (DTI)** as a law-aware intervention.


<p align="center">
  <img src="images/global-ccdf.png" width="1000"/>
</p>


## Abstract

Large Language Model (LLM) multi-agent systems are increasingly deployed as interacting agent societies, yet scaling these systems often yields diminishing or unstable returns, the causes of which remain poorly understood. We present the first large-scale empirical study of coordination dynamics in LLM-based multi-agent systems, introducing an atomic event-level formulation that reconstructs reasoning as cascades of coordination. Analyzing over **1.5 Million** interactions across tasks, topologies, and scales, we uncover *three* coupled laws: coordination follows heavy-tailed cascades, concentrates via preferential attachment into intellectual elites, and produces increasingly frequent extreme events as system size grows. We show that these effects are coupled through a single structural mechanism: an **integration bottleneck**, in which coordination expansion scales with system size while consolidation does not, producing large but weakly integrated reasoning processes. To test this mechanism, we introduce **Deficit-Triggered Integration (DTI)**, which selectively increases integration under imbalance. DTI improves performance precisely where coordination fails, without suppressing large-scale reasoning. Together, our results establish quantitative laws of collective cognition and identify coordination structure as a fundamental, previously unmeasured axis for understanding and improving scalable multi-agent intelligence.


## Key Findings

**H1 — Heavy-tailed coordination cascades.** Coordination event sizes (delegation cascades, revision waves, contradiction bursts, merge fan-in, and total cognitive effort) consistently follow truncated power-law distributions with exponents α̂ ∈ (2, 3), across all tasks, topologies, agent scales, and model families.

**H2 — Emergence of intellectual elites.** Cognitive effort concentrates endogenously in a small subset of agents through preferential attachment in claim selection. The top 10% of active agents account for over 30% of coordination effort at large scales, and this gap widens with system size.

**H3 — Systematic expansion of extreme coordination events.** The expected maximum cascade size grows as ⟨x_max⟩ ∝ N^γ, consistent with Extreme Value Theory predictions derived from the heavy-tailed cascade structure (γ̂_TCE ≈ 0.85 vs. γ_th ≈ 0.82).

These three laws are mechanistically coupled through an **integration bottleneck**: coordination expansion (delegation, contradiction) scales with agent count while consolidation (merge) does not, producing large but weakly integrated reasoning trajectories and explaining non-monotonic scaling behavior.

---

## Method

### Event-Based Coordination Formulation

We decompose multi-agent reasoning into atomic coordination primitives extracted from interaction traces:

| Event Type | What It Captures |
|---|---|
| **Delegation Cascade** | Recursive task decomposition and agent recruitment |
| **Revision Wave** | Iterative refinement of a claim |
| **Contradiction Burst** | Parallel critique centered on one claim |
| **Merge Fan-In** | Information integration bottleneck |
| **Total Cognitive Effort (TCE)** | Aggregate cascade size from a root claim |

Coordination observables are computed from a reconstructed claim DAG and subtask tree. Agents are prompted only with task goals and topology constraints — event types are inferred post-hoc from realized interaction traces, not injected via prompt.

### Deficit-Triggered Integration (DTI)

DTI is a cascade-local intervention that monitors the imbalance between expansion pressure and realized integration. When the integration deficit Δ_r(t) exceeds a condition-specific threshold δ_c, DTI temporarily prioritizes merge operations over further expansion. DTI:

- Preserves the heavy-tailed cascade structure (α̂ unchanged)
- Reduces excess tail mass and moderates elite concentration
- Improves task success most where expansion-integration imbalance is strongest (up to +12.34% on Planning × Mesh/FC)

---

## Experimental Setup

| Component | Configuration |
|---|---|
| Benchmarks | GAIA, SWE-bench Verified, REALM-Bench, MultiAgentBench |
| Task types | QA, reasoning, coding, planning |
| Agent counts | {8, 16, 32, 64, 128, 256, 512} |
| Topologies | Chain, Star, Tree, Hierarchical, Fully Connected, Sparse Mesh, Dynamic Reputation |
| Total runs | ~98,000 (400 tasks × 7 scales × 7 topologies × 5 seeds) |
| Total events | >1.5 million coordination events |
| Execution | LangGraph; shared LLM, prompt, tools, and task instances per run |


## Installation

```bash
git clone https://github.com/[anonymous]/mas-powerlaws.git
cd mas-powerlaws
pip install -r requirements.txt
```

Set your API key:

```bash
export OPENAI_API_KEY=your_key_here
```

---

## Running Experiments

**Dry run (verify sweep plan without executing):**
```bash
python scripts/run_sweep.py --dry-run
```

**Small-scale test:**
```bash
python scripts/run_sweep.py \
    --benchmarks gaia \
    --topologies chain star \
    --scales 8 16 \
    --seeds 0 1
```

**Full sweep:**
```bash
python scripts/run_sweep.py --workers 4
```

The sweep script supports resumption (`--resume`), parallel workers, live progress tracking, and per-run error logging.


## License

This project is licensed under the APACHE 2.0 License. See [LICENSE](LICENSE) for details.
