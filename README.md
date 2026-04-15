# Do Agent Societies Develop Intellectual Elites? The Hidden Power Laws of Collective Cognition in LLM Multi-Agent Systems

## Overview

We present the first large-scale empirical study of coordination dynamics in LLM-based multi-agent systems. Analyzing over **1.5 million coordination events** across tasks, topologies, and agent scales, we uncover three coupled empirical laws governing collective reasoning, and introduce **Deficit-Triggered Integration (DTI)** as a law-aware intervention.


<p align="center">
  <img src="images/global-ccdf.png" width="1000"/>
</p>


## Abstract

Large Language Model (LLM) multi-agent systems are increasingly deployed as interacting agent societies, yet scaling these systems often yields diminishing or unstable returns, the causes of which remain poorly understood. We present the first large-scale empirical study of coordination dynamics in LLM-based multi-agent systems, introducing an atomic event-level formulation that reconstructs reasoning as cascades of coordination. Analyzing over **1.5 Million** interactions across tasks, topologies, and scales, we uncover *three* coupled laws: coordination follows heavy-tailed cascades, concentrates via preferential attachment into intellectual elites, and produces increasingly frequent extreme events as system size grows. We show that these effects are coupled through a single structural mechanism: an **integration bottleneck**, in which coordination expansion scales with system size while consolidation does not, producing large but weakly integrated reasoning processes. To test this mechanism, we introduce **Deficit-Triggered Integration (DTI)**, which selectively increases integration under imbalance. DTI improves performance precisely where coordination fails, without suppressing large-scale reasoning. Together, our results establish quantitative laws of collective cognition and identify coordination structure as a fundamental, previously unmeasured axis for understanding and improving scalable multi-agent intelligence.


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
