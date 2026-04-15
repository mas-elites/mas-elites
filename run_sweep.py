"""
scripts/run_sweep.py
====================
Main execution entry point for the LLM MAS Power-Laws full experimental sweep.

Experiment grid
---------------
  Topologies : 7  (chain, star, tree, full_mesh, sparse_mesh,
                   hybrid_modular, dynamic_reputation)
  Scales     : N ∈ {8, 16, 32, 64, 128, 256, 512}
  Seeds      : 3  (0, 1, 2, 4, 5)
  Benchmarks : GAIA (qa / reasoning), MultiAgentBench (planning / coordination),
               REALM-Bench (planning), SWE-bench (coding)

Output layout
-------------
  data/sweep/
    {benchmark}/{topology}/n{N}/s{seed}/{task_id}/
      events.jsonl          raw TraceRows (one per agent turn + completions)
      task_tree.json        TaskExpander output
      run_metadata.json     H2 metrics + benchmark scores
      run_config.json       full run configuration
    sweep_manifest.jsonl    one entry per completed run
    sweep_errors.jsonl      one entry per failed run
    sweep_progress.json     live counters + ETA
    sweep_plan.json         full grid recorded at launch
    sweep_complete.json     written on successful completion

Usage
-----
  cd ~/mas-powerlaws
  export $(grep -v '^#' .env | xargs)

  # Validate the plan without making any API calls
  python scripts/run_sweep.py --dry-run

  # Targeted subset
  python scripts/run_sweep.py \\
      --benchmarks gaia \\
      --topologies chain star full_mesh \\
      --scales 8 16 \\
      --seeds 0 1

  # Full sweep, 4 parallel workers, resume on restart
  python scripts/run_sweep.py --workers 4 --resume

  # Full sweep with LLM-based event disambiguation (recommended for final runs)
  python scripts/run_sweep.py --workers 4 --resume --llm-judge gpt-4o

Exit codes
----------
  0  all queued runs completed (some may have non-fatal errors — check
     sweep_errors.jsonl for details)
  1  fatal setup error before any run started (missing API key, etc.)
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ── Repository root on sys.path ──────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("sweep")


# ── Experiment grid (paper Section 3.1 / Appendix G) ────────────────────────

ALL_TOPOLOGIES: List[str] = [
    "chain",
    "star",
    "tree",
    "full_mesh",
    "sparse_mesh",
    "hybrid_modular",
    "dynamic_reputation",
]

ALL_SCALES: List[int] = [8, 16, 32, 64, 128]

ALL_SEEDS: List[int] = [0, 1, 2]

# (benchmark, task_family, max_tasks)  — sizes match paper Table 26
BENCHMARK_CONFIGS: List[Tuple[str, str, int]] = [
    ("gaia",     "qa",           20),
    ("gaia",     "reasoning",    20),
    ("marble",   "planning",     20),
    ("marble",   "coordination", 10),
    ("realm",    "planning",     20),
    ("swebench", "coding",       10),
]


# ── Run specification ────────────────────────────────────────────────────────

@dataclass(frozen=True)
class RunSpec:
    """Fully-specified single experimental cell."""
    benchmark:   str
    task_family: str
    task_id:     str
    task_prompt: str
    gold_answer: Optional[str]
    topology:    str
    num_agents:  int
    seed:        int
    run_dir:     Path

    @property
    def run_id(self) -> str:
        return (
            f"{self.benchmark}__{self.topology}"
            f"__n{self.num_agents}__s{self.seed}__{self.task_id}"
        )

    def is_complete(self) -> bool:
        """True if run_metadata.json already exists under run_dir."""
        return (self.run_dir / "run_metadata.json").exists()


# ── Task loading ─────────────────────────────────────────────────────────────

def load_tasks(
    benchmark:   str,
    task_family: str,
    max_tasks:   int,
) -> List[Dict[str, Any]]:
    """
    Load benchmark tasks via the appropriate wrapper module.

    Falls back to clearly-labelled synthetic tasks when the dataset wrapper
    is unavailable (e.g. the dataset has not been downloaded yet).  Synthetic
    tasks are excluded from paper-level analysis via the ``benchmark_source``
    field on each TaskNode.

    Returns a list of normalised task dicts with keys:
        task_id, prompt, gold_answer (optional)
    """
    def _normalise(raw: Any, idx: int) -> Dict[str, Any]:
        if hasattr(raw, "task_id"):
            return {
                "task_id":    raw.task_id,
                "prompt":     getattr(raw, "prompt", None) or getattr(raw, "description", ""),
                "gold_answer": getattr(raw, "gold_answer", None),
            }
        if isinstance(raw, dict):
            return {
                "task_id":    raw.get("task_id") or raw.get("id", f"task_{idx:04d}"),
                "prompt":     raw.get("prompt") or raw.get("description", ""),
                "gold_answer": raw.get("gold_answer") or raw.get("answer"),
            }
        return {
            "task_id":    f"task_{idx:04d}",
            "prompt":     str(raw),
            "gold_answer": None,
        }

    try:
        if benchmark == "gaia":
            from benchmark_wrappers.gaia import load_gaia_tasks
            raw_tasks = load_gaia_tasks(max_tasks=max_tasks, task_family=task_family)
        elif benchmark == "marble":
            from benchmark_wrappers.multiagentbench import load_multiagentbench_tasks
            raw_tasks = load_multiagentbench_tasks(max_tasks=max_tasks, task_family=task_family)
        elif benchmark == "realm":
            from benchmark_wrappers.realm_bench import load_realm_tasks
            raw_tasks = load_realm_tasks(max_tasks=max_tasks, task_family=task_family)
        elif benchmark == "swebench":
            from benchmark_wrappers.swebench import load_swebench_tasks
            raw_tasks = load_swebench_tasks(max_tasks=max_tasks)
        else:
            raw_tasks = []

        return [_normalise(t, i) for i, t in enumerate(raw_tasks)][:max_tasks]

    except Exception as exc:
        log.warning(
            "Could not load %s/%s (%s). Falling back to %d synthetic tasks.",
            benchmark, task_family, exc, max_tasks,
        )
        return [
            {
                "task_id":    f"synthetic_{benchmark}_{task_family}_{i:04d}",
                "prompt": (
                    f"[Synthetic] {task_family} task #{i} for {benchmark}. "
                    "Analyse the problem step by step and provide a complete answer."
                ),
                "gold_answer": None,
            }
            for i in range(max_tasks)
        ]


# ── Run execution ─────────────────────────────────────────────────────────────

def execute_run(
    spec:       RunSpec,
    model_name: str,
    llm_judge:  Optional[str],
) -> Dict[str, Any]:
    """
    Execute one (task, topology, N, seed) cell and run the post-hoc pipeline.

    This function is the unit of work dispatched to worker processes.
    All imports are local so the function can be safely pickled and sent
    across process boundaries by ProcessPoolExecutor.

    Returns a result dict written to sweep_manifest.jsonl.
    """
    t0 = time.time()
    try:
        from dotenv import load_dotenv
        load_dotenv(ROOT / ".env")

        from langchain_openai import ChatOpenAI
        from execution.graph_runner import GraphRunner, BenchmarkTask

        # ── Participating-agent LLM ───────────────────────────────────────────
        agent_llm = ChatOpenAI(
            model=model_name,
            temperature=0.7,
            api_key=os.environ.get("OPENAI_API_KEY"),
        )

        # ── Post-hoc judge LLM (non-participating, separate model) ────────────
        # A distinct model avoids same-model correlation bias in the semantic
        # event classifier (llm_event_classifier.py).  Omitting this falls
        # back to rule-only event classification.
        judge_llm = None
        if llm_judge:
            judge_llm = ChatOpenAI(
                model=llm_judge,
                temperature=0,
                api_key=os.environ.get("OPENAI_API_KEY"),
            )

        task = BenchmarkTask(
            task_id=spec.task_id,
            benchmark=spec.benchmark,
            task_family=spec.task_family,
            prompt=spec.task_prompt,
            gold_answer=spec.gold_answer,
            difficulty="medium",
            requires_tools=(spec.benchmark in ("gaia",)),
            requires_synthesis=(
                spec.task_family in ("planning", "coordination", "reasoning")
            ),
        )

        runner = GraphRunner(
            llm=agent_llm,
            data_root=spec.run_dir.parents[3],  # data/sweep root
            model_name=model_name,
            max_steps=20,
        )

        result = runner.run(
            task=task,
            topology=spec.topology,
            num_agents=spec.num_agents,
            seed=spec.seed,
        )

        # ── Post-hoc pipeline: annotate → DAG → observables ───────────────────
        from analysis.run_pipeline import process_run
        process_run(Path(result.run_dir), llm=judge_llm)

        return {
            "run_id":       spec.run_id,
            "status":       "error" if result.error else "ok",
            "benchmark":    spec.benchmark,
            "task_family":  spec.task_family,
            "task_id":      spec.task_id,
            "topology":     spec.topology,
            "num_agents":   spec.num_agents,
            "seed":         spec.seed,
            "run_dir":      str(result.run_dir),
            "score":        result.score,
            "event_count":  result.event_count,
            "tokens_total": result.tokens_total,
            "wall_time_s":  round(time.time() - t0, 2),
            "error":        (result.error or "")[:500] or None,
            "timestamp":    _now(),
        }

    except Exception:
        return {
            "run_id":      spec.run_id,
            "status":      "fatal",
            "run_dir":     str(spec.run_dir),
            "error":       traceback.format_exc()[-800:],
            "wall_time_s": round(time.time() - t0, 2),
            "timestamp":   _now(),
        }


# ── Progress tracking ─────────────────────────────────────────────────────────

class SweepProgress:
    """Live progress tracker with JSON snapshot for monitoring."""

    def __init__(self, total: int, out_dir: Path) -> None:
        self.total   = total
        self.done    = 0
        self.ok      = 0
        self.errors  = 0
        self.skipped = 0
        self._t0     = time.time()
        self._path   = out_dir / "sweep_progress.json"

    def record(self, status: str) -> None:
        self.done += 1
        if status == "ok":       self.ok      += 1
        elif status == "skipped": self.skipped += 1
        else:                     self.errors  += 1
        self._flush()

    def _flush(self) -> None:
        elapsed = time.time() - self._t0
        rate    = self.done / elapsed if elapsed > 0 else 0.0
        eta_s   = (self.total - self.done) / rate if rate > 0 else 0.0
        self._path.write_text(json.dumps({
            "total":        self.total,
            "done":         self.done,
            "ok":           self.ok,
            "errors":       self.errors,
            "skipped":      self.skipped,
            "pct":          round(100 * self.done / self.total, 1) if self.total else 0,
            "elapsed_s":    round(elapsed),
            "eta_s":        round(eta_s),
            "rate_per_min": round(rate * 60, 1),
            "updated":      _now(),
        }, indent=2))

    def log_line(self, spec: RunSpec, status: str, extra: str = "") -> None:
        elapsed = time.time() - self._t0
        rate    = self.done / elapsed if elapsed > 0 else 0.0
        eta_m   = (self.total - self.done) / rate / 60 if rate > 0 else 0.0
        icon    = {"ok": "✓", "error": "✗", "fatal": "✗✗",
                   "skipped": "→", "dry_run": "·"}.get(status, "?")
        log.info(
            "%s [%4d/%-4d %5.1f%%  ETA %5.1f min]  "
            "%-22s N=%-4d s=%d  %s/%s/%-22s  %s",
            icon,
            self.done, self.total,
            100 * self.done / self.total if self.total else 0.0,
            eta_m,
            spec.topology, spec.num_agents, spec.seed,
            spec.benchmark, spec.task_family, spec.task_id[:22],
            extra,
        )


# ── Grid construction ─────────────────────────────────────────────────────────

def build_run_specs(
    benchmarks: List[str],
    topologies: List[str],
    scales:     List[int],
    seeds:      List[int],
    sweep_dir:  Path,
    resume:     bool,
) -> Tuple[List[RunSpec], int]:
    """
    Enumerate the full experiment grid and load required benchmark tasks.

    Returns (pending_specs, total_planned).  When resume=True, already-
    completed runs are excluded from pending_specs but counted in
    total_planned so progress percentages remain meaningful.
    """
    bench_configs = [
        (b, tf, n) for b, tf, n in BENCHMARK_CONFIGS if b in benchmarks
    ]

    # Load tasks — dedup by (benchmark, task_family) to avoid double loading
    tasks_cache: Dict[str, List[Dict[str, Any]]] = {}
    log.info("Loading benchmark tasks …")
    for benchmark, task_family, max_tasks in bench_configs:
        key = f"{benchmark}/{task_family}"
        if key not in tasks_cache:
            tasks = load_tasks(benchmark, task_family, max_tasks)
            tasks_cache[key] = tasks
            log.info("  %-32s %3d tasks", key, len(tasks))

    # Build full Cartesian product
    all_specs: List[RunSpec] = []
    for benchmark, task_family, _ in bench_configs:
        key = f"{benchmark}/{task_family}"
        for task in tasks_cache.get(key, []):
            for topology in topologies:
                for N in scales:
                    for seed in seeds:
                        run_dir = (
                            sweep_dir / benchmark / topology
                            / f"n{N}" / f"s{seed}" / task["task_id"]
                        )
                        all_specs.append(RunSpec(
                            benchmark=benchmark,
                            task_family=task_family,
                            task_id=task["task_id"],
                            task_prompt=task["prompt"],
                            gold_answer=task.get("gold_answer"),
                            topology=topology,
                            num_agents=N,
                            seed=seed,
                            run_dir=run_dir,
                        ))

    total_planned = len(all_specs)
    pending = [s for s in all_specs if not (resume and s.is_complete())]
    return pending, total_planned


# ── I/O helpers ───────────────────────────────────────────────────────────────

def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _append_jsonl(path: Path, record: Dict[str, Any]) -> None:
    with open(path, "a") as fh:
        fh.write(json.dumps(record) + "\n")


# ── CLI ───────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="run_sweep.py",
        description="LLM MAS Power-Laws — full experimental sweep.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    g = p.add_argument_group("experiment grid")
    g.add_argument(
        "--benchmarks", nargs="+",
        default=[b for b, _, _ in BENCHMARK_CONFIGS],
        choices=["gaia", "marble", "realm", "swebench"],
        metavar="BENCHMARK",
        help="Benchmarks to include.",
    )
    g.add_argument(
        "--topologies", nargs="+",
        default=ALL_TOPOLOGIES,
        choices=ALL_TOPOLOGIES,
        metavar="TOPOLOGY",
        help="Communication topologies to include.",
    )
    g.add_argument(
        "--scales", nargs="+", type=int,
        default=ALL_SCALES,
        metavar="N",
        help="Agent society sizes N.",
    )
    g.add_argument(
        "--seeds", nargs="+", type=int,
        default=ALL_SEEDS,
        metavar="SEED",
        help="Random seeds.",
    )

    e = p.add_argument_group("execution")
    e.add_argument(
        "--model", default="gpt-4o-mini",
        metavar="MODEL",
        help="LLM for participating agents.",
    )
    e.add_argument(
        "--llm-judge", default=None,
        metavar="MODEL",
        help=(
            "Separate LLM for post-hoc semantic event classification. "
            "Should differ from --model to avoid same-model correlation bias. "
            "Omit to use rule-only classification."
        ),
    )
    e.add_argument(
        "--workers", type=int, default=1,
        metavar="N",
        help="Parallel worker processes (1 = sequential).",
    )
    e.add_argument(
        "--data-root", default="data/sweep",
        metavar="PATH",
        help="Root output directory.",
    )
    e.add_argument(
        "--resume", action="store_true",
        help="Skip runs whose run_metadata.json already exists.",
    )
    e.add_argument(
        "--dry-run", action="store_true",
        help="Print the plan without making any API calls.",
    )

    return p


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> int:
    args = build_parser().parse_args()

    # Deduplicate benchmark list while preserving order
    seen: set = set()
    args.benchmarks = [b for b in args.benchmarks if not (b in seen or seen.add(b))]  # type: ignore[func-returns-value]

    sweep_dir     = ROOT / args.data_root
    manifest_path = sweep_dir / "sweep_manifest.jsonl"
    errors_path   = sweep_dir / "sweep_errors.jsonl"

    # ── Pre-flight ───────────────────────────────────────────────────────────
    if not args.dry_run:
        if not os.environ.get("OPENAI_API_KEY"):
            log.error(
                "OPENAI_API_KEY is not set.  "
                "Run:  export $(grep -v '^#' .env | xargs)"
            )
            return 1
        log.info("API key: %s…", os.environ["OPENAI_API_KEY"][:12])

    # ── Header ───────────────────────────────────────────────────────────────
    log.info("=" * 65)
    log.info("  LLM MAS Power-Laws — Experimental Sweep")
    log.info("=" * 65)
    log.info("  Model (agents) : %s", args.model)
    log.info("  Model (judge)  : %s", args.llm_judge or "rule-only")
    log.info("  Benchmarks     : %s", args.benchmarks)
    log.info("  Topologies     : %s", args.topologies)
    log.info("  Scales (N)     : %s", args.scales)
    log.info("  Seeds          : %s", args.seeds)
    log.info("  Workers        : %d", args.workers)
    log.info("  Resume         : %s", args.resume)
    log.info("  Dry run        : %s", args.dry_run)
    log.info("  Output         : %s", sweep_dir)

    # ── Grid ─────────────────────────────────────────────────────────────────
    pending, total_planned = build_run_specs(
        benchmarks=args.benchmarks,
        topologies=args.topologies,
        scales=args.scales,
        seeds=args.seeds,
        sweep_dir=sweep_dir,
        resume=args.resume,
    )
    skipped = total_planned - len(pending)
    log.info("  Planned        : %d", total_planned)
    if skipped:
        log.info("  Skipping       : %d (already complete)", skipped)
    log.info("  Queued         : %d", len(pending))

    if args.dry_run:
        log.info("")
        log.info("DRY RUN — first 20 queued specs:")
        for spec in pending[:20]:
            log.info("  %s", spec.run_id)
        if len(pending) > 20:
            log.info("  … and %d more", len(pending) - 20)
        return 0

    if not pending:
        log.info("Nothing to run.  Omit --resume to re-run existing.")
        return 0

    # ── Persist plan ─────────────────────────────────────────────────────────
    sweep_dir.mkdir(parents=True, exist_ok=True)
    (sweep_dir / "sweep_plan.json").write_text(json.dumps({
        "model":         args.model,
        "llm_judge":     args.llm_judge,
        "benchmarks":    args.benchmarks,
        "topologies":    args.topologies,
        "scales":        args.scales,
        "seeds":         args.seeds,
        "total_planned": total_planned,
        "queued":        len(pending),
        "skipped":       skipped,
        "started":       _now(),
        "sweep_dir":     str(sweep_dir),
    }, indent=2))

    # ── Execute ──────────────────────────────────────────────────────────────
    progress  = SweepProgress(len(pending), sweep_dir)
    t_launch  = time.time()
    log.info("Sweep started at %s", _now())
    log.info("-" * 65)

    def _handle(result: Dict[str, Any], spec: RunSpec) -> None:
        status = result.get("status", "error")
        extra  = ""
        if status == "ok":
            extra = (
                f"events={result.get('event_count', 0)}  "
                f"score={result.get('score')}  "
                f"{result.get('wall_time_s', 0):.1f}s"
            )
        elif result.get("error"):
            extra = result["error"].splitlines()[-1][:80]

        progress.record(status)
        progress.log_line(spec, status, extra)
        _append_jsonl(manifest_path, result)
        if status not in ("ok", "skipped"):
            _append_jsonl(errors_path, result)

    if args.workers == 1:
        # Sequential — simplest, easiest to debug, natural rate-limit compliance
        for spec in pending:
            _handle(execute_run(spec, args.model, args.llm_judge), spec)
    else:
        # Parallel — each subprocess loads its own LLM client independently
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = {
                pool.submit(execute_run, spec, args.model, args.llm_judge): spec
                for spec in pending
            }
            try:
                for future in as_completed(futures):
                    spec = futures[future]
                    try:
                        result = future.result()
                    except Exception:
                        result = {
                            "run_id":      spec.run_id,
                            "status":      "fatal",
                            "run_dir":     str(spec.run_dir),
                            "error":       traceback.format_exc()[-800:],
                            "timestamp":   _now(),
                        }
                    _handle(result, spec)
            except KeyboardInterrupt:
                log.warning("Interrupted — cancelling pending futures …")
                pool.shutdown(wait=False, cancel_futures=True)

    # ── Summary ──────────────────────────────────────────────────────────────
    elapsed = time.time() - t_launch
    log.info("-" * 65)
    log.info("Sweep complete at %s", _now())
    log.info("  Elapsed  : %.2f h  (%d s)", elapsed / 3600, int(elapsed))
    log.info("  Total    : %d", progress.done)
    log.info("  OK       : %d", progress.ok)
    log.info("  Errors   : %d", progress.errors)
    log.info("  Skipped  : %d", progress.skipped)
    log.info("  Manifest : %s", manifest_path)
    log.info("  Data     : %s", sweep_dir)

    if progress.errors:
        log.warning(
            "%d runs had errors — inspect %s for details.",
            progress.errors, errors_path,
        )

    progress._flush()
    (sweep_dir / "sweep_complete.json").write_text(json.dumps({
        "completed":  _now(),
        "total_runs": progress.done,
        "ok":         progress.ok,
        "errors":     progress.errors,
        "skipped":    progress.skipped,
        "elapsed_s":  round(elapsed),
        "model":      args.model,
        "llm_judge":  args.llm_judge,
    }, indent=2))

    return 0


if __name__ == "__main__":
    sys.exit(main())
