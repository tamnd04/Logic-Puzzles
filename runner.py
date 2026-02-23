from __future__ import annotations

import argparse
from typing import Any, Callable, Dict, List, Optional
from core.metrics import run_with_metrics
from core.search import SearchResult


def benchmark_solver(
    *,
    puzzle_name: str,
    algorithm_name: str,
    solver_fn: Callable[[], SearchResult[Any, Any]],
    repeats: int = 1,
) -> Dict[str, Any]:
    runtimes: List[float] = []
    peaks: List[float] = []
    last: Optional[SearchResult[Any, Any]] = None

    for _ in range(max(1, repeats)):
        m = run_with_metrics(solver_fn)
        last = m.result
        runtimes.append(m.runtime_sec)
        peaks.append(m.peak_mem_kb)

    assert last is not None
    return {
        "puzzle": puzzle_name,
        "algorithm": algorithm_name,
        "status": last.status,
        "path_cost": last.path_cost,
        "actions_len": len(last.actions),
        "nodes_expanded": last.stats.nodes_expanded,
        "nodes_generated": last.stats.nodes_generated,
        "max_frontier": last.stats.max_frontier_size,
        "runtime_sec_avg": sum(runtimes) / len(runtimes),
        "runtime_sec_min": min(runtimes),
        "runtime_sec_max": max(runtimes),
        "peak_mem_kb_avg": sum(peaks) / len(peaks),
        "peak_mem_kb_max": max(peaks),
        "repeats": len(runtimes),
    }


def print_rows(rows: List[Dict[str, Any]]) -> None:
    if not rows:
        print("(no rows)")
        return
    cols = list(rows[0].keys())
    widths = {c: max(len(str(c)), max(len(str(r.get(c, ""))) for r in rows)) for c in cols}

    def fmt_row(r: Dict[str, Any]) -> str:
        return " | ".join(str(r.get(c, "")).ljust(widths[c]) for c in cols)

    print(fmt_row({c: c for c in cols}))
    print("-+-".join("-" * widths[c] for c in cols))
    for r in rows:
        print(fmt_row(r))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run logic puzzle solvers (Sudoku or Futoshiki) with various algorithms."
    )
    
    parser.add_argument(
        "game",
        type=str,
        choices=["sudoku", "futoshiki"],
        help="Choose which game to solve: 'sudoku' or 'futoshiki'"
    )
    
    parser.add_argument(
        "--step-by-step",
        action="store_true",
        help="Show the solving process step-by-step"
    )
    
    parser.add_argument(
        "--num-puzzles",
        type=int,
        default=1,
        help="Number of randomly generated puzzles to solve (default: 1)"
    )
    
    args = parser.parse_args()
    
    print(f"Game: {args.game}")
    print(f"Step-by-step mode: {args.step_by_step}")
    print(f"Number of puzzles: {args.num_puzzles}")
    print()
    
    if args.game == "sudoku":
        print("Put sudoku solver function here")
    elif args.game == "futoshiki":
        print("Put sudoku solver function here")


if __name__ == "__main__":
    main()

