from __future__ import annotations

import argparse
from typing import Any, Callable, Dict, List, Optional
from core.metrics import run_with_metrics
from core.search import SearchResult
from pathlib import Path


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

    # ----------------------------------------------------------------------------
    # Argument parsing
    # ----------------------------------------------------------------------------

    parser = argparse.ArgumentParser(
        description="Run logic puzzle solvers (Sudoku or Futoshiki) with various algorithms."
    )
    
    # Sudoku or Futoshiki
    puzzle_group = parser.add_mutually_exclusive_group(required=True)
    puzzle_group.add_argument(
        "--sudoku",
        action="store_true",
        help="Solve a Sudoku puzzle. Either this or --futoshiki must be specified."
    )
    puzzle_group.add_argument(
        "--futoshiki",
        action="store_true",
        help="Solve a Futoshiki puzzle. Either this or --sudoku must be specified."
    )
    
    # Random input or file input
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--count",
        type=int,
        help="Number of randomly generated puzzles to solve. Either this or --input-path must be specified, not both."
    )
    input_group.add_argument(
        "--input-path",
        type=Path,
        help="Path to a text file containing the input puzzle. Either this or --count must be specified, not both."
    )
    
    # Display step-by-step or not
    parser.add_argument(
        "--step-by-step",
        action="store_true",
        help="Show the solving process step-by-step"
    )
    
    args = parser.parse_args()
    
    #----------------------------------------------------------------------------
    # Print configuration
    #----------------------------------------------------------------------------

    print(); print("-" * 40)

    if args.sudoku:
        print("Puzzle type       : Sudoku")
    elif args.futoshiki:
        print("Puzzle type       : Futoshiki")

    if args.count is not None:
        print(f"Number of puzzles : {args.count}")
    if args.input_path is not None:
        print(f"Input path        : {args.input_path}")

    print(f"Step-by-step mode : {args.step_by_step}")

    print("-" * 40); print()

    #---------------------------------------------------------------------------
    # Sudoku
    #---------------------------------------------------------------------------
    
    if args.sudoku:
        print("TODO: Call Sudoku solver module")
        
    #---------------------------------------------------------------------------
    # Futoshiki
    #---------------------------------------------------------------------------
    
    elif args.futoshiki:
        print("TODO: Call Futoshiki solver module")


if __name__ == "__main__":
    main()

