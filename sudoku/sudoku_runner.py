"""
Sudoku runner module - called from the main runner.py with parsed CLI args.
"""

from core.io import load_sudoku
from core.benchmark import benchmark_solver, print_rows
from sudoku.sudoku_helpers import board_to_str
from sudoku.sudoku_generator import create_puzzle
from sudoku.sudoku_solvers import (
    solve_astar_mrv,
    solve_astar,
    solve_dfs,
    solve_dfs_with_visited,
)

SOLVERS = {
    "dfs"               : solve_dfs,
    "dfs_with_visited"  : solve_dfs_with_visited,
    "astar"             : solve_astar,
    "astar_mrv"         : solve_astar_mrv,
}


def run(
    input_path=None,
    count: int = 1,
    step_by_step: bool = False,
    algorithms: list[str] | None = None,
):
    """
    Run Sudoku solver benchmarks.

    Args:
        input_path   : Path to an input puzzle file. If None, random puzzles are generated.
        count        : Number of random puzzles to generate (ignored if input_path is provided).
        step_by_step : Whether to display the step-by-step solution.
        algorithms   : List of algorithm names to run. If None, all 4 are used.
    """
    # Resolve which solvers to use
    if algorithms is None:
        solvers = list(SOLVERS.items())
    else:
        solvers = []
        for name in algorithms:
            if name not in SOLVERS:
                raise ValueError(
                    f"Unknown algorithm '{name}'. "
                    f"Available: {list(SOLVERS.keys())}"
                )
            solvers.append((name, SOLVERS[name]))

    # Build list of puzzle boards
    boards: list[tuple[str, tuple]] = []
    if input_path is not None:
        grid = load_sudoku(input_path)
        flat = tuple(cell for row in grid for cell in row)
        boards.append((str(input_path), flat))
    else:
        for i in range(count):
            grid = create_puzzle(blanks=40)
            flat = tuple(cell for row in grid for cell in row)
            boards.append((f"random_{i + 1}", flat))

    # Run each solver on each puzzle and collect benchmark rows
    rows: list[dict] = []

    for puzzle, state in boards:
        print("\n" + "-" * 30)
        print(f"Input: {puzzle}")
        print("-" * 30)
        print(board_to_str(state))
        print()

        for algorithm, solver_function in solvers:
            row = benchmark_solver(
                puzzle_name         = puzzle,
                algorithm_name      = algorithm,
                solver_fn=lambda fn = solver_function,
                s                   = state: fn(s),
                repeats             = 1,
            )
            rows.append(row)

            # if step_by_step:
            if False:
                result = solver_function(state)
                print(f"\n  [{algorithm}] Solution:")
                if result.actions:
                    lst = list(state)
                    for idx, digit in result.actions:
                        lst[idx] = digit
                    print(board_to_str(tuple(lst)))
            else:
                print(
                    f"  [{algorithm}]"
                    f"{'':<{20 - len(algorithm)}}"
                    f"status={row['status']:<10} "
                    f"nodes_expanded={row['nodes_expanded']:<8} "
                    f"runtime={row['runtime_sec_avg']:.4f}s"
                )

    # Print summary table
    if rows:
        print("\n" + "=" * 40)
        print("Benchmark Summary")
        print("=" * 40)
        print_rows(rows)

    return rows