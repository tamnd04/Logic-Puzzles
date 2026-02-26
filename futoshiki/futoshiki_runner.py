"""
Futoshiki runner module - called from the main runner.py with parsed CLI args.
"""

from pathlib import Path
from core.benchmark import benchmark_solver, print_rows
from futoshiki.futoshiki_helpers import board_to_str, parse_board
from futoshiki.futoshiki_generator import generate_futoshiki
from futoshiki.futoshiki_solvers import (
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
    size: int = 5,  # Added size parameter for generation
    step_by_step: bool = False,
    algorithms: list[str] | None = None,
):
    """
    Run Futoshiki solver benchmarks.

    Args:
        input_path   : Path to an input puzzle file. If None, random puzzles are generated.
        count        : Number of random puzzles to generate (ignored if input_path is provided).
        size         : The grid size to use when generating random puzzles.
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

    # Build list of puzzle boards: Tuple containing (name, flat_state, size, constraints)
    boards: list[tuple[str, tuple, int, set]] = []
    
    if input_path is not None:
        with open(input_path, 'r') as f:
            text = f.read()
        b_size, flat, constraints = parse_board(text)
        boards.append((str(input_path), flat, b_size, constraints))
    else:
        for i in range(count):
            grid, _, constraints = generate_futoshiki(size=size)
            flat = tuple(cell for row in grid for cell in row)
            boards.append((f"random_{size}x{size}_{i + 1}", flat, size, constraints))

    # Run each solver on each puzzle and collect benchmark rows
    rows: list[dict] = []
    
    # Track if we should write step-by-step output files
    write_step_by_step_files = step_by_step

    for puzzle_idx, (puzzle, state, b_size, constraints) in enumerate(boards):
        print("\n" + "-" * 30)
        print(f"Input: {puzzle}")
        print("-" * 30)
        print(board_to_str(state, b_size, constraints))
        print()
        
        # Determine if we should write files for this puzzle
        should_write_files = write_step_by_step_files and (
            input_path is not None or puzzle_idx == 0
        )

        for algorithm, solver_function in solvers:
            # We use a lambda to cleanly inject the extra parameters 
            # (size, constraints) required by Futoshiki solvers
            row = benchmark_solver(
                puzzle_name         = puzzle,
                algorithm_name      = algorithm,
                solver_fn=lambda fn = solver_function,
                s                   = state,
                sz                  = b_size,
                c                   = constraints: fn(s, sz, c),
                repeats             = 1,
            )
            rows.append(row)

            if step_by_step:
                result = solver_function(state, b_size, constraints)
                print(f"\n  [{algorithm}] Step-by-step solution:")
                if result.states:
                    for i, board_state in enumerate(result.states):
                        print(f"\n    Step {i}:")
                        print(board_to_str(board_state, b_size, constraints))
                    print(f"\n  Total steps: {len(result.states)}")
                    
                    if should_write_files:
                        results_dir = Path("results")
                        results_dir.mkdir(exist_ok=True)
                        
                        output_filename = results_dir / f"futoshiki_{algorithm}_steps.txt"
                        
                        with open(output_filename, 'w') as f:
                            f.write(f"Puzzle: {puzzle}\n")
                            f.write(f"Algorithm: {algorithm}\n")
                            f.write(f"Status: {result.status}\n")
                            f.write(f"Total steps: {len(result.states)}\n")
                            f.write(f"Nodes expanded: {row['nodes_expanded']}\n")
                            f.write(f"Nodes generated: {row['nodes_generated']}\n")
                            f.write(f"Max frontier: {row['max_frontier']}\n")
                            f.write(f"Runtime: {row['runtime_sec_avg']:.4f}s\n")
                            f.write(f"Memory: {row['peak_mem_kb_avg']:.2f}KB\n")
                            f.write("\n" + "=" * 50 + "\n")
                            f.write("Step-by-step solution:\n")
                            f.write("=" * 50 + "\n\n")
                            
                            for i, board_state in enumerate(result.states):
                                f.write(f"Step {i}:\n")
                                f.write(board_to_str(board_state, b_size, constraints))
                                f.write("\n\n")
                        
                        print(f"  Output written to: {output_filename}")
                else:
                    print("  (Step-by-step tracking not available)")
            else:
                print(
                    f"  [{algorithm}]"
                    f"{'':<{20 - len(algorithm)}}"
                    f"status={row['status']:<10} "
                    f"nodes_expanded={row['nodes_expanded']:<8} "
                    f"runtime={row['runtime_sec_avg']:.4f}s     "
                    f"memory={row['peak_mem_kb_avg']:.2f}KB"
                )

    # Aggregate results by algorithm across all puzzles
    if rows:
        algo_data = {}
        for row in rows:
            algo = row['algorithm']
            if algo not in algo_data:
                algo_data[algo] = {
                    'nodes_expanded': [],
                    'nodes_generated': [],
                    'max_frontier': [],
                    'runtime_sec_avg': [],
                    'peak_mem_kb_avg': [],
                    'actions_len': [],
                    'status': row['status']
                }
            algo_data[algo]['nodes_expanded'].append(row['nodes_expanded'])
            algo_data[algo]['nodes_generated'].append(row['nodes_generated'])
            algo_data[algo]['max_frontier'].append(row['max_frontier'])
            algo_data[algo]['runtime_sec_avg'].append(row['runtime_sec_avg'])
            algo_data[algo]['peak_mem_kb_avg'].append(row['peak_mem_kb_avg'])
            algo_data[algo]['actions_len'].append(row['actions_len'])
        
        aggregated_rows = []
        for algo, data in algo_data.items():
            aggregated_rows.append({
                'algorithm': algo,
                'status': data['status'],
                'actions_len': sum(data['actions_len']) / len(data['actions_len']),
                'nodes_expanded': sum(data['nodes_expanded']) / len(data['nodes_expanded']),
                'nodes_generated': sum(data['nodes_generated']) / len(data['nodes_generated']),
                'max_frontier': max(data['max_frontier']),
                'runtime_sec_avg': sum(data['runtime_sec_avg']) / len(data['runtime_sec_avg']),
                'runtime_sec_min': min(data['runtime_sec_avg']),
                'runtime_sec_max': max(data['runtime_sec_avg']),
                'peak_mem_kb_avg': sum(data['peak_mem_kb_avg']) / len(data['peak_mem_kb_avg']),
                'peak_mem_kb_min': min(data['peak_mem_kb_avg']),
                'peak_mem_kb_max': max(data['peak_mem_kb_avg']),
            })

        print("\n" + "=" * 40)
        print("Benchmark Summary")
        print("=" * 40)
        print_rows(aggregated_rows)

    return rows