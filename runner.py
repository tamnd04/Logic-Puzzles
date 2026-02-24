"""
Main entry point for running logic puzzle solvers (Sudoku or Futoshiki) with various algorithms.

Usage examples:
    # Solve a Sudoku puzzle from a file, showing step-by-step process
    python runner.py --sudoku --input-path sudoku/sudoku1.txt --step-by-step

    # Solve 5 random Sudoku puzzles without step-by-step
    python runner.py --sudoku --count 5

    # Solve a Futoshiki puzzle from a file, showing step-by-step process
    python runner.py --futoshiki --input-path futoshiki/futoshiki1.txt --step-by-step

    # More options:
    python runner.py --help
"""

from __future__ import annotations

import argparse
from core.benchmark import benchmark_solver, print_rows
from pathlib import Path
import sudoku.sudoku_runner as sudoku_runner


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

    print(); print("=" * 40)

    if args.sudoku:
        print("Puzzle type       : Sudoku")
    elif args.futoshiki:
        print("Puzzle type       : Futoshiki")

    if args.count is not None:
        print(f"Number of puzzles : {args.count}")
    if args.input_path is not None:
        print(f"Input path        : {args.input_path}")

    print(f"Step-by-step mode : {args.step_by_step}")

    print("=" * 40); print()

    #---------------------------------------------------------------------------
    # Sudoku
    #---------------------------------------------------------------------------
    
    if args.sudoku:
        sudoku_runner.run(
            input_path   = args.input_path,
            count        = args.count or 1,
            step_by_step = args.step_by_step,
            algorithms   = ["dfs", "dfs_with_visited", "astar", "astar_mrv"]
        )

    #---------------------------------------------------------------------------
    # Futoshiki
    #---------------------------------------------------------------------------
    
    elif args.futoshiki:
        print("TODO: Call Futoshiki solver module")


if __name__ == "__main__":
    main()

