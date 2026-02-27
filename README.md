# Logic Puzzles Framework

A comprehensive framework for solving logic puzzles (Sudoku and Futoshiki) using various search algorithms including DFS and A* with different heuristics.

## Overview

This project provides:
- Generic search algorithms (DFS, A*)
- Performance benchmarking with runtime and memory tracking
- Step-by-step solution visualization
- Support for Sudoku and Futoshiki puzzles
- CSV export and file output capabilities

## Project Structure

```
logic_puzzles_framework/
├── core/
│   ├── __init__.py
│   ├── benchmark.py      # Benchmarking utilities and result aggregation
│   ├── io.py            # Input parsers (Sudoku, Futoshiki) and output writers
│   ├── metrics.py       # Runtime and memory measurement utilities
│   └── search.py        # Generic search algorithms (DFS, A*)
│
├── sudoku/
│   ├── sudoku_generator.py   # Random Sudoku puzzle generation
│   ├── sudoku_helpers.py     # Board display and validation utilities
│   ├── sudoku_input.txt      # Sample Sudoku input file
│   ├── sudoku_runner.py      # Sudoku-specific runner and benchmarking
│   └── sudoku_solvers.py     # Sudoku SearchProblem implementations
│
├── futoshiki/
│   ├── futoshiki_generator.py   # Random Futoshiki puzzle generation
│   ├── futoshiki_helpers.py     # Board display and validation utilities
│   ├── futoshiki_input.txt      # Sample Futoshiki input file
│   ├── futoshiki_runner.py      # Futoshiki-specific runner and benchmarking
│   └── futoshiki_solvers.py     # Futoshiki SearchProblem implementations
│
├── results/              # Output directory for step-by-step solutions
│   ├── sudoku_dfs_steps.txt
│   ├── sudoku_astar_steps.txt
│   └── ...
│
├── runner.py            # Main entry point with CLI argument parsing
├── README.md
└── .gitignore
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/tamnd04/Logic-Puzzles.git
cd Logic-Puzzles
```

2. Create and activate a virtual environment (optional but recommended):
```bash
python -m venv .venv
# On Windows PowerShell:
.venv\Scripts\Activate.ps1
# On Windows CMD:
.venv\Scripts\activate.bat
# On Linux/Mac:
source .venv/bin/activate
```

3. Install required dependencies:
```bash
pip install py-sudoku-maker
```

## How to Run

### Basic Usage

The main entry point is `runner.py`. You must specify either `--sudoku` or `--futoshiki`, and either `--input-path` or `--count`.

**Solve a specific puzzle from a file:**
```bash
python runner.py --sudoku --input-path sudoku/sudoku_input.txt
python runner.py --futoshiki --input-path futoshiki/futoshiki_input.txt
```

**Generate and solve random puzzles:**
```bash
python runner.py --sudoku --count 5
python runner.py --futoshiki --count 3
```

**Enable step-by-step visualization:**
```bash
python runner.py --sudoku --input-path sudoku/sudoku_input.txt --step-by-step
python runner.py --futoshiki --count 5 --step-by-step
```

### Command-Line Arguments

- `--sudoku` or `--futoshiki`: Choose puzzle type (required, mutually exclusive)
- `--input-path PATH`: Path to input file (mutually exclusive with `--count`)
- `--count N`: Generate and solve N random puzzles (mutually exclusive with `--input-path`)
- `--step-by-step`: Show detailed step-by-step solution process and save to files

### Step-by-Step Output Files

When `--step-by-step` is enabled:
- **With `--input-path`**: Creates 4 output files (one per algorithm) in `results/` directory
- **With `--count`**: Creates 4 output files for the first puzzle only

Output files are named: `{game}_{algorithm}_steps.txt`

Example files:
- `results/sudoku_dfs_steps.txt`
- `results/sudoku_dfs_with_visited_steps.txt`
- `results/sudoku_astar_steps.txt`
- `results/sudoku_astar_mrv_steps.txt`

## Algorithms

The framework implements 4 search algorithms for each puzzle type:

1. **DFS (Depth-First Search)**: Basic depth-first search without visited tracking
2. **DFS with Visited**: DFS with duplicate state detection
3. **A\* (A-star)**: A* search with admissible heuristic (e.g., number of empty cells)
4. **A\* MRV (Minimum Remaining Values)**: A* with MRV heuristic for variable ordering

## Benchmark Summary Columns

After running the solver, a benchmark summary table is displayed with the following columns:

| Column | Description |
|--------|-------------|
| **algorithm** | Name of the search algorithm (dfs, dfs_with_visited, astar, astar_mrv) |
| **status** | Solution status (solved, timeout, failure, cutoff) |
| **actions_len** | Average number of actions/moves in the solution path |
| **nodes_expanded** | Average number of nodes expanded during search |
| **nodes_generated** | Average number of nodes generated (children created) |
| **max_frontier** | Maximum frontier size across all puzzles (peak memory usage indicator) |
| **runtime_sec_avg** | Average runtime in seconds |
| **runtime_sec_min** | Minimum runtime in seconds |
| **runtime_sec_max** | Maximum runtime in seconds |
| **peak_mem_kb_avg** | Average peak memory usage in kilobytes |
| **peak_mem_kb_min** | Minimum peak memory usage in kilobytes |
| **peak_mem_kb_max** | Maximum peak memory usage in kilobytes |

### Metrics Explanation

- **actions_len**: For Sudoku/Futoshiki, this represents the number of cells filled to reach the solution
- **nodes_expanded**: Total states explored by the algorithm
- **nodes_generated**: Total child states created (includes duplicates)
- **max_frontier**: Peak number of nodes in the open list (stack for DFS, priority queue for A*)
  - Lower values indicate more memory-efficient search
  - Shows the maximum across all puzzles (not averaged)
- **Runtime metrics**: Show average, minimum, and maximum execution times across multiple puzzles
- **Memory metrics**: Show average, minimum, and maximum memory consumption across multiple puzzles

## Input Format

### Sudoku Input

9 lines of 9 digits where `0` represents an empty cell:

```
530070000
600195000
098000060
800060003
400803001
700020006
060000280
000419005
000080079
```

### Futoshiki Input

A visual grid format where:
- Numbers represent cell values (0 = empty)
- `<` and `>` symbols between numbers show horizontal inequalities
- `v` symbols between rows show vertical inequalities (pointing downward)
- Empty lines separate rows

Example (5×5 board):
```
0   5   0 < 0   0
            v   v
0   0   0   2   0
        v   v
3   0 > 0 > 0   0
    v
0   0   0   0 < 0

0   2   0   0   0
```

This represents:
- Cell (1,2) has value 5
- Cell (1,3) < Cell (1,4)
- Cells (1,4) and (1,5) have downward inequalities (v)
- Cell (2,4) has value 2
- And so on...

## State Representation

For the generic search functions, states must be hashable when using `use_visited=True`:

- **Sudoku state**: Tuple of 81 integers (flattened 9×9 board)
- **Futoshiki state**: Tuple of N×N integers (flattened N×N board)

## Performance Notes

- **DFS**: Fast but may not find optimal solution; memory efficient
- **DFS with Visited**: Avoids revisiting states; better for graphs with cycles
- **A\***: Guarantees optimal solution with admissible heuristic; higher memory usage
- **A\* MRV**: More informed heuristic; often expands fewer nodes than basic A*

## License

This project is for educational purposes.
