# Logic Puzzles Project – Outline

This project includes:
- `core/search.py`: DFS, and A* (closed set + re-open)
- `core/metrics.py`: runtime + peak Python memory measurement
- `core/io.py`: input parsers (Sudoku + Futoshiki) + CSV writer
- `runner.py`: benchmarking helper that produces rows for report tables
- `sudoku/`: Sudoku SearchProblem + DFS/A* solvers
- `futoshiki/`: Futoshiki SearchProblem + DFS/A* solvers
- `examples/`: sample input files

## Sudoku input
9 lines of 9 digits, `0` means empty.

Example:
530070000
600195000
098000060
800060003
400803001
700020006
060000280
000419005
000080079

### Futoshiki input
- First line: N
- Next N lines: N integers (0 = empty)
- Remaining lines: inequalities in edge-list form, 1-based:
  r1 c1 < r2 c2
  r1 c1 > r2 c2

Example:
  4
  0 0 0 0
  0 0 0 0
  0 0 0 0
  0 0 0 0
  1 1 < 1 2
  2 3 > 3 3

## Notes on states
For the generic search functions, `state` should be hashable if enable `use_visited=True`.
Common choices:
- Sudoku state: tuple of 81 ints
- Futoshiki state: tuple of N*N ints

## Report table columns
- puzzle, algorithm, status
- runtime_sec_avg, peak_mem_kb_avg
- nodes_expanded, nodes_generated, max_frontier
- actions_len (or assignments count)
