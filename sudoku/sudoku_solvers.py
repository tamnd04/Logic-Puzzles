"""
Sudoku solvers using search algorithms (DFS, A*).
"""

from __future__ import annotations
from typing import Iterable, List, Optional, Tuple
from core.search import SearchProblem, SearchResult, dfs, astar


# ---------------------------------------------------------------------------
# State type: tuple of 81 ints, row-major, 0 = empty
# ---------------------------------------------------------------------------
SudokuState = Tuple[int, ...]


def _peers(idx: int) -> List[int]:
    """Return all indices that share a row, column, or 3x3 box with idx."""
    r, c = divmod(idx, 9)
    row = [r * 9 + cc for cc in range(9)]
    col = [rr * 9 + c for rr in range(9)]
    br, bc = (r // 3) * 3, (c // 3) * 3
    box = [( br + dr) * 9 + (bc + dc) for dr in range(3) for dc in range(3)]
    peers = set(row + col + box)
    peers.discard(idx)
    return list(peers)


# Pre-compute peers for all 81 cells
_PEERS: List[List[int]] = [_peers(i) for i in range(81)]


def _candidates(state: SudokuState, idx: int) -> List[int]:
    """Return valid digits (1-9) for empty cell at idx."""
    used = {state[p] for p in _PEERS[idx] if state[p] != 0}
    return [d for d in range(1, 10) if d not in used]


# ---------------------------------------------------------------------------
# SearchProblem implementation
# ---------------------------------------------------------------------------

class SudokuProblem(SearchProblem[SudokuState, Tuple[int, int]]):
    """
    State : tuple of 81 ints (0 = empty), row-major.
    Action: (cell_index, digit)
    Cost  : each assignment costs 1.
    """

    def __init__(self, board: SudokuState) -> None:
        self._initial = board

    def initial_state(self) -> SudokuState:
        return self._initial

    def is_goal(self, state: SudokuState) -> bool:
        return 0 not in state

    def successors(
        self, state: SudokuState
    ) -> Iterable[Tuple[Tuple[int, int], SudokuState, float]]:
        # Pick the first empty cell (leftmost)
        try:
            idx = state.index(0)
        except ValueError:
            return  # no empty cells

        lst = list(state)
        for digit in _candidates(state, idx):
            lst[idx] = digit
            yield (idx, digit), tuple(lst), 1.0
            lst[idx] = 0


class SudokuProblemMRV(SudokuProblem):
    """Same as SudokuProblem but successors pick the cell with fewest candidates (MRV)."""

    def successors(
        self, state: SudokuState
    ) -> Iterable[Tuple[Tuple[int, int], SudokuState, float]]:
        # Find the empty cell with the minimum remaining values
        best_idx: Optional[int] = None
        best_cands: List[int] = []

        for i, v in enumerate(state):
            if v == 0:
                cands = _candidates(state, i)
                if not cands:
                    return  # dead end – yield nothing
                if best_idx is None or len(cands) < len(best_cands):
                    best_idx = i
                    best_cands = cands
                    if len(best_cands) == 1:
                        break  # can't do better

        if best_idx is None:
            return  # no empty cells

        lst = list(state)
        for digit in best_cands:
            lst[best_idx] = digit
            yield (best_idx, digit), tuple(lst), 1.0
            lst[best_idx] = 0


# ---------------------------------------------------------------------------
# Heuristics for A*
# ---------------------------------------------------------------------------

def heuristic_zero(state: SudokuState) -> float:
    """Trivial admissible heuristic (turns A* into uniform-cost / Dijkstra)."""
    return 0.0


def heuristic_mrv(state: SudokuState) -> float:
    """
    Admissible heuristic: number of empty cells.
    Each empty cell needs at least one more assignment, so this never over-estimates.
    (A tighter but still admissible choice is simply the count of unfilled cells.)
    """
    return float(state.count(0))


# ---------------------------------------------------------------------------
# Four public solver functions
# ---------------------------------------------------------------------------

def solve_dfs(board: SudokuState) -> SearchResult:
    """DFS without a visited set."""
    problem = SudokuProblem(board)
    return dfs(problem, use_visited=False)


def solve_dfs_with_visited(board: SudokuState) -> SearchResult:
    """DFS with a visited set (avoids revisiting identical board states)."""
    problem = SudokuProblem(board)
    return dfs(problem, use_visited=True)


def solve_astar(board: SudokuState) -> SearchResult:
    """A* with h=0 (equivalent to Dijkstra / uniform-cost search)."""
    problem = SudokuProblem(board)
    return astar(problem, heuristic=heuristic_zero)


def solve_astar_mrv(board: SudokuState) -> SearchResult:
    """A* with MRV heuristic and MRV-guided successor ordering."""
    problem = SudokuProblemMRV(board)
    return astar(problem, heuristic=heuristic_mrv)
