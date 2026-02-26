"""
Futoshiki solvers using search algorithms (DFS, A*).
"""

from __future__ import annotations
from typing import Iterable, List, Optional, Set, Tuple
from core.search import SearchProblem, SearchResult, dfs, astar
# If your search.py is in the same directory, use: from search import SearchProblem, ...


# ---------------------------------------------------------------------------
# State type: tuple of ints, row-major, 0 = empty
# ---------------------------------------------------------------------------
FutoshikiState = Tuple[int, ...]


def _candidates(
    state: FutoshikiState, 
    idx: int, 
    size: int, 
    constraints: Set[Tuple[Tuple[int, int], Tuple[int, int]]]
) -> List[int]:
    """Return valid digits (1 to size) for the empty cell at idx based on Latin square rules and inequalities."""
    r, c = divmod(idx, size)
    
    # 1. Check Row and Column constraints (Latin Square rules)
    used = set()
    for i in range(size):
        # Row peers
        if i != c:
            val = state[r * size + i]
            if val != 0:
                used.add(val)
        # Column peers
        if i != r:
            val = state[i * size + c]
            if val != 0:
                used.add(val)

    possible = [d for d in range(1, size + 1) if d not in used]
    
    # 2. Check Inequality constraints
    valid = []
    for num in possible:
        is_ok = True
        for (r1, c1), (r2, c2) in constraints:
            # If the constraint applies to the cell we are currently trying to fill
            if r1 == r and c1 == c:
                val2 = state[r2 * size + c2]
                if val2 != 0 and num >= val2:
                    is_ok = False
                    break
            elif r2 == r and c2 == c:
                val1 = state[r1 * size + c1]
                if val1 != 0 and val1 >= num:
                    is_ok = False
                    break
        
        if is_ok:
            valid.append(num)
            
    return valid


# ---------------------------------------------------------------------------
# SearchProblem implementation
# ---------------------------------------------------------------------------

class FutoshikiProblem(SearchProblem[FutoshikiState, Tuple[int, int]]):
    """
    State : tuple of ints (0 = empty), row-major.
    Action: (cell_index, digit)
    Cost  : each assignment costs 1.
    """

    def __init__(
        self, 
        board: FutoshikiState, 
        size: int, 
        constraints: Set[Tuple[Tuple[int, int], Tuple[int, int]]]
    ) -> None:
        self._initial = board
        self.size = size
        self.constraints = constraints

    def initial_state(self) -> FutoshikiState:
        return self._initial

    def is_goal(self, state: FutoshikiState) -> bool:
        return 0 not in state

    def successors(
        self, state: FutoshikiState
    ) -> Iterable[Tuple[Tuple[int, int], FutoshikiState, float]]:
        # Pick the first empty cell (leftmost)
        try:
            idx = state.index(0)
        except ValueError:
            return  # no empty cells

        lst = list(state)
        for digit in _candidates(state, idx, self.size, self.constraints):
            lst[idx] = digit
            yield (idx, digit), tuple(lst), 1.0
            lst[idx] = 0


class FutoshikiProblemMRV(FutoshikiProblem):
    """Same as FutoshikiProblem but successors pick the cell with fewest candidates (MRV)."""

    def successors(
        self, state: FutoshikiState
    ) -> Iterable[Tuple[Tuple[int, int], FutoshikiState, float]]:
        # Find the empty cell with the minimum remaining values
        best_idx: Optional[int] = None
        best_cands: List[int] = []

        for i, v in enumerate(state):
            if v == 0:
                cands = _candidates(state, i, self.size, self.constraints)
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

def heuristic_zero(state: FutoshikiState) -> float:
    """Trivial admissible heuristic (turns A* into uniform-cost / Dijkstra)."""
    return 0.0


def heuristic_mrv(state: FutoshikiState) -> float:
    """
    Admissible heuristic: number of empty cells.
    Each empty cell needs at least one more assignment, so this never over-estimates.
    """
    return float(state.count(0))


# ---------------------------------------------------------------------------
# Four public solver functions
# ---------------------------------------------------------------------------

def solve_dfs(board: FutoshikiState, size: int, constraints: set) -> SearchResult:
    """DFS without a visited set."""
    problem = FutoshikiProblem(board, size, constraints)
    return dfs(problem, use_visited=False)


def solve_dfs_with_visited(board: FutoshikiState, size: int, constraints: set) -> SearchResult:
    """DFS with a visited set (avoids revisiting identical board states)."""
    problem = FutoshikiProblem(board, size, constraints)
    return dfs(problem, use_visited=True)


def solve_astar(board: FutoshikiState, size: int, constraints: set) -> SearchResult:
    """A* with h=0 (equivalent to Dijkstra / uniform-cost search)."""
    problem = FutoshikiProblem(board, size, constraints)
    return astar(problem, heuristic=heuristic_zero)


def solve_astar_mrv(board: FutoshikiState, size: int, constraints: set) -> SearchResult:
    """A* with MRV heuristic and MRV-guided successor ordering."""
    problem = FutoshikiProblemMRV(board, size, constraints)
    return astar(problem, heuristic=heuristic_mrv)