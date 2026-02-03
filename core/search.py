from __future__ import annotations

from dataclasses import dataclass
from typing import (
    Callable,
    Deque,
    Dict,
    Generic,
    Hashable,
    Iterable,
    List,
    Optional,
    Protocol,
    Set,
    Tuple,
    TypeVar,
)
from collections import deque
import heapq


S = TypeVar("S", bound=Hashable)  # State must be hashable if you want visited-set graph search
A = TypeVar("A")                  # Action can be any type (str, tuple, custom object)


class SearchProblem(Protocol[S, A]):
    """Minimal interface for generic search algorithms."""

    def initial_state(self) -> S: ...
    def is_goal(self, state: S) -> bool: ...
    def successors(self, state: S) -> Iterable[Tuple[A, S, float]]:
        """Yield (action, next_state, step_cost)."""
        ...


@dataclass
class Node(Generic[S, A]):
    state: S
    parent: Optional["Node[S, A]"] = None
    action: Optional[A] = None
    path_cost: float = 0.0
    depth: int = 0


@dataclass(frozen=True)
class SearchStats:
    nodes_expanded: int
    nodes_generated: int
    max_frontier_size: int


@dataclass(frozen=True)
class SearchResult(Generic[S, A]):
    status: str                  # "solved" | "timeout" | "cutoff" | "failure"
    goal_state: Optional[S]
    actions: List[A]
    path_cost: float
    stats: SearchStats


def _reconstruct_actions_from_node(node: Node[S, A]) -> List[A]:
    actions: List[A] = []
    cur: Optional[Node[S, A]] = node
    while cur is not None and cur.parent is not None:
        if cur.action is not None:
            actions.append(cur.action)
        cur = cur.parent
    actions.reverse()
    return actions


def _reconstruct_actions_from_parents(
    parents: Dict[S, Tuple[S, A]],
    start: S,
    goal: S,
) -> List[A]:
    actions: List[A] = []
    cur = goal
    while cur != start:
        prev, act = parents[cur]
        actions.append(act)
        cur = prev
    actions.reverse()
    return actions

def dfs(
    problem: SearchProblem[S, A],
    *,
    use_visited: bool = False,
    depth_limit: Optional[int] = None,
    node_limit: Optional[int] = None,
) -> SearchResult[S, A]:
    """Depth-first search (stack-based). Optionally depth-limited."""
    start = Node(problem.initial_state())
    if problem.is_goal(start.state):
        return SearchResult(
            status="solved",
            goal_state=start.state,
            actions=[],
            path_cost=0.0,
            stats=SearchStats(nodes_expanded=0, nodes_generated=1, max_frontier_size=1),
        )

    stack: List[Node[S, A]] = [start]
    visited: Set[S] = set([start.state]) if use_visited else set()

    nodes_generated = 1
    nodes_expanded = 0
    max_frontier = 1

    cutoff_happened = False

    while stack:
        max_frontier = max(max_frontier, len(stack))
        node = stack.pop()
        nodes_expanded += 1

        if node_limit is not None and nodes_expanded > node_limit:
            return SearchResult(
                status="timeout",
                goal_state=None,
                actions=[],
                path_cost=float("inf"),
                stats=SearchStats(nodes_expanded, nodes_generated, max_frontier),
            )

        if depth_limit is not None and node.depth >= depth_limit:
            cutoff_happened = True
            continue

        for action, nxt, step_cost in problem.successors(node.state):
            child = Node(
                state=nxt,
                parent=node,
                action=action,
                path_cost=node.path_cost + step_cost,
                depth=node.depth + 1,
            )
            nodes_generated += 1

            if not use_visited or child.state not in visited:
                if use_visited:
                    visited.add(child.state)

                if problem.is_goal(child.state):
                    return SearchResult(
                        status="solved",
                        goal_state=child.state,
                        actions=_reconstruct_actions_from_node(child),
                        path_cost=child.path_cost,
                        stats=SearchStats(nodes_expanded, nodes_generated, max_frontier),
                    )
                stack.append(child)

    return SearchResult(
        status=("cutoff" if cutoff_happened else "failure"),
        goal_state=None,
        actions=[],
        path_cost=float("inf"),
        stats=SearchStats(nodes_expanded, nodes_generated, max_frontier),
    )


def astar(
    problem: SearchProblem[S, A],
    *,
    heuristic: Callable[[S], float],
    node_limit: Optional[int] = None,
) -> SearchResult[S, A]:
    """Full textbook A* graph search with CLOSED set and RE-OPEN policy.

    f(n) = g(n) + h(n)

    Data structures:
    - g_score[state] = best-known path cost to reach state
    - parents[state] = (parent_state, action_taken)
    - open_heap = priority queue ordered by (f, h, tie)
    - closed = set of expanded states

    Re-open policy:
    - If a better g is found for a state already in CLOSED, it is removed from CLOSED and pushed again.

    Notes:
    - Requires non-negative step costs.
    - With admissible heuristic, this returns an optimal solution.
    """
    start = problem.initial_state()

    if problem.is_goal(start):
        return SearchResult(
            status="solved",
            goal_state=start,
            actions=[],
            path_cost=0.0,
            stats=SearchStats(nodes_expanded=0, nodes_generated=1, max_frontier_size=1),
        )

    parents: Dict[S, Tuple[S, A]] = {}
    g_score: Dict[S, float] = {start: 0.0}
    closed: Set[S] = set()

    # Heap items include g_at_push so we can skip stale entries without decrease-key.
    open_heap: List[Tuple[float, float, int, S, float]] = []
    tie = 0
    h0 = float(heuristic(start))
    heapq.heappush(open_heap, (h0, h0, tie, start, 0.0))
    tie += 1

    nodes_generated = 1
    nodes_expanded = 0
    max_frontier = 1

    while open_heap:
        max_frontier = max(max_frontier, len(open_heap))

        f, h, _, state, g_pushed = heapq.heappop(open_heap)

        # Skip stale heap entries
        g_cur = g_score.get(state)
        if g_cur is None or g_pushed != g_cur:
            continue

        # If it's still closed, it means we popped an old copy before re-opening
        if state in closed:
            continue

        if problem.is_goal(state):
            actions = _reconstruct_actions_from_parents(parents, start, state)
            return SearchResult(
                status="solved",
                goal_state=state,
                actions=actions,
                path_cost=g_cur,
                stats=SearchStats(nodes_expanded, nodes_generated, max_frontier),
            )

        closed.add(state)
        nodes_expanded += 1

        if node_limit is not None and nodes_expanded > node_limit:
            return SearchResult(
                status="timeout",
                goal_state=None,
                actions=[],
                path_cost=float("inf"),
                stats=SearchStats(nodes_expanded, nodes_generated, max_frontier),
            )

        for action, nxt, step_cost in problem.successors(state):
            if step_cost < 0:
                raise ValueError("A* requires non-negative step costs.")

            tentative_g = g_cur + step_cost
            old_g = g_score.get(nxt)

            if old_g is None or tentative_g < old_g:
                parents[nxt] = (state, action)
                g_score[nxt] = tentative_g

                if nxt in closed:
                    closed.remove(nxt)  # re-open

                hn = float(heuristic(nxt))
                fn = tentative_g + hn
                heapq.heappush(open_heap, (fn, hn, tie, nxt, tentative_g))
                tie += 1
                nodes_generated += 1

    return SearchResult(
        status="failure",
        goal_state=None,
        actions=[],
        path_cost=float("inf"),
        stats=SearchStats(nodes_expanded, nodes_generated, max_frontier),
    )
