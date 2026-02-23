from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Dict, Any
import csv


# -----------------------------
# Sudoku I/O
# -----------------------------
def load_sudoku_from_text(text: str) -> List[List[int]]:
    """Parse a Sudoku from text.

    Accepted formats:
    - 9 lines of 9 digits with 0 for empty: 530070000
    - digits separated by spaces: 5 3 0 0 7 0 0 0 0
    - ignores whitespace and common separators
    """
    lines = [ln.strip() for ln in text.splitlines() if ln.strip() and not ln.strip().startswith("#")]
    grid: List[List[int]] = []

    for ln in lines:
        digits = [int(ch) for ch in ln if ch.isdigit()]
        if len(digits) == 9:
            grid.append(digits)
        else:
            parts = ln.replace("|", " ").replace(".", "0").split()
            if parts and all(p.isdigit() for p in parts) and len(parts) == 9:
                grid.append([int(p) for p in parts])

    if len(grid) != 9 or any(len(r) != 9 for r in grid):
        raise ValueError(f"Expected 9x9 Sudoku grid, got {len(grid)} rows.")

    if any(not (0 <= x <= 9) for r in grid for x in r):
        raise ValueError("Sudoku values must be in 0..9")

    return grid


def load_sudoku(path: str | Path) -> List[List[int]]:
    p = Path(path)
    return load_sudoku_from_text(p.read_text(encoding="utf-8"))


# -----------------------------
# Futoshiki I/O
# -----------------------------
@dataclass(frozen=True)
class FutoshikiInput:
    n: int
    grid: List[List[int]]
    inequalities: List[Tuple[Tuple[int, int], str, Tuple[int, int]]]  # ((r,c), '<' or '>', (r,c)) 0-based


def load_futoshiki_from_text(text: str) -> FutoshikiInput:
    lines = [ln.strip() for ln in text.splitlines() if ln.strip() and not ln.strip().startswith("#")]
    if not lines:
        raise ValueError("Empty futoshiki input")

    n = int(lines[0])
    if n < 2:
        raise ValueError("Futoshiki N must be >= 2")

    if len(lines) < 1 + n:
        raise ValueError(f"Expected {n} grid lines after N")

    grid: List[List[int]] = []
    for i in range(1, 1 + n):
        parts = lines[i].replace("|", " ").split()
        row = [int(p) for p in parts]
        if len(row) != n:
            raise ValueError(f"Row {i} expected {n} numbers, got {len(row)}")
        if any(not (0 <= x <= n) for x in row):
            raise ValueError(f"Futoshiki values must be in 0..{n}")
        grid.append(row)

    inequalities: List[Tuple[Tuple[int, int], str, Tuple[int, int]]] = []
    for ln in lines[1 + n:]:
        parts = ln.split()
        if len(parts) != 5 or parts[2] not in ("<", ">"):
            raise ValueError(f"Bad inequality line: {ln!r}. Expected: r1 c1 < r2 c2")
        r1, c1 = int(parts[0]) - 1, int(parts[1]) - 1
        rel = parts[2]
        r2, c2 = int(parts[3]) - 1, int(parts[4]) - 1
        if not (0 <= r1 < n and 0 <= c1 < n and 0 <= r2 < n and 0 <= c2 < n):
            raise ValueError(f"Inequality indices out of range in line: {ln!r}")
        inequalities.append(((r1, c1), rel, (r2, c2)))

    return FutoshikiInput(n=n, grid=grid, inequalities=inequalities)


def load_futoshiki(path: str | Path) -> FutoshikiInput:
    p = Path(path)
    return load_futoshiki_from_text(p.read_text(encoding="utf-8"))


# -----------------------------
# Results table helper
# -----------------------------
def write_results_csv(rows: List[Dict[str, Any]], path: str | Path) -> None:
    if not rows:
        raise ValueError("No rows to write")
    p = Path(path)
    headers = sorted({k for r in rows for k in r.keys()})
    with p.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=headers)
        w.writeheader()
        for r in rows:
            w.writerow(r)
