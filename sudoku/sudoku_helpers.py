"""
Helper functions for parsing and printing Sudoku boards.
"""

from typing import List, Tuple

SudokuState = Tuple[int, ...]

def parse_board(text: str) -> SudokuState:
    """
    Parse a board from a string of 9 lines × 9 digits.
    Lines starting with '#' or '//' are ignored.
    """
    digits: List[int] = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or line.startswith("//"):
            continue
        for ch in line:
            if ch.isdigit():
                digits.append(int(ch))
    if len(digits) != 81:
        raise ValueError(f"Expected 81 digits, got {len(digits)}.")
    return tuple(digits)


def board_to_str(state: SudokuState) -> str:
    """Pretty-print a Sudoku board."""
    lines: List[str] = []
    for r in range(9):
        row = state[r * 9 : r * 9 + 9]
        parts = [
            " ".join(str(d) if d != 0 else "." for d in row[c : c + 3])
            for c in (0, 3, 6)
        ]
        lines.append(" | ".join(parts))
        if r in (2, 5):
            lines.append("------+-------+------")
    return "\n".join(lines)