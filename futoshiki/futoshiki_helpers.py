"""
Helper functions for parsing and printing Futoshiki boards.
"""

from typing import List, Set, Tuple

FutoshikiState = Tuple[int, ...]
ConstraintSet = Set[Tuple[Tuple[int, int], Tuple[int, int]]]

def parse_board(text: str) -> Tuple[int, FutoshikiState, ConstraintSet]:
    """Parse a Futoshiki board from a visual text string."""
    # 1. Clean the text: ignore comments, keep internal spacing
    lines = [line.rstrip() for line in text.split('\n') if not line.lstrip().startswith(("#", "//"))]
    while lines and not lines[0]: lines.pop(0)  # Remove empty top lines
    while lines and not lines[-1]: lines.pop()  # Remove empty bottom lines

    size = (len(lines) + 1) // 2
    state: List[int] = []
    constraints: ConstraintSet = set()
    
    for r in range(size):
        # 2. Parse numbers and horizontal constraints (<, >)
        c = 0
        for token in lines[2 * r].split():
            if token.isdigit():
                state.append(int(token))
                c += 1
            elif token == '<': constraints.add(((r, c - 1), (r, c)))
            elif token == '>': constraints.add(((r, c), (r, c - 1)))

        # 3. Parse vertical constraints (^, v)
        if r < size - 1:
            vert_row = lines[2 * r + 1]
            for c in range(size):
                if c * 4 < len(vert_row):  # Check if a symbol exists at this column index
                    sign = vert_row[c * 4]
                    if sign == '^': constraints.add(((r, c), (r + 1, c)))
                    if sign == 'v': constraints.add(((r + 1, c), (r, c)))

    if len(state) != size * size:
        raise ValueError(f"Expected {size * size} digits, got {len(state)}.")
        
    return size, tuple(state), constraints


def board_to_str(state: FutoshikiState, size: int, constraints: ConstraintSet) -> str:
    """Pretty-print a Futoshiki board with its constraints."""
    lines: List[str] = []
    
    for r in range(size):
        row_str = ""
        for c in range(size):
            row_str += str(state[r * size + c])
            if c < size - 1:
                if ((r, c), (r, c + 1)) in constraints: row_str += " < "
                elif ((r, c + 1), (r, c)) in constraints: row_str += " > "
                else: row_str += "   "
        lines.append(row_str)
        
        if r < size - 1:
            vert_str = ""
            for c in range(size):
                if ((r, c), (r + 1, c)) in constraints: vert_str += "^   "
                elif ((r + 1, c), (r, c)) in constraints: vert_str += "v   "
                else: vert_str += "    "
            lines.append(vert_str)
            
    return "\n".join(lines)