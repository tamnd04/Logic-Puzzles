"""
Helper functions for parsing and printing Futoshiki boards.
"""

from typing import List, Set, Tuple

FutoshikiState = Tuple[int, ...]
ConstraintSet = Set[Tuple[Tuple[int, int], Tuple[int, int]]]
def parse_board(text: str) -> Tuple[int, FutoshikiState, ConstraintSet]:
    """
    Parse a Futoshiki board from a visual text string.
    Lines starting with '#' or '//' are ignored.
    """
    raw_lines = text.splitlines()
    
    # 1. Remove comments but KEEP empty lines (using rstrip to clean trailing spaces)
    clean_lines = [
        line.rstrip() for line in raw_lines 
        if not line.lstrip().startswith(("#", "//"))
    ]
    
    # 2. Strip purely empty lines from the VERY TOP of the file
    while clean_lines and clean_lines[0] == "":
        clean_lines.pop(0)
        
    # 3. Strip purely empty lines from the VERY BOTTOM of the file
    while clean_lines and clean_lines[-1] == "":
        clean_lines.pop()
        
    lines = clean_lines
    if not lines:
        raise ValueError("Input file contains no valid puzzle data.")
    
    # Now the alternating structure is safely preserved!
    size = (len(lines) + 1) // 2
    
    state: List[int] = []
    constraints: ConstraintSet = set()
    
    for r in range(size):
        # Parse the row with numbers (0-9) and horizontal signs (<, >)
        num_line = lines[2 * r]
        tokens = num_line.split()
        
        col = 0
        for token in tokens:
            if token.isdigit():
                state.append(int(token))
                col += 1
            elif token == '<':
                constraints.add(((r, col - 1), (r, col)))
            elif token == '>':
                constraints.add(((r, col), (r, col - 1)))

        # Parse the row with vertical signs (^, v)
        if r < size - 1:
            vert_line = lines[2 * r + 1]
            for c in range(size):
                idx = c * 4
                if len(vert_line) > idx:
                    ch = vert_line[idx]
                    if ch == '^':
                        constraints.add(((r, c), (r + 1, c)))
                    elif ch == 'v':
                        constraints.add(((r + 1, c), (r, c)))

    if len(state) != size * size:
        raise ValueError(f"Expected {size * size} digits, got {len(state)}.")
        
    return size, tuple(state), constraints

def board_to_str(state: FutoshikiState, size: int, constraints: ConstraintSet) -> str:
    """Pretty-print a Futoshiki board with its constraints."""
    lines: List[str] = []
    for r in range(size):
        row_str = ""
        for c in range(size):
            val = state[r * size + c]
            row_str += str(val)  # Just print the number directly!
            
            if c < size - 1:
                if ((r, c), (r, c + 1)) in constraints:
                    row_str += " < "
                elif ((r, c + 1), (r, c)) in constraints:
                    row_str += " > "
                else:
                    row_str += "   "
        lines.append(row_str)
        
        if r < size - 1:
            vert_str = ""
            for c in range(size):
                if ((r, c), (r + 1, c)) in constraints:
                    vert_str += "^   "
                elif ((r + 1, c), (r, c)) in constraints:
                    vert_str += "v   "
                else:
                    vert_str += "    "
            lines.append(vert_str)
            
    return "\n".join(lines)