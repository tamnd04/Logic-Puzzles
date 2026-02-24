"""
Sudoku Puzzle Generator

Steps:
1. Generate a fully solved Sudoku grid using `py_sudoku_maker`
2. Remove 1 value at a time.
3. Each time a value is removed, apply a sudoku solver algorithm to see if the grid can still be solved and to count the number of solutions it leads to.
4. If the resulting grid only has one solution we can carry on the process from step 2. If not we will have to put the value we took away back in the grid.

References: https://www.101computing.net/sudoku-generator-algorithm/
"""

import random
import copy
import py_sudoku_maker

def is_valid(grid, row, col, num):
    """Check if placing num at grid[row][col] is valid according to Sudoku rules."""
    # Check row
    if num in grid[row]:
        return False

    # Check column
    for r in range(9):
        if grid[r][col] == num:
            return False

    # Check 3x3 box
    box_row, box_col = 3 * (row // 3), 3 * (col // 3)
    for r in range(box_row, box_row + 3):
        for c in range(box_col, box_col + 3):
            if grid[r][c] == num:
                return False

    return True


def find_empty(grid):
    """Find the next empty cell (value 0) in the grid."""
    for r in range(9):
        for c in range(9):
            if grid[r][c] == 0:
                return (r, c)
    return None


def count_solutions(grid, limit=2):
    """Count the number of solutions for a given grid, stopping early once
    the count reaches `limit`."""
    count = [0]

    def _solve(g):
        if count[0] >= limit:
            return
        empty = find_empty(g)
        if not empty:
            count[0] += 1
            return
        row, col = empty
        for num in range(1, 10):
            if is_valid(g, row, col, num):
                g[row][col] = num
                _solve(g)
                if count[0] >= limit:
                    g[row][col] = 0
                    return
                g[row][col] = 0

    _solve(grid)
    return count[0]


def remove_numbers(grid, target_blanks=40):
    """Remove numbers from a full grid one at a time, ensuring the puzzle
    always has a unique solution, until the target number of blanks is reached.

    Args:
        grid: A fully solved 9x9 Sudoku grid (will be modified in place).
        target_blanks: Desired number of empty cells (capped at 64 to keep
                       the puzzle solvable).

    Returns:
        The grid with numbers removed (same object as input).
    """
    target_blanks = max(0, min(target_blanks, 64))

    # Build a list of all 81 cell positions and shuffle them
    cells = [(r, c) for r in range(9) for c in range(9)]
    random.shuffle(cells)

    current_blanks = sum(row.count(0) for row in grid)

    for row, col in cells:
        if current_blanks >= target_blanks:
            break

        removed_value = grid[row][col]
        if removed_value == 0:
            continue  # Already empty

        grid[row][col] = 0

        # Check that the puzzle still has exactly one solution
        grid_copy = copy.deepcopy(grid)
        if count_solutions(grid_copy, limit=2) != 1:
            # More than one solution — put the value back
            grid[row][col] = removed_value
        else:
            current_blanks += 1

    return grid


def generate_sudoku(blanks=40):
    """Generate a Sudoku puzzle with a unique solution and approximately
    the requested number of blank cells.

    Args:
        blanks: Desired number of empty cells (0-64). The actual number
                may be slightly lower if uniqueness cannot be maintained.

    Returns:
        puzzle: 9x9 list with 0s for blank cells.
        solution: The fully solved 9x9 grid.
    """
    solution = py_sudoku_maker.generate_sudoku()
    puzzle = copy.deepcopy(solution)
    remove_numbers(puzzle, target_blanks=blanks)
    return puzzle, solution