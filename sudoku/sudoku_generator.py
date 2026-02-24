"""
Sudoku puzzle generator using py_sudoku_maker, which only generates fully solved sudoku grids.
Implements logic for removing some digits to create a playable puzzle.
"""

import random
from py_sudoku_maker import generate_sudoku

# Generate a complete sudoku
sudoku = generate_sudoku()

# Remove numbers to create an unsolved puzzle
def create_puzzle(blanks: int = 40) -> list[list[int]]:
    puzzle = [row[:] for row in sudoku]  # Copy the grid
    positions = [(i, j) for i in range(9) for j in range(9)]
    random.shuffle(positions)
    
    for i in range(blanks):
        row, col = positions[i]
        puzzle[row][col] = 0  # 0 represents empty cell
    
    return puzzle