"""
Futoshiki Puzzle Generator

Steps:
1. Generate a fully solved Latin square grid.
2. Apply random valid inequality constraints based on that solved grid.
3. Remove 1 value at a time.
4. Each time a value is removed, apply a backtracking solver to see if the grid can still be solved and to count the number of solutions it leads to.
5. If the resulting grid only has one solution we can carry on the process from step 3. If not, we put the value we took away back in the grid.
"""

import random
import copy

def is_valid(grid, constraints, row, col, num):
    """Check if placing num at grid[row][col] is valid according to Futoshiki rules."""
    size = len(grid)

    # Check row and column uniqueness
    for i in range(size):
        if grid[row][i] == num or grid[i][col] == num:
            return False

    # Check inequality constraints
    for (r1, c1), (r2, c2) in constraints:
        val1 = num if (r1 == row and c1 == col) else grid[r1][c1]
        val2 = num if (r2 == row and c2 == col) else grid[r2][c2]
        
        if val1 != 0 and val2 != 0:
            if val1 >= val2:
                return False
                
    return True

def find_empty(grid):
    """Find the next empty cell (value 0) in the grid."""
    size = len(grid)
    for r in range(size):
        for c in range(size):
            if grid[r][c] == 0:
                return (r, c)
    return None

def count_solutions(grid, constraints, limit=2):
    """Count the number of solutions for a given grid, stopping early once
    the count reaches `limit`."""
    count = [0]
    size = len(grid)

    def _solve(g):
        if count[0] >= limit:
            return
        empty = find_empty(g)
        if not empty:
            count[0] += 1
            return
        row, col = empty
        for num in range(1, size + 1):
            if is_valid(g, constraints, row, col, num):
                g[row][col] = num
                _solve(g)
                if count[0] >= limit:
                    g[row][col] = 0
                    return
                g[row][col] = 0

    _solve(grid)
    return count[0]

def _generate_latin_square(size):
    """Helper to generate a fully solved size x size Latin square."""
    grid = [[0 for _ in range(size)] for _ in range(size)]
    
    def _solve(g):
        empty = find_empty(g)
        if not empty:
            return True
        row, col = empty
        numbers = list(range(1, size + 1))
        random.shuffle(numbers)
        
        for num in numbers:
            # Pass an empty set for constraints to just fulfill Latin square rules
            if is_valid(g, set(), row, col, num):
                g[row][col] = num
                if _solve(g):
                    return True
                g[row][col] = 0
        return False
        
    _solve(grid)
    return grid

def add_inequalities(grid, num_constraints):
    """Generate random valid inequalities based on the solved grid."""
    size = len(grid)
    constraints = set()
    possible_pairs = []
    
    # Find all adjacent pairs horizontally and vertically
    for r in range(size):
        for c in range(size):
            if c < size - 1:
                possible_pairs.append(((r, c), (r, c + 1)))
            if r < size - 1:
                possible_pairs.append(((r, c), (r + 1, c)))
                
    selected_pairs = random.sample(possible_pairs, min(num_constraints, len(possible_pairs)))
    
    for p1, p2 in selected_pairs:
        val1 = grid[p1[0]][p1[1]]
        val2 = grid[p2[0]][p2[1]]
        if val1 < val2:
            constraints.add((p1, p2))
        else:
            constraints.add((p2, p1))
            
    return constraints

def remove_numbers(grid, constraints, target_blanks=15):
    """Remove numbers from a full grid one at a time, ensuring the puzzle
    always has a unique solution, until the target number of blanks is reached.

    Args:
        grid: A fully solved Futoshiki grid (will be modified in place).
        constraints: The set of inequality tuples for the grid.
        target_blanks: Desired number of empty cells.

    Returns:
        The grid with numbers removed (same object as input).
    """
    size = len(grid)
    max_blanks = size * size
    target_blanks = max(0, min(target_blanks, max_blanks))

    # Build a list of all cell positions and shuffle them
    cells = [(r, c) for r in range(size) for c in range(size)]
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
        if count_solutions(grid_copy, constraints, limit=2) != 1:
            # More than one solution — put the value back
            grid[row][col] = removed_value
        else:
            current_blanks += 1

    return grid

def generate_futoshiki(size=5, num_constraints=None):
    """Generate a Futoshiki puzzle with a unique solution and approximately
    the requested number of blank cells.

    Args:
        size: The dimension of the grid (default 5).
        num_constraints: Desired number of inequalities to generate (default 10).
        blanks: Desired number of empty cells (default 15).

    Returns:
        puzzle: list of lists with 0s for blank cells.
        solution: The fully solved grid.
        constraints: A set of tuples defining the inequalities.
    """
    if num_constraints is None:
        num_constraints = size*2 - 1
    
    if size < 7:
        target_blanks = size * size
    else:
        target_blanks = (size * size) // 2 + 5  
    
    # 1. Generate full board
    solution = _generate_latin_square(size)
    
    # 2. Add structural logic rules
    constraints = add_inequalities(solution, num_constraints)
    
    # 3. Create the puzzle 
    puzzle = copy.deepcopy(solution)
    remove_numbers(puzzle, constraints, target_blanks=target_blanks)
    
    return puzzle, solution, constraints

