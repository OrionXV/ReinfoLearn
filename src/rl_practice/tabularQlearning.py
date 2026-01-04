import numpy as np
from collections import deque


def reset():
    pass 

def step(action): 
    pass

rows = 0
cols = 0

num_states = rows * cols
num_actions = 4 

grid_dic = {
    0 : 'E', # empty 
    1 : 'F', # finish 
    -1 : 'S', # start 
    2 : 'W' # wall
}

EMPTY, FINISH, START, WALL = 0, 1, -1, 2


def make_grid(rows, cols, fill_percent, seed=0, max_attempts=100000):
    rng = np.random.default_rng(seed)

    g = np.zeros((rows, cols), dtype=int)

    start_idx = rng.integers(0, rows * cols)
    end_idx = rng.integers(0, rows * cols)
    while end_idx == start_idx:
        end_idx = rng.integers(0, rows * cols)

    sr, sc = start_idx // cols, start_idx % cols
    er, ec = end_idx // cols, end_idx % cols

    g[sr, sc] = START
    g[er, ec] = FINISH

    target_walls = int(fill_percent * rows * cols)

    wall_count = 0
    attempts = 0

    while wall_count < target_walls and attempts < max_attempts:
        attempts += 1

        idx = rng.integers(0, rows * cols)
        r, c = idx // cols, idx % cols

        if g[r, c] != EMPTY:
            continue

        g[r, c] = WALL

        if check_valid_grid(g):
            wall_count += 1
        else:
            g[r, c] = EMPTY

    return g, start_idx, end_idx

def check_valid_grid(g):
    rows, cols = g.shape
    start_pos = np.argwhere(g == START)
    finish_pos = np.argwhere(g == FINISH)
    
    if len(start_pos) == 0 or len(finish_pos) == 0:
        return False

    sr, sc = start_pos[0]
    fr, fc = finish_pos[0]

    q = deque([(sr, sc)])
    visited = set([(sr, sc)])

    moves = [(-1,0), (1,0), (0,-1), (0,1)]

    while q:
        r, c = q.popleft()

        if (r, c) == (fr, fc):
            return True

        for dr, dc in moves:
            nr, nc = r + dr, c + dc
            if not (0 <= nr < rows and 0 <= nc < cols):
                continue
            if (nr, nc) in visited:
                continue
            if g[nr, nc] == WALL:
                continue

            visited.add((nr, nc))
            q.append((nr, nc))

    return False

g, s, e = make_grid(6, 6, fill_percent=0.25, seed=42)
print(g)
print("valid:", check_valid_grid(g))
