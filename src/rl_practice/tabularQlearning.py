import numpy as np
from collections import deque
import matplotlib.pyplot as plt
import os 

result_folder = 'tabQLearning'
os.makedirs(result_folder, exist_ok = True)
os.chdir(result_folder)


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

    start_idx = 0
    end_idx =  (rows - 1) * cols + (cols - 1)
    
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


rows = 10 
cols = 10
num_states = rows * cols
num_actions = 4
fill_percent = 0.4
grid, s_idx, e_idx = make_grid(rows, cols, fill_percent=fill_percent, seed=42)
print(grid)

num_states = rows * cols

num_actions = 4
UP, RIGHT, DOWN, LEFT = (-1, 0), (0, 1), (1, 0),(0, -1)

actions = [UP, RIGHT, DOWN, LEFT] 
Q_table = np.zeros((num_states, num_actions))
alpha = 0.1
gamma = 0.99
epsilon = 1.0
eps_min = 0.05 
eps_decay = 0.995 

episodes = 500 
max_steps = int(num_states*(1- fill_percent))
train_returns = []

eval_every = 50
eval_episodes = 50
greedy_eval_x = []
greedy_success = []
greedy_avg_return = []

ARROWS = {0: "↑", 1: "→", 2: "↓", 3: "←"}

def reset(grid):
    sr, sc= np.argwhere(grid == START)[0]
    return sr*cols + sc


def step(curr_state, action):
    curr =  [curr_state//cols, curr_state%cols] 
    nr = curr[0] + action[0]
    nc = curr[1] + action[1]
    next_pos = [nr, nc]
    reward = -0.1
    done = False
    if not (0 <= nr < rows) or not (0 <= nc < cols):
        next_pos = curr
        reward = -1
    else:
        if grid[nr, nc] == WALL: 
            next_pos = curr    
            reward = -1 
        else: 
            next_pos = [nr, nc]
            if grid[nr, nc] == FINISH:
                reward = +10
                done = True

    next_state = next_pos[0]*cols + next_pos[1]
    return next_state, reward, done


def eps_greedy(Q, state, epsilon):
    if np.random.rand() < epsilon:
        return np.random.randint(num_actions)  
    else:
        return np.argmax(Q[state])              


def evaluate(Q_table, grid, episodes=200, max_steps=500):
    successes = 0
    total_returns = []
    steps_to_finish = []

    for _ in range(episodes):
        s = reset(grid)
        ep_return = 0.0
        done = False

        for t in range(max_steps):
            a = int(np.argmax(Q_table[s]))          
            s, r, done = step(s, actions[a])        
            ep_return += r
            if done:
                successes += 1
                steps_to_finish.append(t + 1)
                break

        total_returns.append(ep_return)

    success_rate = successes / episodes
    avg_return = float(np.mean(total_returns))
    avg_steps = float(np.mean(steps_to_finish)) if steps_to_finish else None

    return success_rate, avg_return, avg_steps

for _e in range(episodes):
    s = reset(grid=grid)
    ep_return = 0.0

    for t in range(max_steps):
        a = eps_greedy(Q_table, s, epsilon=epsilon)
        s2, r, done = step(s, action=actions[a])

        ep_return += r

        target = r if done else (r + gamma * np.max(Q_table[s2]))
        Q_table[s, a] += alpha * (target - Q_table[s, a])

        s = s2
        if done:
            break

    train_returns.append(ep_return)
    epsilon = max(eps_min, epsilon * eps_decay)

    if (_e + 1) % eval_every == 0: # Greedy 
        sr, avg_ret, _ = evaluate(Q_table, grid, episodes=eval_episodes, max_steps=max_steps)
        greedy_eval_x.append(_e + 1)
        greedy_success.append(sr)
        greedy_avg_return.append(avg_ret)



def print_policy(grid, Q_table):
    rows, cols = grid.shape

    for r in range(rows):
        row_syms = []
        for c in range(cols):
            cell = grid[r, c]
            s = r * cols + c

            if cell == WALL:
                row_syms.append("W")
            elif cell == START:
                row_syms.append("S")
            elif cell == FINISH:
                row_syms.append("F")
            else:
                a = int(np.argmax(Q_table[s]))
                row_syms.append(ARROWS[a])

        print(" ".join(f"{x:>2}" for x in row_syms))

print_policy(grid, Q_table)
def print_and_save_q_table(Q_table, rows, cols, filename="Q_table_pretty.txt"):
    with open(filename, "w") as f:
        for r in range(rows):
            for c in range(cols):
                s = r * cols + c
                qvals = Q_table[s]
                line = (
                    f"State ({r},{c}) | "
                    f"U:{qvals[0]:6.2f} "
                    f"R:{qvals[1]:6.2f} "
                    f"D:{qvals[2]:6.2f} "
                    f"L:{qvals[3]:6.2f}\n"
                )
                print(line, end="")
                f.write(line)
            f.write("\n")

print_and_save_q_table(Q_table, rows, cols)

def save_policy(grid, Q_table, filename="policy.txt"):
    rows, cols = grid.shape
    with open(filename, "w") as f:
        for r in range(rows):
            row_syms = []
            for c in range(cols):
                cell = grid[r, c]
                s = r * cols + c
                if cell == WALL:
                    row_syms.append("W")
                elif cell == START:
                    row_syms.append("S")
                elif cell == FINISH:
                    row_syms.append("F")
                else:
                    row_syms.append(ARROWS[int(np.argmax(Q_table[s]))])
            line = " ".join(row_syms) + "\n"
            print(line, end="")
            f.write(line)

save_policy(grid, Q_table)


# Training return
plt.figure()
plt.plot(train_returns)
plt.xlabel("Episode")
plt.ylabel("Train return (ε-greedy)")
plt.title("Training return per episode")
plt.tight_layout()
plt.savefig("train_returns.png", dpi=150)
plt.close()

# Greedy average return
plt.figure()
plt.plot(greedy_eval_x, greedy_avg_return)
plt.xlabel("Episode")
plt.ylabel("Greedy avg return")
plt.title("Greedy evaluation over training")
plt.tight_layout()
plt.savefig("greedy_avg_return.png", dpi=150)
plt.close()

# Greedy success rate
plt.figure()
plt.plot(greedy_eval_x, greedy_success)
plt.xlabel("Episode")
plt.ylabel("Greedy success rate")
plt.ylim(0, 1.05)
plt.title("Greedy success rate over training")
plt.tight_layout()
plt.savefig("greedy_success_rate.png", dpi=150)
plt.close()

