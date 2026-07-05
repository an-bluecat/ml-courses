"""
Chapter 2: Q-Learning on a Gridworld — Now we have STATES.

THE STORY:
  A robot is in a 4x4 grid. It starts top-left, treasure is bottom-right.
  There's a pit at (1,1) that gives -10 reward. Every step costs -1.
  Finding treasure gives +10.

  The robot doesn't have a map. It has to bump around and learn:
  "from THIS state, which action leads to the best FUTURE reward?"

  That question — "what's the value of action A in state S?" — is Q(s, a).

THE FIVE CONCEPTS from Chapter 2:
  State:   where the robot is (row, col)
  Action:  up/down/left/right
  Reward:  -1 per step, -10 pit, +10 treasure
  Policy:  "in each state, pick the action with highest Q"
  Return:  total discounted reward from here to the end

THE ALGORITHM (Q-learning):
  Q(s,a) ← Q(s,a) + α * [r + γ * max_a' Q(s',a') - Q(s,a)]

  In English: "nudge Q toward the reward I just got + the best I can do from the next state"

Run it:  python 2_q_learning.py
"""
import numpy as np

np.random.seed(42)

# === THE ENVIRONMENT ===
GRID_SIZE = 4
START = (0, 0)
TREASURE = (3, 3)
PIT = (1, 1)

ACTIONS = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}  # up, down, left, right
ACTION_NAMES = {0: "↑", 1: "↓", 2: "←", 3: "→"}

def step(state, action):
    """Take an action, return (next_state, reward, done)."""
    dr, dc = ACTIONS[action]
    nr, nc = state[0] + dr, state[1] + dc
    # Walls: stay in place
    nr = max(0, min(GRID_SIZE - 1, nr))
    nc = max(0, min(GRID_SIZE - 1, nc))
    next_state = (nr, nc)

    if next_state == TREASURE:
        return next_state, +10.0, True
    elif next_state == PIT:
        return next_state, -10.0, True
    else:
        return next_state, -1.0, False


# === THE AGENT ===
# Q-table: for every (state, action) pair, store the estimated value
Q = np.zeros((GRID_SIZE, GRID_SIZE, 4))

alpha = 0.1     # learning rate: how fast we update Q
gamma = 0.9     # discount factor: how much we care about future rewards
epsilon = 0.2   # exploration rate

n_episodes = 500

print("=== Q-Learning Gridworld ===")
print(f"Grid: {GRID_SIZE}x{GRID_SIZE}, Start: {START}, Treasure: {TREASURE}, Pit: {PIT}\n")

for episode in range(1, n_episodes + 1):
    state = START
    total_reward = 0

    for t in range(50):  # max 50 steps per episode
        # --- DECIDE ---
        if np.random.random() < epsilon:
            action = np.random.randint(4)
        else:
            action = np.argmax(Q[state[0], state[1]])

        # --- ACT ---
        next_state, reward, done = step(state, action)
        total_reward += reward

        # --- LEARN: the Q-learning update ---
        # "What I thought" vs "what actually happened + best future"
        old_q = Q[state[0], state[1], action]
        best_future = np.max(Q[next_state[0], next_state[1]])
        target = reward + gamma * best_future  # TD target
        Q[state[0], state[1], action] += alpha * (target - old_q)

        state = next_state
        if done:
            break

    if episode in [1, 10, 50, 100, 500]:
        print(f"Episode {episode:>3d} | Total reward: {total_reward:>6.1f} | Steps: {t+1}")

# --- SHOW THE LEARNED POLICY ---
print("\n=== Learned Policy (best action per state) ===")
for r in range(GRID_SIZE):
    row = ""
    for c in range(GRID_SIZE):
        if (r, c) == TREASURE:
            row += "  ★  "
        elif (r, c) == PIT:
            row += "  ✕  "
        else:
            best = np.argmax(Q[r, c])
            row += f"  {ACTION_NAMES[best]}  "
    print(row)

print("\n=== Q-values at START (0,0) ===")
for a in range(4):
    print(f"  {ACTION_NAMES[a]}: {Q[0, 0, a]:.2f}")

# === WHAT TO NOTICE ===
# 1. Early episodes: random stumbling, often falls in pit → big negative reward
# 2. Later episodes: agent learns the path, reaches treasure in few steps
# 3. The policy shows arrows pointing toward treasure, away from pit
# 4. Q-values at start: "down" and "right" should be highest (toward treasure)
#
# KEY INSIGHT: Q-learning builds a complete map of "value of each action in each state"
# This is a TABLE — it only works when states are small and discrete.
# What if states are images? Continuous? Huge? You need a neural network.
# That's Chapter 3.
