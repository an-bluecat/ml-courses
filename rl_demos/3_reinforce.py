"""
Chapter 3: REINFORCE — Policy Gradient with a Neural Network

THE STORY:
  Same gridworld, but now instead of a Q-table, we use a neural network
  that directly outputs: "given this state, here are the probabilities
  for each action."

  This IS the StatQuest video example, generalized:
  - Run an episode (sequence of state→action→reward)
  - Compute the return G for each step
  - If G is high → make that action MORE likely (increase log π)
  - If G is low → make that action LESS likely

THE ALGORITHM (REINFORCE):
  1. Run a full episode, collecting (state, action, reward) at each step
  2. Compute returns: Gₜ = rₜ + γ·rₜ₊₁ + γ²·rₜ₊₂ + ...
  3. Loss = -Σ log π(aₜ|sₜ) · Gₜ
     (negative because we want to MAXIMIZE reward, but optimizers MINIMIZE)
  4. Backprop and update the network

CONNECTION TO STATQUEST:
  - "Make a guess" = sample action from π(a|s)
  - "Cross entropy" = -log π(a|s)
  - "Multiply by reward" = multiply gradient by Gₜ
  - "Negative reward flips direction" = negative Gₜ reverses the gradient

Run it:  python 3_reinforce.py
"""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

torch.manual_seed(42)
np.random.seed(42)

# === THE ENVIRONMENT (same gridworld as Chapter 2) ===
GRID_SIZE = 4
START = (0, 0)
TREASURE = (3, 3)
PIT = (1, 1)
ACTIONS = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}
ACTION_NAMES = {0: "↑", 1: "↓", 2: "←", 3: "→"}

def step(state, action):
    dr, dc = ACTIONS[action]
    nr = max(0, min(GRID_SIZE - 1, state[0] + dr))
    nc = max(0, min(GRID_SIZE - 1, state[1] + dc))
    next_state = (nr, nc)
    if next_state == TREASURE:
        return next_state, +10.0, True
    elif next_state == PIT:
        return next_state, -10.0, True
    else:
        return next_state, -1.0, False

def state_to_tensor(state):
    """One-hot encode the state (16 cells → 16-dim vector)."""
    t = torch.zeros(GRID_SIZE * GRID_SIZE)
    t[state[0] * GRID_SIZE + state[1]] = 1.0
    return t


# === THE POLICY NETWORK ===
# Input: one-hot state (16) → Hidden (32) → Output: action probabilities (4)
policy = nn.Sequential(
    nn.Linear(16, 32),
    nn.ReLU(),
    nn.Linear(32, 4),
    nn.Softmax(dim=-1),
)
optimizer = optim.Adam(policy.parameters(), lr=0.01)
gamma = 0.99

print("=== REINFORCE on Gridworld ===\n")

for episode in range(1, 501):
    # --- COLLECT ONE EPISODE ---
    states, actions, rewards = [], [], []
    state = START

    for t in range(50):
        s_tensor = state_to_tensor(state)
        probs = policy(s_tensor)

        # Sample action from the probability distribution (this is the "guess")
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()

        next_state, reward, done = step(state, action.item())

        states.append(s_tensor)
        actions.append(action)
        rewards.append(reward)

        state = next_state
        if done:
            break

    # --- COMPUTE RETURNS (Gₜ) ---
    # Walk backwards through the episode, accumulating discounted rewards
    returns = []
    G = 0
    for r in reversed(rewards):
        G = r + gamma * G
        returns.insert(0, G)
    returns = torch.tensor(returns)

    # Normalize returns (helps training stability, not strictly necessary)
    if len(returns) > 1:
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)

    # --- COMPUTE LOSS AND UPDATE ---
    # Loss = -Σ log π(aₜ|sₜ) · Gₜ
    # Positive Gₜ → decrease loss → make this action MORE likely
    # Negative Gₜ → increase loss → make this action LESS likely
    loss = torch.tensor(0.0)
    for s, a, G_t in zip(states, actions, returns):
        probs = policy(s)
        log_prob = torch.log(probs[a])
        loss += -log_prob * G_t     # THIS IS THE POLICY GRADIENT

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # --- PRINT ---
    total_reward = sum(rewards)
    if episode % 50 == 0 or episode == 1:
        print(f"Episode {episode:>3d} | Reward: {total_reward:>6.1f} | "
              f"Steps: {len(rewards):>2d} | Loss: {loss.item():.3f}")

# --- SHOW LEARNED POLICY ---
print("\n=== Learned Policy ===")
for r in range(GRID_SIZE):
    row = ""
    for c in range(GRID_SIZE):
        if (r, c) == TREASURE:
            row += "  ★  "
        elif (r, c) == PIT:
            row += "  ✕  "
        else:
            s = state_to_tensor((r, c))
            with torch.no_grad():
                probs = policy(s)
            best = torch.argmax(probs).item()
            row += f"  {ACTION_NAMES[best]}  "
    print(row)

# Show probability distribution at START
with torch.no_grad():
    probs = policy(state_to_tensor(START))
print(f"\nAction probs at START: ↑={probs[0]:.2f} ↓={probs[1]:.2f} ←={probs[2]:.2f} →={probs[3]:.2f}")

# === WHAT TO NOTICE ===
# 1. The loss = -log π(a|s) · G  is EXACTLY the StatQuest formula:
#    - log π(a|s) = "cross entropy of the guess"
#    - G = "the reward signal that confirms or flips the gradient"
# 2. Early episodes: random actions, terrible reward
# 3. Later: network learns to go toward treasure, avoid pit
# 4. But training is NOISY — reward fluctuates a lot between episodes
#    That's because G has high variance. Some episodes get lucky, others don't.
#
# THE PROBLEM: This noise makes REINFORCE slow and unstable.
# The fix? Subtract a baseline. That's Chapter 4.
