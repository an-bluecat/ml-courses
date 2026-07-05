"""
Chapter 4: Actor-Critic — Taming the Variance

THE PROBLEM WITH REINFORCE:
  The return Gₜ is noisy. Episode A might get +8 by luck, Episode B gets +2
  by bad luck, even though the actions were equally good. REINFORCE can't tell
  the difference — it just pushes A's actions 4x harder.

THE FIX — ADVANTAGE:
  Instead of "was the return high?", ask "was the return HIGHER THAN EXPECTED?"

  Advantage = Gₜ - V(sₜ)

  V(sₜ) is the value function: "from this state, what return do I expect on average?"
  - Advantage > 0: "better than expected — reinforce this action"
  - Advantage < 0: "worse than expected — discourage this action"
  - Advantage ≈ 0: "exactly as expected — don't change anything"

THE ARCHITECTURE:
  Actor:  the policy network π(a|s) — picks actions
  Critic: the value network V(s)   — predicts expected return

  They train together:
  - Critic learns to predict returns (supervised, MSE loss)
  - Actor uses (Gₜ - V(sₜ)) instead of raw Gₜ in the policy gradient

Run it:  python 4_actor_critic.py
"""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

torch.manual_seed(42)
np.random.seed(42)

# === ENVIRONMENT (same gridworld) ===
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
    t = torch.zeros(GRID_SIZE * GRID_SIZE)
    t[state[0] * GRID_SIZE + state[1]] = 1.0
    return t


# === ACTOR (policy) and CRITIC (value) ===
class ActorCritic(nn.Module):
    def __init__(self):
        super().__init__()
        self.shared = nn.Sequential(nn.Linear(16, 64), nn.ReLU())
        self.actor = nn.Sequential(nn.Linear(64, 4), nn.Softmax(dim=-1))
        self.critic = nn.Linear(64, 1)  # outputs a single value V(s)

    def forward(self, x):
        h = self.shared(x)
        return self.actor(h), self.critic(h)

model = ActorCritic()
optimizer = optim.Adam(model.parameters(), lr=0.005)
gamma = 0.99

print("=== Actor-Critic on Gridworld ===\n")

reward_history = []

for episode in range(1, 501):
    states, actions, rewards, values = [], [], [], []
    state = START

    for t in range(50):
        s_tensor = state_to_tensor(state)
        probs, value = model(s_tensor)

        dist = torch.distributions.Categorical(probs)
        action = dist.sample()

        next_state, reward, done = step(state, action.item())

        states.append(s_tensor)
        actions.append(action)
        rewards.append(reward)
        values.append(value.squeeze())

        state = next_state
        if done:
            break

    # --- COMPUTE RETURNS ---
    returns = []
    G = 0
    for r in reversed(rewards):
        G = r + gamma * G
        returns.insert(0, G)
    returns = torch.tensor(returns)

    # --- COMPUTE ADVANTAGE ---
    # Advantage = actual return - predicted value
    # This is the "was it better or worse than expected?" signal
    values_tensor = torch.stack(values)
    advantages = returns - values_tensor.detach()  # detach: don't backprop advantage through critic

    # --- ACTOR LOSS (policy gradient with advantage instead of raw return) ---
    actor_loss = torch.tensor(0.0)
    for s, a, adv in zip(states, actions, advantages):
        probs, _ = model(s)
        log_prob = torch.log(probs[a])
        actor_loss += -log_prob * adv    # ADVANTAGE replaces raw Gₜ

    # --- CRITIC LOSS (learn to predict returns) ---
    # Simple MSE: make V(s) close to the actual return G
    critic_loss = nn.functional.mse_loss(values_tensor, returns)

    # --- COMBINED LOSS ---
    loss = actor_loss + 0.5 * critic_loss

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    total_reward = sum(rewards)
    reward_history.append(total_reward)

    if episode % 50 == 0 or episode == 1:
        avg = np.mean(reward_history[-50:])
        print(f"Episode {episode:>3d} | Reward: {total_reward:>6.1f} | "
              f"Avg(50): {avg:>6.1f} | Actor loss: {actor_loss.item():>7.2f} | "
              f"Critic loss: {critic_loss.item():.3f}")

# --- SHOW POLICY AND VALUES ---
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
                probs, _ = model(s)
            best = torch.argmax(probs).item()
            row += f"  {ACTION_NAMES[best]}  "
    print(row)

print("\n=== Learned Values V(s) ===")
for r in range(GRID_SIZE):
    row = ""
    for c in range(GRID_SIZE):
        s = state_to_tensor((r, c))
        with torch.no_grad():
            _, v = model(s)
        row += f" {v.item():>5.1f} "
    print(row)

# === WHAT TO NOTICE ===
# 1. Compare the Avg(50) reward curve to Chapter 3's REINFORCE:
#    Actor-Critic should be smoother and converge faster.
# 2. The value map should show: high near treasure, low near pit, medium elsewhere
# 3. The advantage is the KEY difference:
#    - REINFORCE: "return was +5, so push this action" (but maybe +5 is normal!)
#    - Actor-Critic: "return was +5 but I expected +3, so advantage = +2, mild push"
# 4. Critic loss should decrease over time — it's getting better at predicting returns
#
# NEXT: Chapter 5 adds one more idea — "don't change the policy TOO much in one update."
# That's PPO.
