"""
Chapter 5: PPO — Proximal Policy Optimization

THE PROBLEM WITH ACTOR-CRITIC:
  Sometimes a single lucky (or unlucky) episode causes a huge gradient update
  that completely changes the policy. The agent was doing well, then one bad
  update destroys everything. This is the "catastrophic update" problem.

THE FIX — CLIPPING:
  PPO says: "you can update the policy, but not TOO much in one step."

  Instead of:  loss = -log π(a|s) · advantage
  PPO uses:    loss = -min(ratio · A, clip(ratio, 1-ε, 1+ε) · A)

  where ratio = π_new(a|s) / π_old(a|s)

  If ratio is close to 1 → new policy is similar to old → allow the update
  If ratio strays beyond [1-ε, 1+ε] → clip it → prevent drastic change

  This is like saying: "learn from this experience, but don't overreact."

THE FULL PICTURE:
  Chapter 1: Bandit      → "which action is good?" (no states)
  Chapter 2: Q-learning  → "which action is good in which state?" (table)
  Chapter 3: REINFORCE   → "use a neural network, weight by return" (noisy)
  Chapter 4: Actor-Critic → "subtract baseline to reduce noise" (sometimes unstable)
  Chapter 5: PPO         → "clip updates to stay stable" (this is what people actually use)

Run it:  python 5_ppo.py
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


# === ACTOR-CRITIC MODEL ===
class ActorCritic(nn.Module):
    def __init__(self):
        super().__init__()
        self.shared = nn.Sequential(nn.Linear(16, 64), nn.ReLU())
        self.actor = nn.Sequential(nn.Linear(64, 4), nn.Softmax(dim=-1))
        self.critic = nn.Linear(64, 1)

    def forward(self, x):
        h = self.shared(x)
        return self.actor(h), self.critic(h)

model = ActorCritic()
optimizer = optim.Adam(model.parameters(), lr=0.005)
gamma = 0.99
clip_epsilon = 0.2   # THE PPO CLIPPING PARAMETER
ppo_epochs = 4       # re-use each batch of experience this many times
entropy_coeff = 0.01 # encourage exploration (prevents policy from collapsing)

print("=== PPO on Gridworld ===\n")

reward_history = []

for episode in range(1, 1001):
    # --- COLLECT EPISODE ---
    states, actions, rewards, old_log_probs, values = [], [], [], [], []
    state = START

    for t in range(50):
        s_tensor = state_to_tensor(state)
        with torch.no_grad():
            probs, value = model(s_tensor)

        dist = torch.distributions.Categorical(probs)
        action = dist.sample()

        next_state, reward, done = step(state, action.item())

        states.append(s_tensor)
        actions.append(action)
        rewards.append(reward)
        old_log_probs.append(dist.log_prob(action))  # SAVE the old probability
        values.append(value.squeeze())

        state = next_state
        if done:
            break

    # --- COMPUTE RETURNS AND ADVANTAGES ---
    returns = []
    G = 0
    for r in reversed(rewards):
        G = r + gamma * G
        returns.insert(0, G)
    returns = torch.tensor(returns)
    old_log_probs = torch.stack(old_log_probs)
    values_tensor = torch.stack(values)
    advantages = returns - values_tensor.detach()

    if len(advantages) > 1:
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    # --- PPO UPDATE: re-use this experience multiple times ---
    states_batch = torch.stack(states)
    actions_batch = torch.stack(actions)

    for _ in range(ppo_epochs):
        # Get CURRENT probabilities (these change each epoch!)
        probs, new_values = model(states_batch)
        dist = torch.distributions.Categorical(probs)
        new_log_probs = dist.log_prob(actions_batch)

        # --- THE RATIO: how much has the policy changed? ---
        # ratio = π_new(a|s) / π_old(a|s)
        # In log space: exp(log π_new - log π_old)
        ratio = torch.exp(new_log_probs - old_log_probs.detach())

        # --- THE CLIPPED OBJECTIVE ---
        # Unclipped: ratio * advantage (same as normal policy gradient)
        # Clipped: clip ratio to [1-ε, 1+ε], then multiply by advantage
        # Take the MINIMUM of both → pessimistic bound → prevents overreaction
        unclipped = ratio * advantages
        clipped = torch.clamp(ratio, 1 - clip_epsilon, 1 + clip_epsilon) * advantages
        actor_loss = -torch.min(unclipped, clipped).mean()

        # Entropy bonus: encourages exploration by penalizing certainty
        # Without this, the policy can collapse to always picking one action
        entropy = dist.entropy().mean()

        # Critic loss (same as before)
        critic_loss = nn.functional.mse_loss(new_values.squeeze(), returns)

        loss = actor_loss + 0.5 * critic_loss - entropy_coeff * entropy

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    total_reward = sum(rewards)
    reward_history.append(total_reward)

    if episode % 100 == 0 or episode == 1:
        avg = np.mean(reward_history[-100:])
        print(f"Episode {episode:>4d} | Reward: {total_reward:>6.1f} | "
              f"Avg(100): {avg:>6.1f} | Ratio range: [{ratio.min():.2f}, {ratio.max():.2f}]")

# --- SHOW RESULTS ---
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

print("\n=== The Full RL Journey ===")
print("1. Bandit:       no states, just actions → learn which arm is best")
print("2. Q-learning:   states + actions → build a table of values")
print("3. REINFORCE:    replace table with neural net → scale to big problems")
print("4. Actor-Critic: subtract baseline → reduce noise in gradient")
print("5. PPO:          clip ratio → prevent catastrophic updates")
print("\nFrom here → RLHF: replace environment reward with human preference model.")
print("That's how ChatGPT was trained.")

# === WHAT TO NOTICE ===
# 1. "Ratio range" should stay near [1.0, 1.0] — the clipping is working
# 2. If ratio hits 0.8 or 1.2, the clip kicks in and prevents larger changes
# 3. ppo_epochs=4 means we squeeze more learning from each episode
#    (REINFORCE throws away each episode after one update — wasteful!)
# 4. Training should be more stable than Chapter 4
#
# THE BRIDGE TO RLHF:
# Replace the gridworld reward with: "a human (or a reward model) rates the output"
# The policy is an LLM. The "action" is the next token. The "state" is the prompt + tokens so far.
# PPO updates the LLM to generate text that scores higher on the reward model.
# That's RLHF. Same algorithm. Different environment.
