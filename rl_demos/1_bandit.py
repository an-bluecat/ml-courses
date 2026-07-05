"""
Chapter 1: The Multi-Armed Bandit — RL stripped to its bones.

No states. No sequences. Just: pick an action, get a reward, learn.

THE STORY:
  You walk into a casino with 3 slot machines. Each has a hidden payout rate
  you don't know. You have 1000 pulls total. How do you figure out which
  machine is best AND maximize your winnings at the same time?

  This is the explore-exploit dilemma — the first core tension of RL.

THE ALGORITHM (epsilon-greedy):
  - With probability epsilon: pull a random machine  (EXPLORE)
  - With probability 1-epsilon: pull the best-so-far  (EXPLOIT)
  - Update your estimate of that machine's value

Run it:  python 1_bandit.py
"""
import numpy as np

np.random.seed(42)

# === THE ENVIRONMENT ===
# 3 slot machines with hidden true payout rates (you don't know these!)
TRUE_PAYOUTS = [0.2, 0.5, 0.8]  # Machine 2 is the best

def pull_machine(machine_id):
    """Pull a lever. Returns 1 (win) or 0 (lose) based on hidden probability."""
    return 1.0 if np.random.random() < TRUE_PAYOUTS[machine_id] else 0.0


# === THE AGENT ===
n_machines = 3
Q = np.zeros(n_machines)       # estimated value of each machine (starts at 0)
N = np.zeros(n_machines)       # how many times we've pulled each machine
epsilon = 0.1                  # 10% of the time, explore randomly

total_reward = 0
n_steps = 1000

print("=== Multi-Armed Bandit ===")
print(f"True payouts (hidden from agent): {TRUE_PAYOUTS}\n")

for step in range(1, n_steps + 1):
    # --- DECIDE: explore or exploit? ---
    if np.random.random() < epsilon:
        action = np.random.randint(n_machines)   # explore: random machine
    else:
        action = np.argmax(Q)                     # exploit: best known machine

    # --- ACT: pull the lever ---
    reward = pull_machine(action)
    total_reward += reward

    # --- LEARN: update estimate with running average ---
    N[action] += 1
    # Q_new = Q_old + (1/N) * (reward - Q_old)
    # This is just an incremental mean: each new reward nudges the estimate
    Q[action] += (1.0 / N[action]) * (reward - Q[action])

    # --- PRINT progress ---
    if step in [10, 50, 100, 500, 1000]:
        print(f"Step {step:>4d} | Estimates: [{Q[0]:.3f}, {Q[1]:.3f}, {Q[2]:.3f}] "
              f"| Pulls: [{int(N[0])}, {int(N[1])}, {int(N[2])}] "
              f"| Total reward: {total_reward:.0f}")

print(f"\nAgent's final answer: Machine {np.argmax(Q)} is best "
      f"(estimated payout: {Q[np.argmax(Q)]:.3f})")
print(f"Truth: Machine {np.argmax(TRUE_PAYOUTS)} is best "
      f"(true payout: {max(TRUE_PAYOUTS)})")

# === WHAT TO NOTICE ===
# 1. Early on, estimates are noisy — the agent hasn't pulled enough
# 2. Over time, estimates converge toward the true values
# 3. Machine 2 gets pulled the most (exploit) but others still get sampled (explore)
# 4. If epsilon=0: pure exploit, might get stuck on a bad machine forever
# 5. If epsilon=1: pure explore, never capitalizes on what it learns
#
# WHAT'S MISSING: There are no STATES here. The best machine is always the best.
# Real RL problems have states — "which machine is best depends on the situation."
# That's what Chapter 2 adds.
