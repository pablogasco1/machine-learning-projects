# Reinforcement Learning for Business Problems

A hands-on notebook series that goes from "what is RL?" to RL applied to problems a consulting engagement is likely to actually touch: pricing, inventory, marketing personalization, and staffing.

Run the notebooks in order — each one reuses vocabulary and code patterns from the last.

| # | Notebook | New concept | Business framing |
|---|---|---|---|
| 1 | `01_multi_armed_bandit.ipynb` | Exploration vs. exploitation, epsilon-greedy, regret | Which of several options (ads/offers/designs) is best? |
| 2 | `02_gridworld_qlearning.ipynb` | Full MDP: state, policy, Q-table, the Q-learning update | Toy navigation task — the mechanics every later notebook reuses |
| 3 | `03_dynamic_pricing.ipynb` | State-dependent decisions over a finite horizon | Revenue management: pricing perishable, capacity-limited inventory (flights, hotels, events) |
| 4 | `04_inventory_management.ipynb` | Actions that change the state itself | Reorder policy vs. the classical (s, S) heuristic from Operations Research |
| 5 | `05_marketing_contextual_bandit.ipynb` | Contextual bandits (bandit + context, no state transition) | Personalized discount targeting vs. a single company-wide offer |
| 6 | `06_staff_scheduling.ipynb` | Cyclical demand + carryover state together | Rostering staff against a weekly demand pattern and a cost-compounding backlog |

## Running them

All notebooks use only `numpy`, `matplotlib`, and (in notebook 5's "going further" discussion) reference `scikit-learn` concepts — everything already in this repo's `requirements.txt`. Each notebook is self-contained: open it and run all cells top to bottom.

## The one idea that repeats six times

Every business case in this series is solved with the same recipe:

1. **Frame it as an MDP (or a bandit, if there's no state):** define the state, the actions, and a reward function that matches the business objective (revenue, cost, margin).
2. **Learn a policy from simulated trial and error** — epsilon-greedy Q-learning for stateful problems, epsilon-greedy incremental averaging for bandits. No labeled "correct answer" is ever needed.
3. **Validate against a realistic baseline**, not just "did the reward go up." The baseline is usually the best fixed rule, or the best classical heuristic already in use — not a strawman.
4. **Inspect the learned policy itself** (a heatmap, a bar chart, a policy curve), not just the headline metric. This is how you catch two real failure modes: states the agent rarely visited (its estimates there aren't trustworthy), and states where the value differences are so small that the "best" action is really just noise. Both come up in this series (notebooks 3, 4, and 6) and both are the kind of thing a stakeholder will ask about.

The specific algorithm (tabular Q-learning here) is the least important part — swap it for LinUCB, Thompson Sampling, or deep RL once the state/action/reward framing and the validation habits are solid.
