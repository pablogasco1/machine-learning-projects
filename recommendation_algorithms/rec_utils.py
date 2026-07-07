"""
Shared utilities for the recommendation-systems learning notebooks.

Every notebook in this folder imports from here so that:
  - all methods are trained/evaluated on the *exact same* data and splits
  - the evaluation protocol (leave-one-out + Recall@K / NDCG@K) is identical
  - each notebook can stay focused on the algorithm it's teaching, not on
    boilerplate data wrangling.

The dataset itself is synthetic but built to *behave* like a real implicit-feedback
recommender dataset (think: a tiny MovieLens): users have latent taste vectors
over a handful of interpretable "genres", items have genre vectors, and
interactions are sampled from user-item affinity plus popularity plus noise.
Because we control the ground truth, we can sanity-check every model later by
checking whether it recovers the genre structure (e.g. via a 2D PCA plot).
"""

import json
import os

import numpy as np
import pandas as pd

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")

GENRE_NAMES = [
    "Action", "Comedy", "Drama", "Romance",
    "SciFi", "Horror", "Documentary", "Animation",
]
N_GENRES = len(GENRE_NAMES)

# A fixed, CVD-validated categorical palette (one color per genre, in a fixed
# order -- never reassigned/cycled) plus a sequential blue ramp for magnitude
# (e.g. popularity) and chart chrome colors. Reused by every notebook so all
# plots in this series look consistent.
GENRE_COLORS = [
    "#2a78d6",  # Action    - blue
    "#1baf7a",  # Comedy    - aqua
    "#eda100",  # Drama     - yellow
    "#008300",  # Romance   - green
    "#4a3aa7",  # SciFi     - violet
    "#e34948",  # Horror    - red
    "#e87ba4",  # Documentary - magenta
    "#eb6834",  # Animation - orange
]
SEQUENTIAL_BLUE = ["#cde2fb", "#9ec5f4", "#5598e7", "#2a78d6", "#184f95"]
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID_COLOR = "#e1e0d9"


def setup_plot_style():
    """Apply a consistent, minimal chart style across all notebooks."""
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "figure.facecolor": "#fcfcfb",
        "axes.facecolor": "#fcfcfb",
        "axes.edgecolor": GRID_COLOR,
        "axes.labelcolor": INK_SECONDARY,
        "axes.grid": True,
        "grid.color": GRID_COLOR,
        "grid.linewidth": 0.8,
        "text.color": INK,
        "xtick.color": INK_MUTED,
        "ytick.color": INK_MUTED,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "font.size": 11,
        "figure.dpi": 100,
    })


# ---------------------------------------------------------------------------
# 1. Synthetic data generation
# ---------------------------------------------------------------------------

def generate_synthetic_data(n_users=600, n_items=400, interactions_per_user=25,
                             n_genres=N_GENRES, seed=42):
    """Generate a synthetic implicit-feedback dataset.

    Returns
    -------
    interactions : DataFrame[user_id, item_id, t]
        One row per (user, item) interaction, `t` is the position of the
        interaction in that user's history (0 = oldest), used later for a
        realistic time-based leave-one-out split.
    user_prefs : ndarray (n_users, n_genres)
        Ground-truth latent taste vector per user (for inspection/plots only,
        no model is allowed to see this).
    item_genres : ndarray (n_items, n_genres)
        Ground-truth latent genre vector per item (for inspection/plots only).
    item_pop : ndarray (n_items,)
        Ground-truth popularity bias per item (some items are just watched a
        lot regardless of taste match -- like blockbusters).

    Note on the constants below: they're tuned (see project notes) so that a
    model which actually recovers user taste clearly beats a plain popularity
    baseline -- e.g. oracle recall@10 ~= 0.18 vs popularity recall@10 ~= 0.05.
    That gap is what makes the later notebooks' comparisons informative.
    """
    rng = np.random.default_rng(seed)

    # Each user likes a mix of 1-2 genres strongly, the rest weakly.
    user_prefs = rng.dirichlet(alpha=np.full(n_genres, 0.25), size=n_users)

    # Each item belongs mostly to 1 genre with a little bit of overlap.
    item_genres = rng.dirichlet(alpha=np.full(n_genres, 0.25), size=n_items)

    # Popularity bias: a few blockbusters, a long tail of niche items.
    item_pop = rng.pareto(a=2.0, size=n_items)
    item_pop = (item_pop - item_pop.min()) / (item_pop.max() - item_pop.min())

    rows = []
    for u in range(n_users):
        # Affinity of this user to every item = taste match + popularity bias.
        affinity = user_prefs[u] @ item_genres.T           # (n_items,)
        affinity = affinity + 0.3 * item_pop
        affinity = affinity + rng.normal(0, 0.04, size=n_items)  # noise

        # Sample interactions_per_user distinct items, weighted by affinity.
        probs = np.exp(affinity * 6.0)
        probs = probs / probs.sum()
        chosen = rng.choice(n_items, size=interactions_per_user,
                             replace=False, p=probs)

        # Order them randomly in "time" -- earlier picks = earlier history.
        order = rng.permutation(interactions_per_user)
        for t, idx in enumerate(order):
            rows.append((u, int(chosen[idx]), t))

    interactions = pd.DataFrame(rows, columns=["user_id", "item_id", "t"])
    return interactions, user_prefs, item_genres, item_pop


# ---------------------------------------------------------------------------
# 2. Train / test split
# ---------------------------------------------------------------------------

def leave_one_out_split(interactions):
    """Classic leave-one-out protocol: each user's *last* interaction (max t)
    goes to the test set, everything earlier goes to train. This mimics
    "predict what the user will watch next" and avoids leaking the future
    into training.
    """
    is_last = interactions.groupby("user_id")["t"].transform("max") == interactions["t"]
    test = interactions[is_last].reset_index(drop=True)
    train = interactions[~is_last].reset_index(drop=True)
    return train, test


# ---------------------------------------------------------------------------
# 3. Retrieval metrics
# ---------------------------------------------------------------------------

def recall_at_k(ranked_items, true_item, k):
    return 1.0 if true_item in ranked_items[:k] else 0.0


def ndcg_at_k(ranked_items, true_item, k):
    top_k = ranked_items[:k]
    if true_item not in top_k:
        return 0.0
    rank = top_k.index(true_item)  # 0-indexed
    
    # The formula is 1/log2(p + 1). This specific "+1" is chosen deliberately 
    # so that the top position (p=1) gives log2(2) = 1 
    return 1.0 / np.log2(rank + 2)  # +2 because rank is 0-indexed



def evaluate_recommender(recommend_fn, train, test, k_list=(10, 20)):
    """Evaluate a recommender against the leave-one-out test set.

    Parameters
    ----------
    recommend_fn : callable(user_id, n) -> list[item_id]
        Must return up to `n` recommended item ids for the user, ranked
        best-first, EXCLUDING items already seen in train.
    k_list : the cutoffs to report Recall@K and NDCG@K for.

    Returns
    -------
    dict of {"recall@K": value, "ndcg@K": value, ...} averaged over all
    test users.
    """
    max_k = max(k_list)
    recalls = {k: [] for k in k_list}
    ndcgs = {k: [] for k in k_list}

    for _, row in test.iterrows():
        u, true_item = row["user_id"], row["item_id"]
        ranked = recommend_fn(u, max_k)
        for k in k_list:
            recalls[k].append(recall_at_k(ranked, true_item, k))
            ndcgs[k].append(ndcg_at_k(ranked, true_item, k))

    metrics = {}
    for k in k_list:
        metrics[f"recall@{k}"] = float(np.mean(recalls[k]))
        metrics[f"ndcg@{k}"] = float(np.mean(ndcgs[k]))
    return metrics


def user_seen_items(train):
    """dict user_id -> set(item_id) already interacted with in train."""
    return train.groupby("user_id")["item_id"].apply(set).to_dict()


def user_item_lists(train):
    """dict user_id -> list(item_id) and dict item_id -> list(user_id), the
    basic adjacency lists most retrieval models (ALS, item2vec, ...) build
    their update rules on top of.
    """
    user_items = train.groupby("user_id")["item_id"].apply(list).to_dict()
    item_users = train.groupby("item_id")["user_id"].apply(list).to_dict()
    return user_items, item_users


def train_als(user_items, item_users, n_users, n_items,
              f=16, alpha=5.0, reg=0.05, n_iters=15, seed=0):
    """Implicit-feedback ALS (Hu, Koren & Volinsky, 2008). Identical to the
    from-scratch implementation walked through in `01_retrieval_als.ipynb` --
    reused here as a "given" retrieval component so later notebooks (ranking,
    two-tower) can build on top of it without re-deriving it.
    """
    rng = np.random.default_rng(seed)
    X = rng.normal(0, 0.1, size=(n_users, f))
    Y = rng.normal(0, 0.1, size=(n_items, f))
    I_f = np.eye(f)

    for _ in range(n_iters):
        YtY = Y.T @ Y
        for u in range(n_users):
            items = user_items.get(u, [])
            if not items:
                continue
            Yu = Y[items]
            A = YtY + alpha * (Yu.T @ Yu) + reg * I_f
            b = (1 + alpha) * Yu.sum(axis=0)
            X[u] = np.linalg.solve(A, b)

        XtX = X.T @ X
        for i in range(n_items):
            users = item_users.get(i, [])
            if not users:
                continue
            Xi = X[users]
            A = XtX + alpha * (Xi.T @ Xi) + reg * I_f
            b = (1 + alpha) * Xi.sum(axis=0)
            Y[i] = np.linalg.solve(A, b)

    return X, Y


def rank_topn(scores, seen, n):
    """Given a 1D array of scores over all items, return the top-n item ids
    best-first, skipping anything already in `seen`.
    """
    idx = np.argsort(-scores)
    out = []
    for i in idx:
        if i not in seen:
            out.append(int(i))
        if len(out) == n:
            break
    return out


# ---------------------------------------------------------------------------
# 4. Persisting results so the final comparison notebook can load them back
# ---------------------------------------------------------------------------

def save_results(name, metrics):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    path = os.path.join(RESULTS_DIR, f"{name}.json")
    with open(path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved results to {path}")


def load_all_results():
    out = {}
    if not os.path.isdir(RESULTS_DIR):
        return out
    for fname in sorted(os.listdir(RESULTS_DIR)):
        if fname.endswith(".json"):
            with open(os.path.join(RESULTS_DIR, fname)) as f:
                out[fname[:-5]] = json.load(f)
    return out
