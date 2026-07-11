"""GA and Random search baselines for FSS inverse design.

Both methods use the MIONet world model for S-parameter evaluation
and the v11 reward function for fitness scoring.
"""

import time

import numpy as np
import torch

from common.config import DEVICE, N_FREQ, GRID_SIZE, QUARTER_SIZE, STANDARD_FREQS
from inverse.reward import calculate_sparam_reward_v11, aggregate_stability_reward
from inverse.task import build_active_design_task


# ===================================================================
#  Pattern mirroring
# ===================================================================

def quarter_to_full(q):
    """Mirror a (5,5) quarter to a symmetric (10,10) full pattern."""
    q = np.asarray(q, dtype=np.float32)
    lo = np.flipud(q)
    le = np.concatenate((q, lo), axis=0)
    ri = np.fliplr(le)
    return np.concatenate((le, ri), axis=1).astype(np.float32)


# ===================================================================
#  Batch evaluation
# ===================================================================

def evaluate_pattern(wm, full, specs):
    """Batch-evaluate one full (10,10) pattern against all specs.

    Returns the aggregated v11 stability reward.
    """
    full = np.asarray(full, dtype=np.float32)
    ft = (
        torch.from_numpy(full).float().to(DEVICE).unsqueeze(0).unsqueeze(0)
    )

    n = len(specs)
    Mt_batch = ft.repeat(n, 1, 1, 1)
    thetas = torch.tensor(
        [float(s["theta"]) for s in specs], device=DEVICE,
    )
    pols = torch.tensor(
        [int(s["pol"]) for s in specs], device=DEVICE, dtype=torch.long,
    )

    with torch.no_grad():
        s11_batch, s21_batch, _ = wm.predict_stats(
            Mt_batch, theta=thetas, pol=pols,
        )

    rewards = []
    for i, spec in enumerate(specs):
        s11 = s11_batch[i].detach().cpu().numpy().flatten()
        s21 = s21_batch[i].detach().cpu().numpy().flatten()
        r, _, _ = calculate_sparam_reward_v11(
            np.abs(s11), np.abs(s21), STANDARD_FREQS, spec,
        )
        rewards.append(float(r))

    return aggregate_stability_reward(rewards, specs, stability_weight=0.25)


# ===================================================================
#  GA operators
# ===================================================================

def _init_population(size, n_bits=25):
    """Initialise a GA population.

    60% random binary, 40% sparse (few random ones).
    """
    pop = []
    for _ in range(int(size * 0.6)):
        pop.append(np.random.randint(0, 2, n_bits).astype(np.uint8))
    for _ in range(size - len(pop)):
        ind = np.zeros(n_bits, dtype=np.uint8)
        n_ones = np.random.randint(1, max(2, n_bits // 4))
        ind[np.random.choice(n_bits, n_ones, replace=False)] = 1
        pop.append(ind)
    return pop


def _tournament_select(pop, fits, k=3):
    """Tournament selection: pick *k* individuals, return the fittest."""
    idxs = np.random.choice(len(pop), k, replace=False)
    return pop[idxs[np.argmax([fits[i] for i in idxs])]].copy()


def _uniform_crossover(p1, p2, prob=0.8):
    """Uniform crossover with probability *prob*."""
    if np.random.random() > prob:
        return p1.copy(), p2.copy()
    mask = np.random.random(len(p1)) < 0.5
    return np.where(mask, p1, p2), np.where(mask, p2, p1)


def _mutate(ind, prob=0.06):
    """Bit-flip mutation with per-bit probability *prob*."""
    mask = np.random.random(len(ind)) < prob
    ind[mask] = 1 - ind[mask]
    return ind


# ===================================================================
#  GA search
# ===================================================================

def run_ga(wm, specs, n_queries=600, pop_size=30, tournament_k=3,
           crossover_prob=0.8, mutation_prob=0.06, elitism=2,
           early_stop_stale=8, verbose=True):
    """Genetic algorithm search using the world model for fitness.

    Parameters
    ----------
    wm : LocalEnsembleWorldModel
        Pre-loaded world model.
    specs : list of dict
        FSS design specifications (from ``build_active_design_task``).
    n_queries : int
        Total number of WM evaluations allowed.
    pop_size : int
        Population size per generation.
    tournament_k : int
        Tournament selection size.
    crossover_prob : float
        Probability of applying crossover.
    mutation_prob : float
        Per-bit mutation probability.
    elitism : int
        Number of elite individuals preserved each generation.
    early_stop_stale : int
        Stop if no improvement for this many generations.
    verbose : bool
        Print progress.

    Returns
    -------
    best_full : ndarray  (10, 10)
    best_fitness : float
    n_evals : int
    history : list of float
        Best fitness after each generation.
    """
    n_bits = QUARTER_SIZE * QUARTER_SIZE  # 25
    n_gens = max(0, (n_queries - pop_size) // pop_size)

    pop = _init_population(pop_size, n_bits)
    fits = np.array([
        evaluate_pattern(
            wm,
            quarter_to_full(p.reshape(QUARTER_SIZE, QUARTER_SIZE)),
            specs,
        )
        for p in pop
    ])
    n_evals = pop_size

    best_idx = np.argmax(fits)
    best_ind = pop[best_idx].copy()
    best_fit = fits[best_idx]
    history = [best_fit]
    stale = 0

    for gen in range(1, n_gens + 1):
        if n_evals + pop_size > n_queries:
            break

        # Elitism
        elite_idxs = np.argsort(fits)[-elitism:]
        elites = [pop[i].copy() for i in elite_idxs]

        new_pop = []
        while len(new_pop) < pop_size - elitism:
            c1, c2 = _uniform_crossover(
                _tournament_select(pop, fits, tournament_k),
                _tournament_select(pop, fits, tournament_k),
                crossover_prob,
            )
            new_pop.append(_mutate(c1, mutation_prob))
            if len(new_pop) < pop_size - elitism:
                new_pop.append(_mutate(c2, mutation_prob))

        pop = elites + new_pop[:pop_size - elitism]
        fits = np.array([
            evaluate_pattern(
                wm,
                quarter_to_full(p.reshape(QUARTER_SIZE, QUARTER_SIZE)),
                specs,
            )
            for p in pop
        ])
        n_evals += pop_size

        gen_best_idx = np.argmax(fits)
        gen_best_fit = fits[gen_best_idx]
        if gen_best_fit > best_fit:
            best_fit = gen_best_fit
            best_ind = pop[gen_best_idx].copy()
            stale = 0
        else:
            stale += 1
        history.append(best_fit)

        if verbose and gen % 5 == 0:
            print(
                f"    GA gen{gen:>3d} | best={best_fit:.4f}  "
                f"mean={np.mean(fits):.4f}  stale={stale}"
            )

        if stale >= early_stop_stale:
            if verbose:
                print(f"    GA early stop at gen {gen} (stale={stale})")
            break

    best_full = quarter_to_full(
        best_ind.reshape(QUARTER_SIZE, QUARTER_SIZE),
    )
    return best_full, best_fit, n_evals, history


# ===================================================================
#  Random search
# ===================================================================

def run_random(wm, specs, n_queries=600, verbose=True):
    """Random search using the world model for fitness.

    Parameters
    ----------
    wm : LocalEnsembleWorldModel
    specs : list of dict
    n_queries : int
        Number of random patterns to evaluate.
    verbose : bool
        Print progress.

    Returns
    -------
    best_full : ndarray  (10, 10)
    best_fitness : float
    all_fits : list of float
    """
    best_full = None
    best_fit = -float("inf")
    all_fits = []

    for i in range(n_queries):
        q = np.random.randint(
            0, 2, (QUARTER_SIZE, QUARTER_SIZE),
        ).astype(np.uint8)
        # Ensure at least one metal pixel
        if q.sum() == 0:
            q[
                np.random.randint(0, QUARTER_SIZE),
                np.random.randint(0, QUARTER_SIZE),
            ] = 1
        full = quarter_to_full(q)
        fit = evaluate_pattern(wm, full, specs)
        all_fits.append(fit)
        if fit > best_fit:
            best_fit = fit
            best_full = full.copy()

        if verbose and (i + 1) % 150 == 0:
            print(f"    Random {i + 1:>4d}/{n_queries} | best={best_fit:.4f}")

    return best_full, best_fit, all_fits
