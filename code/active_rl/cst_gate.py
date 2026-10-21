"""Uncertainty-gated, budgeted, deduplicated CST verification.

env.py / eval_sweep.py call should_trigger_cst / select_cst_query_specs /
evaluate_cst_query_specs via module attributes (cst_gate.<name>) so that the
tau monitor can wrap them at runtime.
"""

import random
from dataclasses import dataclass, field

import numpy as np

from active_rl.config import (
    N_FREQ,
    ANGLES_DEG,
    DEFAULT_CST_BUDGET,
    DEFAULT_CST_INTERVAL_EPISODES,
    DEFAULT_MIN_SURROGATE_REWARD,
    UNCERTAINTY_THRESHOLD,
    SURROGATE_IMPROVE_MARGIN,
    EPSILON_CST_FORCE,
)
from active_rl.reward import task_reward_v11


def pattern_key(pattern_10x10):
    pattern = np.asarray(pattern_10x10, dtype=np.float32)
    return tuple(pattern.astype(int).flatten().tolist())


@dataclass
class ActiveQueryState:
    cst_budget: int = DEFAULT_CST_BUDGET
    cst_interval_episodes: int = DEFAULT_CST_INTERVAL_EPISODES
    min_surrogate_reward: float = DEFAULT_MIN_SURROGATE_REWARD
    uncertainty_threshold: float = UNCERTAINTY_THRESHOLD
    improve_margin: float = SURROGATE_IMPROVE_MARGIN
    force_prob: float = EPSILON_CST_FORCE
    best_surrogate_r: float = -float("inf")
    n_design_queries: int = 0
    queried_patterns: set = field(default_factory=set)

    def observe_surrogate(self, reward):
        self.best_surrogate_r = max(self.best_surrogate_r, float(reward))

    def mark_queried(self, pattern_10x10, reward):
        self.queried_patterns.add(pattern_key(pattern_10x10))
        self.n_design_queries += 1
        self.observe_surrogate(reward)


def should_trigger_cst(query_state, pattern_10x10, current_func_r, uncertainty,
                       episode_index, rng=None):
    if rng is None:
        rng = random.random
    current_func_r = float(current_func_r)
    uncertainty = float(uncertainty)
    budget_exhausted = has_cst_budget_limit(query_state) and query_state.n_design_queries >= int(query_state.cst_budget)
    interval = max(1, int(query_state.cst_interval_episodes))
    interval_blocked = int(episode_index) % interval != 0
    already_seen = pattern_key(pattern_10x10) in query_state.queried_patterns
    below_min_reward = current_func_r < float(query_state.min_surrogate_reward)

    has_baseline = np.isfinite(query_state.best_surrogate_r)
    cold_start_candidate = (not has_baseline) and not below_min_reward
    predicted_new_best = (
        has_baseline
        and current_func_r > (query_state.best_surrogate_r + query_state.improve_margin)
    )
    high_uncertainty = uncertainty > float(query_state.uncertainty_threshold)
    forced = rng() < float(query_state.force_prob)

    trigger = (
        not budget_exhausted
        and not interval_blocked
        and not already_seen
        and not below_min_reward
        and (cold_start_candidate or predicted_new_best or high_uncertainty or forced)
    )
    return trigger, {
        "budget_exhausted": bool(budget_exhausted),
        "interval_blocked": bool(interval_blocked),
        "already_seen": bool(already_seen),
        "below_min_reward": bool(below_min_reward),
        "cold_start_candidate": bool(cold_start_candidate),
        "predicted_new_best": bool(predicted_new_best),
        "high_uncertainty": bool(high_uncertainty),
        "forced": bool(forced),
    }


def select_representative_specs(specs):
    selected = []
    for theta in (ANGLES_DEG[0], ANGLES_DEG[-1]):
        for pol in (0, 1):
            for spec in specs:
                if int(spec.get("theta", -1)) == int(theta) and int(spec.get("pol", -1)) == pol:
                    selected.append(spec)
                    break
    return selected


def select_cst_query_specs(specs, mode="representative"):
    if mode == "representative":
        return select_representative_specs(specs)
    if mode == "full":
        return list(specs)
    raise ValueError(f"Unknown cst query mode: {mode}")


def get_cst_call_count(cst_client):
    if cst_client is None:
        return 0
    return int(getattr(cst_client, "n_calls", 0))
def _spec_pol_int(spec):
    pol = spec.get("pol", 0)
    if isinstance(pol, str):
        return 0 if pol.strip().upper() in ("TE", "0") else 1
    return int(pol)


def _normalize_both_result(result):
    if isinstance(result, dict):
        if "TE" in result or "TM" in result:
            return {0: result.get("TE"), 1: result.get("TM")}
        return {int(k): v for k, v in result.items()}
    te_s11, te_s21, tm_s11, tm_s21 = result
    return {0: (te_s11, te_s21), 1: (tm_s11, tm_s21)}


def evaluate_cst_query_specs(cst, wm, pattern_10x10, query_specs, freqs, add_to_buffer=True):
    rewards = []
    used_specs = []
    success = 0
    remaining = list(query_specs)

    if hasattr(cst, "simulate_both"):
        grouped = {}
        for spec in remaining:
            grouped.setdefault(float(spec.get("theta", 0.0)), []).append(spec)

        consumed = set()
        for theta, specs_at_theta in grouped.items():
            pols = {_spec_pol_int(spec) for spec in specs_at_theta}
            if not {0, 1}.issubset(pols):
                continue
            try:
                both = _normalize_both_result(cst.simulate_both(pattern_10x10, theta=theta))
                for spec in specs_at_theta:
                    pol = _spec_pol_int(spec)
                    if pol not in both or both[pol] is None:
                        continue
                    s11_cst, s21_cst = both[pol]
                    cst_reward, _, _ = task_reward_v11(
                        s11_cst[:N_FREQ], s21_cst[:N_FREQ], freqs, spec)
                    rewards.append(cst_reward)
                    used_specs.append(spec)
                    if add_to_buffer and wm is not None:
                        wm.add_to_buffer(pattern_10x10, s11_cst, s21_cst, theta=theta, pol=pol)
                    success += 1
                    consumed.add(id(spec))
            except Exception as exc:
                labels = ", ".join(spec.get("label", "spec") for spec in specs_at_theta)
                print(f"  [CST skip] BOTH theta={theta:g} ({labels}): {exc}")
        remaining = [spec for spec in remaining if id(spec) not in consumed]

    for spec in remaining:
        th = spec.get("theta", 0.0)
        pol_val = _spec_pol_int(spec)
        try:
            s11_cst, s21_cst = cst.simulate(pattern_10x10, theta=th, pol=pol_val)
            cst_reward, _, _ = task_reward_v11(
                s11_cst[:N_FREQ], s21_cst[:N_FREQ], freqs, spec)
            rewards.append(cst_reward)
            used_specs.append(spec)
            if add_to_buffer and wm is not None:
                wm.add_to_buffer(pattern_10x10, s11_cst, s21_cst, theta=th, pol=pol_val)
            success += 1
        except Exception as exc:
            print(f"  [CST skip] {spec.get('label', 'spec')}: {exc}")

    return rewards, used_specs, success


def has_cst_budget_limit(query_state):
    return int(getattr(query_state, "cst_budget", 0)) > 0


def cst_budget_label(query_state):
    if query_state is None or not has_cst_budget_limit(query_state):
        return "unlimited"
    return str(int(query_state.cst_budget))
