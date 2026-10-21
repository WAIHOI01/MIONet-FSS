# active_rl — Active PPO training code (paper configuration)

This folder contains the **active-reinforcement-learning training code**
used to produce the paper's RL inverse-design results. It is a flat package
refactor (2026-10-21) of the version chain kept verbatim in the research
repository (v8 → v9 → v11 → v12): the canonical **v11 semantics** (terminal
S-parameter reward `task_reward_v11`, fixed-target task sampling, 12 S-parameter
curve observations) are the single native implementation — no runtime
monkey-patching.

## Files

| File | Role |
|---|---|
| `train_rl_active_learning_world_model.py` | **Canonical entry point** (thin shim → `active_rl.cli`, so it also works when run as a plain script from any cwd). |
| `train.py` | Same entry via `python -m active_rl.train`. |
| `config.py` | Shared constants, paths, `results_world_model/` output dir; sets `KMP_DUPLICATE_LIB_OK` and warning filters before torch loads. |
| `networks.py` | Model definitions (MIONet, MIONetKernel, PhysicsBranch, Chebyshev frequency basis) — merged from the former `mionet_common.py` + `chebyshev_basis.py`; checkpoint layouts unchanged. |
| `cst_client.py` | CST TCP client (4-byte length framing, cache, retries) and `MockCST`. |
| `reward.py` | `task_reward_v11` terminal S-parameter reward chain + `aggregate_stability_reward`. |
| `tasks.py` | Task builders, fixed v11 4×4 (fc, bw) grid sampling, target masks, `V11_CURVE_KEYS`. |
| `world_model.py` | `LocalEnsembleWorldModel` (3 MIONet copies, progressive-noise finetuning) — the surrogate evaluated on every env step. |
| `cst_gate.py` | Uncertainty gating (`should_trigger_cst`), CST query selection, `evaluate_cst_query_specs`, budget bookkeeping. |
| `env.py` | `V11AngleDispersionEnv`: quarter-pixel placement episode, gate/query via the `cst_gate` module (late-bound for instrumentation). |
| `features.py` | `V11AngleDispersionFeatures` (image CNN + target MLP + curve MLP → 256-d). |
| `callbacks.py` | `AngleVizCallback`: snapshots, `cycle_metrics.jsonl`, best-artifact saving. |
| `eval_sweep.py` | Post-training policy rollout and (fc, bw) parameter sweep. |
| `trainer.py` | `train(args)`: PPO construction (module-level `PPO` = instrumentation patch point), RNG order, `run_meta.json` + v11 augmentation. |
| `cli.py` | The 27 trainer flags + 3 fixed-target flags; `main()`. |
| `tau_monitor.py` | τ-threshold sweep monitor (gate audit, CST call timing, certified-best log, candidates). Produces the `monitor_meta.json` files archived with the response letter. |
| `run_rl_hp_ablation.py` | PPO hyperparameter ablation launcher: 5 configs (ref / ent0 / ent0p1 / clip0p1 / clip0p3) × 3 seeds, 100k steps, mode agent, no CST. |
| `aggregate_rl_hp_ablation.py` | Folds ablation run dirs into per-config mean ± SD tables. |
| `aggregate_rl_hp_from_logs.py` | Same aggregation directly from trainer stdout logs (`<root>/ablation_logs/<tag>_s<seed>.log`). |
| `server/best_system_eq_robust_dropout_M{1,2}_c_theta0p0to60p0_order16.pth` | Pretrained world-model checkpoints (order-16 Chebyshev ensemble members) that the formal agents load by default. |

## Usage

Offline RL training (paper ablation protocol, mode agent) — all three forms
are equivalent:

```bash
python active_rl/train_rl_active_learning_world_model.py \
    --task-kind mode --design-mode transmit --no-cst \
    --timesteps 100000 --episode-len 25 --selfplay-round-episodes 64 \
    --seed 42 --tag my_run
# or
python -m active_rl.train --task-kind mode ...
# or
python active_rl/train.py --task-kind mode ...
```

Without `--no-cst`, selective CST verification requires the CST TCP server from
the research repository (default port 65433; see `../cst/client.py`).

Hyperparameter ablation (writes per-run logs, then aggregate):

```bash
python active_rl/run_rl_hp_ablation.py --dry-run          # print the 15 commands
python active_rl/run_rl_hp_ablation.py --jobs 3           # run (logs -> <root>/ablation_logs/)
python active_rl/aggregate_rl_hp_from_logs.py \
    --log-dir active_rl/results_world_model/ablation_logs --out hp_ablation.md
```

τ-threshold uncertainty audit (seed-43 protocol used in the response letter):

```bash
python active_rl/tau_monitor.py --help
```

## Notes

- The checkpoints in `server/` are the June-2026 order-16 models; the paper's
  forward-model ensemble and all formal RL agents (200k steps) start from M1
  (`--world-model m1`, the default).
- The clean, self-contained reimplementation of the v11 environment, reward,
  and task builders lives in `../inverse/` (used by `rl_search.py` /
  `ga_search.py` for design search with a trained policy). This folder holds
  the training-side research code; the two coexist by design and are not
  cross-imported.
- Run outputs are isolated under `active_rl/results_world_model/` and do
  not touch the clean reimplementation in `../inverse/`.
