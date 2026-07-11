# MIONet-FSS: A Chebyshev-Encoded Multi-Input Operator Network for Pixelated Frequency Selective Surface Modeling and Active Inverse Design

Deep-learning-driven forward prediction and inverse design of Frequency Selective
Surfaces (FSS) using **MIONet** (Multiple-Input Operator Networks).  The system
learns to predict full S-parameter magnitude curves |S11(f)| and |S21(f)| from
a 10x10 binary metalisation pattern, incident angle, and polarisation (TE/TM),
and uses that surrogate inside RL, GA, and random-search optimisers to discover
layouts that satisfy user-defined filtering tasks.

## Citation

If you use this code in your research, please cite the accompanying paper:

> **MIONet-FSS: A Chebyshev-Encoded Multi-Input Operator Network for Pixelated Frequency Selective Surface Modeling and Active Inverse Design**
>
> Weikai Luo, Da Li, Zhang Pei, Jiahui Wang, Wenzhe Wu, Ruifeng Li, Jinyan Ma, Chongxiao Qu, and Er-Ping Li
>
> *Submitted for publication.*
>
> ```bibtex
> @article{luo2026mionetfss,
>   title   = {{MIONet-FSS}: A {Chebyshev}-Encoded Multi-Input Operator Network
>              for Pixelated Frequency Selective Surface Modeling and Active
>              Inverse Design},
>   author  = {Weikai Luo, ...},
>   journal = {submitted for publication},
>   year    = {2026},
> }
> ```

## Project structure

```
MIONet-FSS/
  code/
    common/          Shared modules: config, MIONet, dataset, loss, basis
    forward/         Training entry point + world-model inference wrapper
    inverse/         RL environment (V11AngleDispersionEnv), reward, task spec, GA/RL search
    cst/             CST Studio Suite TCP client + batch-validation script
    eval/            Pass-rate computation, benchmark runner, summary tables
    results_figures/ Generated plots
  README.md
```

## Two-environment setup

The project requires **two** conda environments because CST Studio Suite only
supports a specific Python version:

### 1. `pytorch` environment (deep learning)

```bash
conda create -n pytorch python=3.10 -y
conda activate pytorch
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install numpy gymnasium matplotlib scipy
```

Used for: training MIONet, RL/GA inverse design, pass-rate evaluation, and all
plotting.

### 2. `cst_gen` environment (CST simulation server)

```bash
conda create -n cst_gen python=3.8 -y
conda activate cst_gen
pip install numpy
```

Used for: the CST TCP server (`code/server/`) and the validation client
(`code/cst/`).  This environment must match the Python version bundled with
CST Studio Suite.

## CST server

Start the CST simulation server (from the `cst_gen` environment):

```bash
conda activate cst_gen
python code/cst_server.py --host 127.0.0.1 --port 65433
```

Stop with `Ctrl+C`.  The server handles one request at a time and caches
results in-memory for the session.

## Quick start

### Train a forward model (M1 MIONet)

```bash
conda activate pytorch
cd MIONet-FSS/code
python forward/train.py --epochs 300 --batch-size 64 --lr 1e-3 --output ./checkpoints
```

Loads the paired TE+TM dataset, splits by geometry, and trains a 3-branch
MIONet with dual-mode physics-aware loss.

### Inverse design (RL)

```bash
conda activate pytorch
cd MIONet-FSS/code
python inverse/rl_search.py --task-kind mode --design-mode transmit --episodes 500
```

Runs PPO on the V11AngleDispersionEnv using the trained MIONet as a surrogate.

### Inverse design (GA baseline)

```bash
conda activate pytorch
cd MIONet-FSS/code
python inverse/ga_search.py --task-kind joint --n-queries 600
```

### CST validation

```bash
conda activate pytorch
cd MIONet-FSS/code
python cst/validate.py --task-kind joint --fc 10 12 14 --bw 1 2 3 4
```

Requires the CST server to be running.

### Plot results

```bash
conda activate pytorch
cd MIONet-FSS/code
python eval/benchmark.py
```

## Module overview

| Module | Purpose |
|---|---|
| `common/config.py` | Global constants: DEVICE, N_FREQ, STANDARD_FREQS, paths |
| `common/basis.py` | Chebyshev / Legendre polynomial frequency basis |
| `common/mionet.py` | MIONet architecture: SpatialBranch, PhysicsBranch, ChebyshevTrunk, HadamardFusion |
| `common/dataset.py` | PyTorch Dataset for dual-pol S-parameters, train/val split by geometry |
| `common/loss.py` | DualModePhysicsLoss: MSE + energy-conservation penalty |
| `forward/train.py` | Training entry point for M1 MIONet |
| `forward/world_model.py` | Inference-only ensemble wrapper for RL/GA evaluation |
| `inverse/env.py` | Gymnasium environment for sequential quarter-pixel placement |
| `inverse/reward.py` | V11 reward: in-band score minus out-band penalty + stability |
| `inverse/task.py` | Task-spec builders for mode, angle, and joint design tasks |
| `inverse/ga_search.py` | Genetic-algorithm and random-search baselines |
| `inverse/rl_search.py` | PPO-based RL search using the V11AngleDispersionEnv |
| `cst/client.py` | TCP client for CST full-wave simulation server |
| `cst/validate.py` | Batch CST validation of RL/GA/dataset designs |
| `eval/pass_rate.py` | Specification-level pass-rate checks with physical dB thresholds |
| `eval/benchmark.py` | Benchmark runner: loads results and computes pass rates |
| `eval/summary_table.py` | LaTeX summary table generation |

## Key design decisions

### Zero inheritance hierarchy

All modules are self-contained.  The V11 reward, environment, and task builders
have no inheritance from older v9/v8 versions.  Constants formerly scattered
across multiple version files are inlined directly into the single file that
uses them.  This eliminates fragile cross-version coupling.

### Dual-environment isolation

The `pytorch` and `cst_gen` environments have zero overlapping dependencies.
The CST client communicates over a language-agnostic TCP socket (port 65433)
using JSON + length-prefix framing, so the two environments can run on
different machines if needed.  The `cst/` package can be copied to the CST
environment for direct access, or kept in `pytorch` and pointed at the server
via `--cst-host`.

### Plot independence

All plotting logic lives in `eval/` as standalone scripts that consume NPZ
result files.  They do not import from `inverse/` or `forward/`, making it
possible to generate publication-ready figures on a headless server with no
CST or GPU access.

### MIONet architecture

Three independent branches -- spatial (CNN), physics (MLP), frequency
(Chebyshev-trunk MLP) -- fused via Hadamard product and decoded through a
rational-function output head.  Polarisation is handled entirely through the
PhysicsBranch as a [pol_TE, pol_TM] one-hot input, eliminating hard-coded
rot90 transforms.

### Forward-model ensemble

The `LocalEnsembleWorldModel` loads a single checkpoint and creates 3 copies
with progressively-scaled Gaussian noise for ensemble diversity, providing
variance-based uncertainty estimates alongside the mean prediction.
