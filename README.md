# 🧠 Dissecting QRL

[![Paper](https://img.shields.io/badge/Paper-arXiv-blue)](https://arxiv.org/abs/2511.17112)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](./LICENSE)
[![Framework: SimplyQRL](https://img.shields.io/badge/Framework-SimplyQRL-lightgrey)](https://github.com/javier-lazaro/SimplyQRL)

> Companion repository for the paper *Dissecting Quantum Reinforcement Learning: A Systematic Evaluation of Key Components* (Lazaro *et al.*, 2025)

---

## 📘 Overview
This repository contains the experimental setup for the paper *Dissecting Quantum Reinforcement Learning: A Systematic Evaluation of Key Components* (Lazaro *et al.*, 2025).

It implements a systematic, component-wise evaluation of hybrid QRL pipelines under a unified PPO–CartPole framework:

- **Post-PQC inference** → Analysis of Output Reuse (OR) in PQC-based and classical agents
- **Observation embedding** → Comparison of angle-embedding philosophies using Data Reuploading (DR)
- **Ansatz structure** → Entanglement ablations on initially entangled and unentangled ansätze

All experiments are implemented using [SimplyQRL](https://github.com/javier-lazaro/SimplyQRL), a modular benchmarking framework for hybrid quantum reinforcement learning.

For reproducibility, this repository includes a self-contained copy of the SimplyQRL library (see `src/simplyqrl/`), so the paper experiments do not depend on changes to the upstream repository.

---

## ⚙️ Installation
```bash
git clone https://github.com/javier-lazaro/qrl-dissection.git
cd qrl-dissection
poetry install
```

Or use:

```bash
pip install -r requirements.txt
```

The package metadata targets **Python >= 3.12**. For reproducibility, the core training dependencies in `pyproject.toml` are pinned to the versions recorded in the historical development environment associated with the released runs: `gymnasium==1.0.0`, `numpy==2.0.2`, `pennylane==0.40.0`, `pennylane-lightning==0.40.0`, `tensorboard==2.19.0`, `torch==2.6.0`, and `tqdm==4.67.1`.

Keeping PennyLane and PennyLane-Lightning on the same release avoids the plugin version skew present in the original public dependency lock. After changing dependency pins, regenerate `poetry.lock` and `requirements.txt` from `pyproject.toml` rather than editing the lockfile by hand.

---

## 🔬 Reproducibility

### PPO protocol used for the paper

The experiment runners in `src/experiments/` explicitly set the PPO configuration used for the released paper runs. These values should be treated as the experimental protocol rather than inferred from the generic defaults in `src/simplyqrl/ppo.py`.

| Parameter | Value |
|---|---:|
| Environment | `CartPole-v1` |
| Training budget | `100_000` environment steps |
| Seeds | `1`–`10` |
| `learning_rate` | `2.5e-4` |
| `num_envs` | `1` |
| `num_steps` | `512` |
| `anneal_lr` | `True` |
| `gamma` | `0.99` |
| `gae_lambda` | `0.95` |
| `num_minibatches` | `16` |
| `update_epochs` | `4` |
| `norm_adv` | `True` |
| `clip_coef` | `0.2` |
| `clip_vloss` | `True` |
| `ent_coef` | `0.01` |
| `vf_coef` | `0.5` |
| `max_grad_norm` | `0.5` |
| `target_kl` | `None` |

With one environment and `num_steps=512`, each PPO rollout contains 512 transitions. Using 16 minibatches gives a minibatch size of 32.

### Core software environment

The released training logs trace back to a historical development environment whose recorded core versions were PennyLane 0.40.0, PennyLane-Lightning 0.40.0, PyTorch 2.6.0, Gymnasium 1.0.0, NumPy 2.0.2, TensorBoard 2.19.0, and tqdm 4.67.1. These versions are pinned in `pyproject.toml` so that the public runners use the same core software stack instead of relying on a later, mismatched PennyLane/Lightning combination.

### Output Reuse convention

For a PQC output vector with `n_qubits` expectation values, an Output Reuse factor `R` means **R replicas of the complete output vector** are passed to the linear head. The head therefore receives `n_qubits * R` features.

For example, with 4 qubits and `R=4`, the PQC produces 4 expectation values and the linear head receives 16 features. The classical control follows the same operation after feature selection; with 3 selected observation features and `R=4`, its head receives 12 features.

---

## 🚀 Running Experiments
Each script in `src/experiments/` corresponds to a paper section:

| Section | Experiment | Script |
|----------|------------|--------|
| 4.1 | Post-PQC inference | `post-pqc-inference.py` |
| 4.2 | Observation embedding | `pcq-embeddings.py` |
| 4.3 | Ansatz structure | `pqc-entanglement.py` |

Example:
```bash
poetry run python src/experiments/post-pqc-inference.py
```

The scripts expose representative configurations and annotate the values swept in the paper. The released results contain ten seeds per reported configuration.

Results are stored in `results/` and can be inspected with TensorBoard:
```bash
poetry run tensorboard --logdir results/
```

---

## 📚 Citation
If you use this repository or its results, please cite:

```bibtex
@article{lazaro2025dissecting,
  title   = {Dissecting Quantum Reinforcement Learning: A Systematic Evaluation of Key Components},
  author  = {Lazaro, Javier and Vazquez, Juan-Ignacio and Garcia Bringas, Pablo},
  journal = {arXiv preprint arXiv:2511.17112},
  year    = {2025},
  doi     = {10.48550/arXiv.2511.17112},
  url     = {https://arxiv.org/abs/2511.17112}
}
```

and the associated framework:

```bibtex
@inbook{
  author    = {Lazaro, Javier and Vazquez, Juan-Ignacio and Garc{\'i}a Bringas, Pablo},
  title     = {SimplyQRL: A Modular Benchmarking Library for Hybrid Quantum Reinforcement Learning},
  booktitle = {Hybrid Artificial Intelligent Systems},
  publisher = {Springer Nature Switzerland},
  year      = {2025},
  month     = {oct},
  pages     = {239--250},
  ISBN      = {9783032084620},
  ISSN      = {1611-3349},
  DOI       = {10.1007/978-3-032-08462-0_19},
  url       = {https://doi.org/10.1007/978-3-032-08462-0_19}
}
```

---

**License:** MIT © 2025 Javier Lázaro González
