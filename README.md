# 🧠 Dissecting QRL

[![Paper](https://img.shields.io/badge/Paper-PDF-blue)](./PFM___Articulo-5.pdf)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](./LICENSE)
[![Framework: SimplyQRL](https://img.shields.io/badge/Framework-SimplyQRL-lightgrey)](https://github.com/javier-lazaro/SimplyQRL)

> Companion repository for the paper *Dissecting Quantum Reinforcement Learning: A Systematic Evaluation of Key Components* (Lazaro *et al.*, 2025)
---

## 📘 Overview
This repository contains the full experimental setup for the paper *Dissecting Quantum Reinforcement Learning: A Systematic Evaluation of Key Components* (Lazaro *et al.*, 2025).  

It implements a systematic, component-wise evaluation of hybrid QRL pipelines under a unified PPO–CartPole framework:

- **Post-PQC inference** → Analysis of Output Reuse (OR) in PQC-based and classical agents  
- **Observation embedding** → Comparison of angle-embedding philosophies using Data Reuploading (DR)  
- **Ansatz structure** → Entanglement ablations on initially entangled and unentangled ansätze  

All experiments are implemented using [SimplyQRL](https://github.com/javierlazaro9/SimplyQRL), a modular benchmarking framework for hybrid quantum reinforcement learning.  

For reproducibility, this repository includes a self-contained copy of the SimplyQRL library (see `src/simplyqrl/`), allowing all experiments to be executed out of the box without external dependencies on the upstream version.

---

## ⚙️ Installation
```bash
git clone https://github.com/javier-lazaro/qrl-dissection.git
cd qrl-dissection
poetry install
```
Or use `pip install -r requirements.txt`.

**Tested on:** Python 3.10+, PyTorch ≥ 2.2, PennyLane ≥ 0.36, Gymnasium ≥ 0.29.

---

## 🚀 Running Experiments
Each script in `src/experiments/` corresponds to a paper section:

| Section | Experiment | Script |
|----------|-------------|---------|
| 4.1 | Post-PQC inference | `post-pqc-inference.py` |
| 4.2 | Observaton embedding | `pcq-embeddings.py` |
| 4.3 | Ansatz structure | `pqc-entanglement.py` |

Example:
```bash
poetry run python src/experiments/post-pqc-inference.py
```

Results are logged in `results/` and viewable with TensorBoard:
```bash
poetry run tensorboard --logdir results/
```

---

## 📚 Citation
If you use this repository or its results, please cite:

```bibtex
@misc{
  author = {Javier Lázaro},
  title  = {Dissecting Quantum Reinforcement Learning: A Systematic Evaluation of Key Components},
  year   = {2025},
  institution = {University of Deusto},
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
