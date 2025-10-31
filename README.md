# 🧠 Ablation on Hybrid QRL

[![Paper](https://img.shields.io/badge/Paper-PDF-blue)](./PFM___Articulo-5.pdf)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](./LICENSE)
[![Framework: SimplyQRL](https://img.shields.io/badge/Framework-SimplyQRL-lightgrey)](https://github.com/javier-lazaro/SimplyQRL)

> Companion repository for the paper *Dissecting Quantum Reinforcement Learning: A Systematic Evaluation of Key Components* (Lazaro et. al., 2025)
---

## 📘 Overview
This repository contains the full experimental setup for the paper *An Ablation Study on Hybrid Quantum Reinforcement Learning* (University of Deusto, 2025). It implements a systematic ablation of hybrid QRL pipelines under a unified PPO–CartPole framework:

- **Post-PQC inference** → Output Reuse (OR)  
- **Observation embedding** → Data Reuploading (DR)  
- **Ansatz structure** → Entanglement ablations

All experiments are implemented using [SimplyQRL](https://github.com/javier-lazaro/SimplyQRL), ensuring consistent PPO training across classical, quantum, and hybrid agents.

---

## ⚙️ Installation
```bash
git clone https://github.com/javierlazaro/Ablation-on-hybrid-QRL.git
cd Ablation-on-hybrid-QRL
poetry install
```
Or use `pip install -r requirements.txt`.

**Tested on:** Python 3.10+, PyTorch ≥ 2.2, PennyLane ≥ 0.36, Gymnasium ≥ 0.29.

---

## 🚀 Running Experiments
Each script in `src/experiments/` corresponds to a paper section:

| Section | Experiment | Script |
|----------|-------------|---------|
| 4.1 | Output Reuse (OR) | `post-pqc-inference.py` |
| 4.2 | Data Reuploading (DR) | `pcq-embeddings.py` |
| 4.3 | Entanglement Ablation | `pqc-entanglement.py` |

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

**License:** MIT © 2025 Javier Lázaro
