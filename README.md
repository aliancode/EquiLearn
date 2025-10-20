
# ⚖️ EquiLearn

**A Multi-Objective Framework for Culturally Responsive AI Tutors**
📚 *By Siham Rebbah – Faculty of Sciences, Chouaib Doukkali University*
📧 [rebbah.s444@ucd.ac.ma]

---

## 🌍 Overview

**EquiLearn** is a lightweight yet powerful framework that operationalizes *cultural fairness* and *epistemic positionality* in AI-driven tutoring systems.
It extends traditional **Knowledge Tracing (KT)** models (like DKT) with a **multi-objective learning scheme** that balances:

1. 🎯 **Mastery** — student learning prediction accuracy
2. ⚖️ **Fairness** — reduced cultural bias via *Epistemic Gini* and *Cultural Adaptation Feedback*

EquiLearn integrates **Khan Academy multilingual data** and **EdNet-KT1** student traces to simulate and optimize equitable AI tutoring at scale.

---

## 🧠 Core Features

| Component                                    | Description                                                                        |
| -------------------------------------------- | ---------------------------------------------------------------------------------- |
| 🧩 **Epistemic Positionality Vector (EPV)**  | Encodes cultural, cognitive, and expressive dimensions of each learning resource.  |
| 💬 **Curriculum Negotiation Protocol (CNP)** | Adapts content dynamically when students flag material as “not relatable.”         |
| ⚙️ **Multi-Objective RL (MORL)**             | Learns to trade off between mastery (AUC) and fairness (variance minimization).    |
| 📊 **Fairness Metrics**                      | Uses *Epistemic Gini* and *Group-wise RMSE* to quantify representational equity.   |
| 🔁 **Reproducible Experiments**              | Fully deterministic seeds, result saving, and fairness-performance visualizations. |

---

## 📂 Repository Structure

```
EquiLearn/
│
├── EquiLearn_Data/              # Local data folder (required)
│   ├── khan_metadata.csv        # Khan Academy metadata (with EPV columns)
│   └── ednet_sample.csv         # EdNet-KT1 user interactions
│
├── EquiLearn.py                 # Main experiment file (run this)
├── results_equilearn.csv        # Output metrics (generated automatically)
├── plots/                       # Saved AUC & fairness visualizations
└── README.md                    # Project documentation
```

---

## ⚙️ Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/EquiLearn.git
cd EquiLearn

# (Optional) create a virtual environment
python -m venv env
source env/bin/activate  # or env\Scripts\activate on Windows

# Install dependencies
pip install -r requirements.txt
```

### requirements.txt

```
torch
numpy
pandas
scipy
scikit-learn
matplotlib
```

---

## 🚀 Usage

Place your real datasets in the folder:

```
EquiLearn_Data/
```

Then run:

```bash
python EquiLearn.py
```

Outputs:

* ✅ `results_equilearn.csv` — mastery & fairness metrics
* 📈 plots for AUC, fairness, and equity curves

---

## 📊 Example Results

| Model         | AUC ↑     | Fairness (1–Gini) ↑ | ΔFairness   | ΔAUC   |
| ------------- | --------- | ------------------- | ----------- | ------ |
| DKT Baseline  | 0.742     | 0.693               | —           | —      |
| **EquiLearn** | **0.734** | **0.875**           | **+26.2 %** | −1.1 % |

> ⚖️ EquiLearn achieves **substantially higher cultural fairness** with minimal impact on accuracy.

---

## 📜 Citation

If you use this framework, please cite:

```
@article{Rebbah2025EquiLearn,
  author    = {Siham Rebbah},
  title     = {EquiLearn: A Multi-Objective Framework for Culturally Responsive AI Tutors},
  journal   = {International Journal of Emerging Technologies in Learning (iJET)},
  year      = {2025},
  note      = {Under Review}
}
```

---

## 🧩 Acknowledgments

* **EdNet** dataset – Riiid AI Research (2020)
* **Khan Academy multilingual content** (public repository)
* **Faculty of Sciences, Chouaib Doukkali University**



## 💡 Future Work

* Integrate **transformer-based fairness adaptation**
* Evaluate on **real multilingual classroom data**
* Extend to **EquiLearn-2.0 (LLM alignment version)**

