<div align="center">

# 🏭 GNR-VAL v5.2

### Industrial CAD Compliance Validator

**Varroc Eureka 3.0 · Problem Statement 9**

[!\[Live Demo](https://img.shields.io/badge/🚀\_Live\_Demo-Streamlit-FF4B4B?style=for-the-badge)](https://gnr-val-cad-validator-demo.streamlit.app/)
[!\[Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge\&logo=python\&logoColor=white)](https://python.org)
[!\[PyTorch](https://img.shields.io/badge/PyTorch-GNN-EE4C2C?style=for-the-badge\&logo=pytorch\&logoColor=white)](https://pytorch.org)
[!\[Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?style=for-the-badge\&logo=streamlit\&logoColor=white)](https://streamlit.io)

> \*TransformerConv GNN + ISO 2768 / ASME Y14.5 / GD\&T / DFM Rule Engine\*

</div>

\---

## 📋 Table of Contents

* [About](#-about)
* [Live Demo](#-live-demo)
* [Features](#-features)
* [Model Performance](#-model-performance)
* [Screenshots](#-screenshots)
* [Demo Videos](#-demo-videos)
* [Tech Stack](#-tech-stack)
* [Local Setup](#-local-setup)
* [Project Structure](#-project-structure)

\---

## 🔍 About

**GNR-Val v5.2** is an AI-powered industrial CAD compliance validator built for **Varroc Eureka 3.0 (PS-9)**. It combines a **Graph Neural Network (TransformerConv GNN)** with a deterministic **ISO/ASME rule engine** to validate CAD component designs for manufacturing compliance — automatically classifying them as **Compliant**, **Review-Needed**, or **Non-Compliant**.

The system uses a **fusion decision architecture** where ML predictions and rule-based severity scores are combined into a single **Fused Risk Score**, with safety overrides for critical violations.

\---

## 🚀 Live Demo

**👉** [**https://gnr-val-cad-validator-demo.streamlit.app/**](https://gnr-val-cad-validator-demo.streamlit.app/)

No installation required — try it directly in your browser!

\---

## ✨ Features

|Feature|Description|
|-|-|
|🧠 **GNN Validator**|TransformerConv GNN with edge-aware encoding for component graph analysis|
|📐 **Rule Engine**|ISO 2768, ASME Y14.5, GD\&T, DFM compliance checks|
|🔗 **Fusion Scoring**|ML + Rule Engine fused risk score with safety override logic|
|📂 **STEP File Upload**|Parse real CAD `.STEP`/`.STP` files directly|
|🎲 **Random Demo**|Generate 20 synthetic components instantly for testing|
|📊 **Model Analytics**|Live accuracy, precision, recall, F1 + confusion matrix|
|📄 **Export Reports**|Download compliance reports as `.TXT` or `.JSON`|
|🏢 **Multi-Org Support**|Configurable organisation \& product context|

\---

## 📈 Model Performance

<div align="center">

|Metric|Score|
|-|-|
|✅ Accuracy|**98.68%**|
|🎯 Precision (Macro)|**98.94%**|
|🔁 Recall (Macro)|**98.52%**|
|⚖️ F1 Score (Macro)|**98.7%**|

</div>

### Model Architecture

|Component|Details|
|-|-|
|Encoder|3× TransformerConv (edge-aware, edge\_dim=2)|
|Dims|128 → 128 → 64|
|Decoder|Autoencoder (latent→8)|
|Classifier|64 → 32 → 16 → 3|
|Total Params|**116,971**|
|Standards|ISO 2768, ASME Y14.5, GD\&T, DFM|

\---

## 📸 Screenshots

### Main Interface — Component Geometry Input

!\[Main UI](assets/ss1\_main\_ui.png)

### Model Analytics — GNN Performance Dashboard

!\[Model Analytics](assets/ss2\_model\_analytics.png)

### Confusion Matrix

!\[Confusion Matrix](assets/ss3\_confusion\_matrix.png)

### Model Architecture Details

!\[Model Architecture](assets/ss4\_model\_arch.png)

### STEP File Upload Mode

!\[STEP Upload](assets/ss5\_step\_upload.png)

### Random Demo Mode (20 Synthetic Components)

!\[Random Demo](assets/ss6\_random\_demo.png)

### Compliance Report Preview

!\[Report Preview](assets/ss7\_report\_preview.png)

### Violation Log + Export Options

!\[Violations](assets/ss8\_violations.png)

\---

## 🎥 Demo Videos

### Demo 1 — Manual Form Validation

!\[Demo 1](assets/demo1.gif)

### Demo 2 — Model Analytics \& Confusion Matrix

!\[Demo 2](assets/demo2.gif)

### Demo 3 — Random Demo + Export Report

!\[Demo 3](assets/demo3.gif)

\---

## 🛠️ Tech Stack

```
GNR-Val v5.2
├── ML Framework      → PyTorch + PyTorch Geometric (TransformerConv GNN)
├── Graph Processing  → NetworkX
├── Rule Engine       → Custom ISO 2768 / ASME Y14.5 / GD\&T / DFM engine
├── Fusion Layer      → ML + Rule score weighted fusion (fusion.py)
├── Frontend          → Streamlit
├── Visualization     → Plotly, Matplotlib, Seaborn
└── Data              → Pandas, NumPy, Scikit-learn
```

\---

## ⚙️ Local Setup

### Option 1 — Automated Script (Recommended)

```bash
git clone https://github.com/adityadev00/GNR-Val-CAD-Validator.git
cd GNR-Val-CAD-Validator
chmod +x setup\_gnrval.sh
./setup\_gnrval.sh
```

### Option 2 — Manual Setup

```bash
# 1. Clone the repo
git clone https://github.com/adityadev00/GNR-Val-CAD-Validator.git
cd GNR-Val-CAD-Validator

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the Streamlit app
streamlit run app.py

# 4. (Optional) Run the Jupyter Notebook
python -m notebook EurekaPS9\_\_3\_.ipynb
```

### Requirements

* Python 3.10+
* pip packages: see `requirements.txt`
* Models: `gnrval\_final\_combined.pt` + `gnrval\_real\_trained.pth` (included in repo)

\---

## 📁 Project Structure

```
GNR-Val-CAD-Validator/
│
├── app.py                    # Main Streamlit application
├── fusion.py                 # ML + Rule Engine fusion logic
├── EurekaPS9\_\_3\_.ipynb       # Research notebook (training + analysis)
│
├── gnrval\_final\_combined.pt  # Trained GNN model (combined)
├── gnrval\_real\_trained.pth   # Trained GNN model (real data)
│
├── requirements.txt          # Python dependencies
├── setup\_gnrval.sh           # Automated setup script
│
├── assets/                   # Screenshots \& demo GIFs
└── README.md
```

\---

## 🏆 Competition Context

Built for **Varroc Eureka 3.0** — Problem Statement 9 (PS-9).

The challenge required building an intelligent CAD compliance system for industrial manufacturing. GNR-Val tackles this by treating CAD assemblies as **graphs**, applying GNN-based learning for pattern recognition, and layering deterministic engineering rules on top for safety-critical override decisions.

\---

<div align="center">

**GNR-Val v5.2 · TransformerConv GNN + ISO 2768 / ASME Y14.5 Rule Engine · Varroc Eureka 3.0**

Made by [adityadev00](https://github.com/adityadev00)

</div>

