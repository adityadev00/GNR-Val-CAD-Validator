#!/bin/bash
# ─────────────────────────────────────────────────────────────
# GNR-Val v5.2 — Local Setup Script
# Varroc Eureka 3.0 | Problem Statement 9
# ─────────────────────────────────────────────────────────────
# Run this script once to install all required dependencies.
# Usage:
#   chmod +x setup_gnrval.sh
#   ./setup_gnrval.sh
# ─────────────────────────────────────────────────────────────

set -e

echo "======================================"
echo " GNR-Val v5.2 — Dependency Installer"
echo "======================================"

# 1. Upgrade pip
echo ""
echo "[1/3] Upgrading pip..."
pip install --upgrade pip

# 2. Install core dependencies
echo ""
echo "[2/3] Installing core packages..."
pip install \
  torch \
  torch-geometric \
  networkx \
  scikit-learn \
  ipywidgets \
  pandas \
  numpy \
  matplotlib \
  plotly \
  streamlit \
  pyngrok \
  jupyter \
  notebook

# 3. Enable ipywidgets for Jupyter
echo ""
echo "[3/3] Enabling ipywidgets extension..."
jupyter nbextension enable --py widgetsnbextension --sys-prefix 2>/dev/null || true

echo ""
echo "======================================"
echo " ✅ All dependencies installed!"
echo ""
echo " To run the notebook:"
echo "   jupyter notebook EurekaPS9__3_.ipynb"
echo ""
echo " To run the Streamlit app (after running cells 1-12):"
echo "   streamlit run app.py"
echo "======================================"
