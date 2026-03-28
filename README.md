# Predictive Maintenance Model for Aircraft Engines

A deep learning system that predicts the **Remaining Useful Life (RUL)** of turbofan engines using multi-sensor time-series data, enabling proactive maintenance scheduling. Built with a stacked LSTM architecture, interpreted via Gradient Saliency and SHAP, and optimised for edge deployment via TFLite quantisation.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Setup & How to Run](#setup--how-to-run)
- [Results & Metrics](#results--metrics)
- [Interpretability](#interpretability--shap--gradient-saliency)
- [Model Optimisation](#model-optimisation--quantisation)
- [Repository Structure](#repository-structure)
- [Team](#team)

---

## Project Overview

Aircraft engine failures are costly and dangerous. Predictive maintenance — knowing *when* a component will fail before it does — allows airlines and MROs to schedule maintenance proactively rather than reactively.

This project builds an end-to-end pipeline that:
- Ingests raw multi-sensor engine telemetry from the NASA C-MAPSS dataset
- Engineers sliding-window time-series sequences
- Trains a stacked LSTM to regress normalised RUL
- Explains predictions using Gradient Saliency and SHAP
- Exports a quantised TFLite model for edge/embedded deployment

**Course:** UCS321 — AI for Engineers  
**Dataset:** NASA C-MAPSS FD001 (Turbofan Engine Degradation Simulation)

---

## Dataset

The [C-MAPSS dataset](https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/) (Commercial Modular Aero-Propulsion System Simulation) is published by NASA and is the standard benchmark for turbofan RUL prediction.

**Files used:**

| File | Description |
|---|---|
| `train_FD001.txt` | Multi-engine training run-to-failure data |
| `test_FD001.txt` | Test engine sensor readings (truncated) |
| `RUL_FD001.txt` | Ground-truth RUL values for test engines |

**Key parameters:**

| Parameter | Value |
|---|---|
| Operating conditions | 1 (sea level) |
| Fault modes | 1 (HPC degradation) |
| Features used | 3 op settings + 21 sensors = 24 features |
| RUL cap | 125 cycles |
| Window size | 50 cycles |
| Stride | 1 |

> **To download:** Visit the [NASA PCoE data repository](https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/) and place all three `.txt` files in the project root (or `/content/` in Colab).

---

## Model Architecture

A stacked LSTM network trained to regress normalised RUL (0–1 scale, where 1 = full life remaining).

```
Input → (50 timesteps × 24 features)
  ↓
LSTM(128, return_sequences=True)
  ↓
Dropout(0.25)
  ↓
LSTM(64)
  ↓
Dense(64, ReLU)
  ↓
Dropout(0.20)
  ↓
Dense(1)  ← predicted RUL (normalised)
```

**Training configuration:**

| Setting | Value |
|---|---|
| Loss function | Huber |
| Optimiser | Adam (lr = 1e-3) |
| Batch size | 256 |
| Max epochs | 40 |
| Early stopping patience | 6 epochs |
| LR reduction patience | 3 epochs (factor 0.5) |
| Validation strategy | GroupShuffleSplit (15%, engine-level) |
| Mixed precision | float16 (GPU) |

> GroupShuffleSplit is used deliberately so that no engine's cycles appear in both train and validation — preventing data leakage.

---

## Setup & How to Run

### Option A — Google Colab (recommended)

1. Open the notebook in Colab
2. Upload `train_FD001.txt`, `test_FD001.txt`, `RUL_FD001.txt` when prompted
3. Run all cells top to bottom

### Option B — Local

```bash
git clone https://github.com/Abhishek2405079/Predictive-Maintenance-Model-for-Aircraft-Engines.git
cd Predictive-Maintenance-Model-for-Aircraft-Engines
pip install -r requirements.txt
```

Place the three C-MAPSS data files in the project root, then run:

```bash
python predictive_maintenance_model_for_aircraft_engines.py
```

### Requirements

```
tensorflow
numpy
pandas
scikit-learn
matplotlib
seaborn
shap
```

> For exact pinned versions, see `requirements.txt`.

---

## Results & Metrics

Training stopped at **epoch 11** via early stopping (best val_loss: `0.004493`).

| Split | RMSE (norm) | MAE (norm) | RMSE (cycles) | MAE (cycles) |
|---|---|---|---|---|
| Validation | 0.0948 | 0.0726 | 11.85 cycles | 9.08 cycles |
| Test | 0.1185 | 0.0933 | 14.81 cycles | 11.66 cycles |

On the test set, the model predicts RUL to within approximately **±11.7 cycles** on average, against a RUL cap of 125 cycles — a normalised MAE of ~9.3%.

---

## Interpretability — SHAP & Gradient Saliency

Two complementary interpretability methods are used to identify which sensors drive predictions.

### Gradient Saliency

Computes the gradient of the model output with respect to each input timestep and feature. Higher absolute gradient = more influence on the prediction.

**Top 5 features by Gradient Saliency:**

| Rank | Feature | Importance Score |
|---|---|---|
| 1 | sensor_12 | 0.005200 |
| 2 | sensor_9 | 0.004671 |
| 3 | sensor_13 | 0.004236 |
| 4 | sensor_7 | 0.004082 |
| 5 | sensor_4 | 0.003839 |

### SHAP (GradientExplainer)

Uses Shapley values via `shap.GradientExplainer` with a random background sample of 64 training windows. Provides a theoretically grounded attribution of each feature's contribution.

**Top 5 features by SHAP:**

| Rank | Feature | Mean \|SHAP\| |
|---|---|---|
| 1 | sensor_12 | 0.003548 |
| 2 | sensor_13 | 0.003343 |
| 3 | sensor_9 | 0.003285 |
| 4 | sensor_7 | 0.002611 |
| 5 | sensor_14 | 0.002603 |

**Key finding:** Both methods independently agree that `sensor_12`, `sensor_9`, and `sensor_13` are the most influential features — providing high-confidence attribution. In C-MAPSS, these correspond to fan inlet temperature, physical core speed, and bypass ratio related sensors respectively.

---

## Model Optimisation — Quantisation

To make the model deployable on edge hardware (e.g., onboard avionics or maintenance tablets), the model is:

1. Rebuilt using `LSTMCell` + `unroll=True` for TFLite compatibility
2. Weights transferred exactly from the trained model (verified: max prediction diff < 1e-5)
3. Converted to TFLite with `DEFAULT` optimisation (dynamic range quantisation)

**Size results:**

| Format | Size | Reduction |
|---|---|---|
| Original `.keras` model | 2629.14 KB | — |
| Quantized `.tflite` model | 425.08 KB | **83.83%** |

An 83.83% size reduction with no meaningful loss in prediction accuracy makes the model viable for embedded deployment.

---

## Repository Structure

```
Predictive-Maintenance-Model-for-Aircraft-Engines/
├── predictive_maintenance_model_for_aircraft_engines.py   # Full pipeline
├── requirements.txt                                       # Python dependencies
├── .gitignore
└── README.md
```

---

## Team

Built by a team of 5 as part of the **UCS321 — AI for Engineers** course project.

---

## References

- Saxena, A. et al. (2008). *Damage Propagation Modeling for Aircraft Engine Run-to-Failure Simulation.* NASA Ames Research Center.
- [NASA C-MAPSS Dataset](https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/)
- Lundberg, S. & Lee, S.I. (2017). *A Unified Approach to Interpreting Model Predictions.* NeurIPS.
- TensorFlow Lite: [https://www.tensorflow.org/lite](https://www.tensorflow.org/lite)
