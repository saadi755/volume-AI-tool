
# 📦 Volume AI Tool

> An intelligent Python-based tool that uses **Machine Learning** to suggest optimal coordinate systems for triple integral volume problems — combined with integral verification, volume estimation, and 3D shape visualization.

---

## 📌 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Demo](#demo)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [How It Works](#how-it-works)
  - [Feature Extraction](#1-feature-extraction)
  - [ML Model Training](#2-ml-model-training)
  - [Coordinate System Suggestion](#3-coordinate-system-suggestion)
  - [Integral Verification](#4-integral-verification)
  - [Volume Estimation](#5-volume-estimation)
  - [3D Visualization](#6-3d-visualization)
- [Dataset](#dataset)
- [Technologies Used](#technologies-used)
- [Limitations & Future Work](#limitations--future-work)
- [License](#license)

---

## Overview

The **Volume AI Tool** automates a core multivariable calculus task: determining the most appropriate coordinate system (rectangular, cylindrical, or spherical) for computing triple integrals over 3D regions. It combines:

- **Natural Language Processing** (via regex and keyword extraction) to understand region descriptions
- **Random Forest Classification** (scikit-learn) to predict the best coordinate system
- **Symbolic formula parsing** to estimate volumes of standard solids
- **Matplotlib 3D visualization** to render the described region

This tool is useful for students, educators, and engineers dealing with volume computation problems in multivariable calculus.

---

## Features

| Feature | Description |
|---|---|
| 🔵 Coordinate Suggestion | Predicts the best coordinate system using ML |
| ✅ Integral Verification | Validates the user's integral setup against the predicted system |
| 📐 Volume Estimation | Estimates volume of spheres, cones, and cylinders using standard formulas |
| 🖼️ 3D Visualization | Renders the 3D region with Matplotlib |
| 🤖 ML-Powered | Trained on a JSON dataset of 3D region descriptions |

---

## Demo

**Coordinate Suggestion:**
```
Enter region description: sphere with radius 4
🔵 Suggested: spherical (Confidence: 0.67)
For computing the volume of this region, spherical coordinates are recommended (confidence: 0.67)
```

**3D Visualization:**

The tool renders a live 3D plot of the described shape (sphere, cone, or cylinder) using Matplotlib's interactive figure window.

---

## Project Structure

```
volume-ai-tool/
│
├── coord_trans.py                  # Main Python script (all logic here)
├── expanded_volume_dataset.json    # Training dataset (region descriptions + coordinate labels)
│
├── README.md                       # Project documentation (this file)
│
└── requirements.txt                # Python dependencies
```

### File Descriptions

| File | Purpose |
|---|---|
| `coord_trans.py` | Core application: feature extraction, ML training, coordinate suggestion, integral verification, volume estimation, visualization, and CLI menu |
| `expanded_volume_dataset.json` | JSON dataset with fields: `desc` (region description), `coord` (correct coordinate system), `integral` (integral setup), `volume` (computed volume) |
| `requirements.txt` | Lists all pip-installable dependencies for easy setup |

---

## Installation

### Prerequisites

- Python 3.8 or higher
- pip

### Steps

**1. Clone the repository**

```bash
git clone https://github.com/<your-username>/volume-ai-tool.git
cd volume-ai-tool
```

**2. (Recommended) Create a virtual environment**

```bash
python -m venv venv

# On Windows:
venv\Scripts\activate

# On macOS/Linux:
source venv/bin/activate
```

**3. Install dependencies**

```bash
pip install -r requirements.txt
```

---

## Usage

Run the main script:

```bash
python coord_trans.py
```

You will see an interactive menu:

```
📦 Volume AI Tool
✅ Model trained! Accuracy: 0.XX

What would you like to do?
1. Suggest coordinate system
2. Verify integral setup
3. Estimate volume
4. Visualize region
5. Exit
Enter choice (1-5):
```

### Menu Options

**Option 1 — Suggest coordinate system**

Input a natural language description of a 3D region and get a recommended coordinate system with confidence score.

```
Enter region description: A solid bounded above by the sphere x^2 + y^2 + z^2 = 9 and below by the cone
🔵 Suggested: spherical (Confidence: 0.85)
```

**Option 2 — Verify integral setup**

Input a region description, your chosen coordinate system, and your integral text. The tool will check whether your setup is consistent with the ML recommendation.

```
Enter region description: cylinder with radius 3
Enter coordinate system used (e.g., cylindrical): cylindrical
Enter your integral text: r from 0 to 3, theta from 0 to 2pi, z from 0 to 5, integrand = r
✅ The integral setup appears correct.
```

**Option 3 — Estimate volume**

Provide a plain-text description with the shape name and numeric dimensions. The tool extracts numbers and applies standard formulas.

```
Enter description: cone radius 3 height 5
Estimated volume: 47.1239
```

**Option 4 — Visualize region**

Input a shape description and a 3D Matplotlib figure will open showing the rendered solid.

```
Enter region description to visualize: a sphere with radius 4
```

---

## How It Works

### 1. Feature Extraction

The `extract_features()` function converts a natural language description into a fixed set of boolean features:

| Feature Key | What it detects |
|---|---|
| `sphere` | Keywords: "sphere", "hemisphere" |
| `cylinder` | Keyword: "cylinder" |
| `cone` | Keyword: "cone" |
| `paraboloid` | Keyword: "paraboloid" |
| `ellipsoid` | Keyword: "ellipsoid" |
| `x^2+y^2` | Regex: `x^2 + y^2` pattern |
| `x^2+y^2+z^2` | Regex: `x^2 + y^2 + z^2` pattern |
| `bounded_z` | Phrase: "bounded between z" |

### 2. ML Model Training

- The JSON dataset is loaded and each `desc` field is passed through `extract_features()`.
- The resulting boolean feature vectors are used to train a **Random Forest Classifier** (100 trees, `random_state=42`).
- An 80/20 train-test split is used and accuracy is reported at startup.

### 3. Coordinate System Suggestion

- Features are extracted from the user's description.
- The trained model predicts: `rectangular`, `cylindrical`, or `spherical`.
- `predict_proba()` gives a confidence score alongside the prediction.

### 4. Integral Verification

The `verify_integral_setup()` function:
- Calls `suggest_coordinate_system()` internally.
- Compares the user's chosen system with the model's suggestion.
- Performs basic sanity checks on the integral text (e.g., cylindrical integrals must include `r`; spherical must include `ρ²sin(ϕ)`).
- Returns a verdict and a descriptive message.

### 5. Volume Estimation

`estimate_volume()` uses regex to extract numeric parameters from text and applies:

| Shape | Formula |
|---|---|
| Cone | `(1/3) × π × r² × h` |
| Sphere | `(4/3) × π × r³` |
| Cylinder | `π × r² × h` |

### 6. 3D Visualization

`visualize_region()` uses Matplotlib's 3D toolkit to plot:

- **Sphere** — parametric surface in spherical coords (skyblue)
- **Cylinder** — meshgrid of theta and z (lightgreen)
- **Cone** — tapered cylindrical surface (salmon)

All shapes use fixed default dimensions for visual demonstration.

---

## Dataset

The training data is stored in `expanded_volume_dataset.json`. Each entry has the following structure:

```json
{
  "desc": "Cone with base radius 2 and height 6",
  "coord": "cylindrical",
  "integral": "r from 0 to 2, θ from 0 to 2π, z from 0 to 6 - r, integrand = r",
  "volume": 58.6431
}
```

| Field | Type | Description |
|---|---|---|
| `desc` | string | Natural language description of the 3D region |
| `coord` | string | Correct coordinate system label |
| `integral` | string | Full integral setup for reference |
| `volume` | float | Numerically computed volume |

---

## Technologies Used

| Library | Purpose |
|---|---|
| `numpy` | Numerical computation and meshgrid generation |
| `pandas` | Feature DataFrame construction |
| `scikit-learn` | Random Forest classifier, train-test split, accuracy scoring |
| `matplotlib` | 3D surface plotting |
| `json` | Loading the training dataset |
| `re` | Regex-based feature and parameter extraction |

---

## Limitations & Future Work

**Current Limitations:**
- Volume estimation only supports cones, spheres, and cylinders with numeric radius/height in plain text
- Visualization uses fixed default dimensions regardless of user input
- Integral verification checks are heuristic and limited in scope
- The ML model relies on keyword presence — it cannot reason about complex symbolic descriptions

**Potential Extensions:**
- Integrate a symbolic math engine (e.g., `sympy`) for exact integral computation
- Add support for paraboloids, ellipsoids, and compound regions in visualization
- Replace regex-based parsing with an NLP model or LLM API for richer understanding
- Expand the dataset to improve classification accuracy
- Build a web-based frontend (Flask/Streamlit) for easier access

---

## License

This project is for educational purposes. Feel free to use and extend it.

---

*Built as part of an AI/ML coursework project — combining multivariable calculus with machine learning.*
