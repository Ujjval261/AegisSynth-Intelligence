# AegisSynth Intelligence

A privacy-first synthetic data platform built with Python and Streamlit.

## Recruiter Snapshot

- Built an end-to-end app to generate privacy-safe synthetic tabular data from real CSV datasets.
- Implemented SDV-based generation using CTGAN, TVAE, and GaussianCopula with configurable training controls.
- Added quality and privacy validation using KS statistical tests and nearest-neighbor distance scoring.
- Delivered production-style usability: auth, auto-tuning, model save/load, and multi-format export (CSV/Excel/JSON/Parquet).

## Overview

AegisSynth helps teams generate synthetic tabular data from real CSV files while preserving statistical utility and reducing direct data exposure risk.

## Core Features

- Email authentication with Firebase (login/signup)
- CSV upload with encoding fallback (`utf-8`, `latin-1`, `ISO-8859-1`, `cp1252`)
- Multi-page workflow:
  - `Home`
  - `Data Explorer`
  - `AI Generator`
  - `Quality Analysis`
  - `Privacy Metrics`
  - `Model Hub`
- Synthetic generation models (SDV single-table):
  - `CTGAN`
  - `TVAE`
  - `GaussianCopula`
- Preprocessing controls:
  - Feature selection
  - High-cardinality column removal
  - Missing value handling
  - Optional numeric normalization
- Smart controls:
  - Presets (`Fast`, `Balanced`, `Max`)
  - Dataset-based Auto-Tune
  - Optional LLM Copilot guidance (OpenAI API)
- Evaluation:
  - Statistical similarity checks (KS test)
  - Distribution comparison plots
  - Nearest-neighbor distance based privacy score
- Output and model operations:
  - Export synthetic data as `CSV`, `Excel`, `JSON`, `Parquet`
  - Save/load trained synthesizer models (`.pkl`)
  - Generation run report download

## Tech Stack

- `streamlit`
- `pandas`
- `numpy`
- `sdv`
- `plotly`
- `scipy`
- `scikit-learn`
- `joblib`
- `openpyxl`
- `pyarrow`

## Installation

```bash
pip install -r requirements.txt
```

## Run

```bash
streamlit run app.py
```

App runs at: `http://localhost:8501`

## Configuration

Use Streamlit secrets (`.streamlit/secrets.toml`) or environment variables.

### Required for authentication

- `firebase_web_api_key` or `FIREBASE_WEB_API_KEY`
- `firebase_database_url` or `FIREBASE_DATABASE_URL`

### Optional for LLM Copilot

- `openai_api_key` or `OPENAI_API_KEY`
- `openai_model` or `OPENAI_MODEL` (default used by app if not provided)

## Usage Flow

1. Sign in / create account
2. Upload CSV dataset
3. Explore data and quality indicators
4. Configure generation (manual, presets, or Auto-Tune)
5. Generate synthetic records
6. Validate quality and privacy metrics
7. Export data or save/load model

## Notes

- Large datasets can increase training time and memory usage
- `openpyxl` is needed for Excel export
- `pyarrow` is needed for Parquet export

## Developer

Built by **Ujjval Dwivedi**
