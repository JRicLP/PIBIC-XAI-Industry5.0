# Explainable Models for Industrial Processes

![Status](https://img.shields.io/badge/Status-In_Development-yellow)

Software artifacts and experiments for the Scientific Initiation project at the University of Pernambuco (UPE).

**Theme:** Explainable Models for Industrial Processes: Advances and Applications of AI in Industry 5.0.

**Goal:** build predictive maintenance pipelines with explainability, auditability and operator-oriented reports for industrial datasets, with emphasis on the NASA C-MAPSS Remaining Useful Life (RUL) problem.

## Problem

The NASA C-MAPSS dataset simulates turbofan engine degradation. The main task is to predict the Remaining Useful Life (RUL) of each engine from operational cycles and sensor readings. The project combines regression models, explainability reports and an LLM assistant to make model outputs easier to inspect and act on.

The NASA pipeline follows the common piecewise linear RUL target used in the literature: RUL values above 125 cycles are clipped to 125. This reduces the influence of very healthy early-life engines and improves comparability with published C-MAPSS work such as Heimes (2008) and Babu et al. (2016).

## Setup

```bash
python -m venv .venv
```

```bash
pip install -r requirements.txt
```

## Configuration

Create a local environment file from the example:

```bash
cp .env.example .env
```

Then set:

```env
GEMINI_API_KEY=sua_chave_aqui
```

The `.env` file is ignored by Git. The public template is `.env.example`.

## Execution

### Modular Pipeline

```bash
python -m src.main
```

The modular pipeline runs ETL, model training, official NASA test-set evaluation, Shapash explainability and the Gemini-based operator report.

### Notebooks

For the NASA workflow, run:

```text
08 -> 09 -> 10 -> 11
```

- `08_etl_nasa_cmaps.ipynb`: loads and processes NASA C-MAPSS data.
- `09_auditoria_tecnica_nasa.ipynb`: demo notebook that calls the modular NASA pipeline from `src/`.
- `10_registro_mlflow_nasa.ipynb`: deprecated historical notebook; MLflow logging now happens in `src/main.py`.
- `11_assistente_llm_operador.ipynb`: generates an operator-oriented report with Gemini.

## Project Architecture

```text
src/
  config.py           Shared paths, constants and model arena
  data_pipeline.py    NASA C-MAPSS ETL
  train.py            Group-aware model training
  evaluation.py       Regression metrics and validation plots
  explainability.py   Shapash reports and critical-engine extraction
  llm_assistant.py    Gemini report for operators

notebooks/
  01-07               AI4I and Hydraulic experiments
  08-11               NASA C-MAPSS workflow

docs/
  nasa/               Regenerable NASA reports and figures
```

The official raw NASA path is `data/raw/nasa/`, exposed in code as `NASA_DATA_DIR`/`RAW_DATA_DIR` in `src.config`. Processed NASA data is stored in `data/processed/nasa/`.

## Results

Current NASA baseline report:

| Dataset | Model | MAE | RMSE |
| --- | --- | ---: | ---: |
| NASA C-MAPSS FD001 | Random Forest Regressor | 29.6888 | 41.5214 |

Validation figure:

```text
docs/nasa/Verificacao_RUL_Real_vs_Predito.png
```

Shapash HTML reports are regenerable artifacts and are ignored by Git.

Narrative NASA analysis:

```text
docs/RESULTADOS_NASA.md
docs/nasa/resultados.json
```

The official NASA evaluation uses `train_FD001.txt` for training and the last cycle of each engine in `test_FD001.txt`, compared against `RUL_FD001.txt`, for MAE, RMSE, R2, NASA Score and MAE by RUL range.

## Explainability

Shapash is used to generate global feature-importance reports and local explanations for the most critical engine in the test set. The modular pipeline returns operational metadata (`engine_id`, `time_cycle`, `rul_predito`) so the report identifies the real engine and cycle, not only a dataframe index.

## LLM Assistant

The Gemini assistant receives the critical-engine diagnosis and the main contributing sensors, then produces a short maintenance-oriented report. API access is configured through `GEMINI_API_KEY` in `.env`; no API key should be committed to the repository.

## Reproducibility

- Dependencies are listed in UTF-8 in `requirements.txt`.
- Notebook outputs are stripped before commit.
- `nbstripout` is installed as a Git filter for this repository.
- Generated data, MLflow runs, local environment files and Shapash HTML artifacts are ignored.

## Bibliography

- NASA C-MAPSS Turbofan Engine Degradation Simulation Data Set.
- Heimes, F. O. (2008). Recurrent neural networks for remaining useful life estimation.
- Babu, G. S., Zhao, P., and Li, X. L. (2016). Deep convolutional neural network based regression approach for estimation of remaining useful life.
- Shapash documentation for model explainability dashboards.
- MLflow documentation for experiment tracking.
- Google GenAI SDK documentation for Gemini integration.

## License

MIT License
