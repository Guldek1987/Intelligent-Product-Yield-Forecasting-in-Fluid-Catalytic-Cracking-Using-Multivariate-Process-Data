# Dataset context and reproducible preparation

This folder contains only the source provenance, dictionaries, scenario
metadata, preparation code, and lightweight loader required to understand or
rebuild `../FCCU_DYNAMIC_NN_DATASET.csv`.

## Contents

- `data/raw/ml_pse_fccu/` — source license and download metadata.
- `data/interim/ml_pse_fccu/` — 47- and 75-column dictionaries, modeling
  targets, recommended exogenous variables, scenario metadata, and manifest.
- `docs/sources/` — dataset-source and task-alignment notes.
- `project_files/scripts/prepare_ml_pse_fccu.py` — deterministic preparation
  code.
- `project_files/src/fccu/` — loader and feature/target helpers.

The working CSV remains at the repository root so that all four notebooks can
be opened and inspected without path reconfiguration.

## Source

- Repository: <https://github.com/ML-PSE/Fluid-Catalytic-Cracking-Unit-Dataset-for-Process-Monitoring-Evaluation>
- Description: <https://mlforpse.com/fccu-dataset/>
- Downloaded source commit: `f08b8f8586c2c4678d155af27c8e027fc71875a6`

The source data are simulated FCCU trajectories for process-monitoring
evaluation; they are not industrial plant measurements.
