# Reviewer-closure tables

These compact CSV files reproduce the principal saved tables from the executed
notebooks. They are published for inspection; no model was retrained while the
repository was reorganized.

- `main_10_seed_runs.csv`: all ten frozen-architecture runs.
- `main_cross_seed_summary.csv`: macro mean, SD, and 95% t intervals.
- `dm_hac_by_seed.csv`: dependence-aware Diebold–Mariano results.
- `paired_wilcoxon_and_mbb.csv`: scenario-paired Wilcoxon and canonical moving-block bootstrap.
- `rolling_origin_protocol.csv`: development-only rolling-origin counts and saved results.
- `rolling_origin_cutoffs.csv`: exact scenario-wise time cutoffs used by the saved protocol.
- `loso_macro_rmse.csv`: held-out-scenario macro RMSE.
- `multi_horizon_macro_rmse.csv`: 1-, 5-, 15-, and 30-minute test RMSE.
- `hyperparameter_search_space.csv`: executed 10-trial Optuna search space.
- `column_provenance.csv`: complete 47-to-75 column accounting.
- `surrogate_fidelity.csv`: separate fidelity audit; not the source of direct Shapley values.
