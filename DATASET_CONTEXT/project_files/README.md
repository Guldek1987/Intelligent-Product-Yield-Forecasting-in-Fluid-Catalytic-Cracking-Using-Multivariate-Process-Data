# FCCU Dynamic NN Project Files

These are helper files for rebuilding and using the main dataset:

`../../FCCU_DYNAMIC_NN_DATASET.csv`

## What the main dataset is

The main CSV is prepared specifically for the task described in `DYNAMIC NN for FCCU.docx`:

- dynamic FCCU modeling
- product forecasting
- recurrent neural networks such as LSTM or GRU
- chronological train/validation/test splits

It is not just a raw merge of ML-PSE files. It includes:

- readable column names
- scenario and fault metadata
- `operating_phase` labels
- per-scenario `dataset_split` labels
- gasoline proxy and product-yield proxy columns in `wt% of feed`

## Important paths

- `scripts/prepare_ml_pse_fccu.py` - rebuild the main CSV and context files
- `src/fccu/data.py` - Python loader helpers
- `../data/interim/ml_pse_fccu/fccu_dynamic_nn_dataset_dictionary.csv` - full main-dataset column dictionary
- `../data/interim/ml_pse_fccu/modeling_targets.csv` - recommended target columns
- `../data/interim/ml_pse_fccu/recommended_feature_columns.csv` - recommended feature columns

## Rebuild command

Run from the project root:

```bash
python3 DATASET_CONTEXT/project_files/scripts/prepare_ml_pse_fccu.py
```

## Python example

Run from the project root:

```python
from src.fccu.data import load_all_scenarios, default_feature_columns, product_target_columns

df = load_all_scenarios()
features = default_feature_columns(df)
targets = product_target_columns()
```

If `src` is not already on `PYTHONPATH`, use:

```bash
PYTHONPATH=DATASET_CONTEXT/project_files python3 your_script.py
```

## Target interpretation

The source ML-PSE dataset does not directly provide gasoline yield, dry gas yield, and coke yield exactly as in the project document. The prepared dataset therefore exposes the closest measurable targets:

- `gasoline_proxy_wt_pct_of_feed` = `FLN + FHN`
- `light_cycle_oil_wt_pct_of_feed`
- `lpg_wt_pct_of_feed`
- `slurry_wt_pct_of_feed`

These are the recommended starting targets for Dynamic NN experiments.
