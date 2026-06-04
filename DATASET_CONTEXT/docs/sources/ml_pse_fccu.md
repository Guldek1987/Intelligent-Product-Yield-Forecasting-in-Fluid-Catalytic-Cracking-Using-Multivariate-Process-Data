# ML-PSE FCCU Dataset

Source repository:

https://github.com/ML-PSE/Fluid-Catalytic-Cracking-Unit-Dataset-for-Process-Monitoring-Evaluation

Source page:

https://mlforpse.com/fccu-dataset/

Downloaded repository HEAD:

`f08b8f8586c2c4678d155af27c8e027fc71875a6`

The source page states that each CSV file has 47 columns. The first column is sampling time in minutes. The remaining 46 columns are measured signals in the order shown in the source variable-description table.

Raw files are stored unchanged in `DATASET_CONTEXT/data/raw/ml_pse_fccu/`. Analysis-ready context copies with column names and scenario labels are generated in `DATASET_CONTEXT/data/interim/ml_pse_fccu/`.

The main working dataset is `FCCU_DYNAMIC_NN_DATASET.csv` in the project root.
