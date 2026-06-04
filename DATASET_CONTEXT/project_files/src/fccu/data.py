"""Data-loading helpers for the FCCU Dynamic NN project."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def _find_project_root() -> Path:
    for path in Path(__file__).resolve().parents:
        if (path / "FCCU_DYNAMIC_NN_DATASET.csv").exists() and (path / "DATASET_CONTEXT").exists():
            return path
    raise RuntimeError("Could not find project root with FCCU_DYNAMIC_NN_DATASET.csv")


PROJECT_ROOT = _find_project_root()
DATASET_PATH = PROJECT_ROOT / "FCCU_DYNAMIC_NN_DATASET.csv"
DATASET_CONTEXT_DIR = PROJECT_ROOT / "DATASET_CONTEXT"
INTERIM_DIR = DATASET_CONTEXT_DIR / "data" / "interim" / "ml_pse_fccu"
ALL_SCENARIOS_PATH = DATASET_PATH
SOURCE_VARIABLE_DICTIONARY_PATH = INTERIM_DIR / "variable_dictionary.csv"
SCENARIO_METADATA_PATH = INTERIM_DIR / "scenario_metadata.csv"
MODELING_TARGETS_PATH = INTERIM_DIR / "modeling_targets.csv"
RECOMMENDED_FEATURES_PATH = INTERIM_DIR / "recommended_feature_columns.csv"
DATASET_DICTIONARY_PATH = INTERIM_DIR / "fccu_dynamic_nn_dataset_dictionary.csv"

METADATA_COLUMNS = {
    "dataset_row_id",
    "scenario_id",
    "condition",
    "operating_phase",
    "dataset_split",
    "fault_type",
    "source_file",
    "row_in_scenario",
    "time_min",
    "time_hour",
    "time_day",
    "is_faulty_scenario",
    "fault_start_min",
    "fault_transition_end_min",
    "is_after_fault_start",
    "time_since_fault_start_min",
    "time_since_fault_start_hour",
}


def load_all_scenarios(path: Path | None = None) -> pd.DataFrame:
    """Load the main modeling-ready FCCU dataset."""
    data_path = path or ALL_SCENARIOS_PATH
    return pd.read_csv(data_path, low_memory=False)


def load_scenario(scenario_id: str) -> pd.DataFrame:
    """Load one prepared scenario by scenario_id."""
    path = INTERIM_DIR / "headered" / f"{scenario_id}.csv"
    return pd.read_csv(path, low_memory=False)


def load_source_variable_dictionary() -> pd.DataFrame:
    """Load the original ML-PSE source variable dictionary."""
    return pd.read_csv(SOURCE_VARIABLE_DICTIONARY_PATH)


def load_dataset_dictionary() -> pd.DataFrame:
    """Load the final main-dataset column dictionary."""
    return pd.read_csv(DATASET_DICTIONARY_PATH)


def load_scenario_metadata() -> pd.DataFrame:
    """Load scenario-level labels and fault timing metadata."""
    return pd.read_csv(SCENARIO_METADATA_PATH)


def load_modeling_targets() -> pd.DataFrame:
    """Load recommended forecasting targets for the project."""
    return pd.read_csv(MODELING_TARGETS_PATH)


def load_recommended_features() -> pd.DataFrame:
    """Load the recommended feature list for baseline dynamic models."""
    return pd.read_csv(RECOMMENDED_FEATURES_PATH)


def product_target_columns(*, target_family: str = "yield_proxy") -> list[str]:
    """Return recommended target columns.

    Parameters
    ----------
    target_family:
        `yield_proxy` returns wt% of feed targets.
        `flow` returns direct product flow targets.
        `all` returns both.
    """
    targets = load_modeling_targets()
    if target_family == "all":
        return targets["column_name"].tolist()
    if target_family == "yield_proxy":
        return targets.loc[targets["target_family"].str.contains("yield"), "column_name"].tolist()
    if target_family == "flow":
        return targets.loc[targets["target_family"] == "flow", "column_name"].tolist()
    raise ValueError(f"Unsupported target_family: {target_family}")


def default_feature_columns(frame: pd.DataFrame | None = None) -> list[str]:
    """Return the recommended baseline feature set for dynamic FCCU models."""
    features = load_recommended_features()["column_name"].tolist()
    if frame is None:
        return features
    return [column for column in features if column in frame.columns]


def normal_prefault_subset(frame: pd.DataFrame) -> pd.DataFrame:
    """Return rows from normal operation and pre-fault periods only."""
    return frame.loc[frame["operating_phase"].isin(["normal_operation", "pre_fault"])].copy()


def chronological_split(
    frame: pd.DataFrame,
    *,
    split_column: str = "dataset_split",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split a prepared frame using the precomputed chronological split labels."""
    train = frame.loc[frame[split_column] == "train"].copy()
    validation = frame.loc[frame[split_column] == "validation"].copy()
    test = frame.loc[frame[split_column] == "test"].copy()
    return train, validation, test
