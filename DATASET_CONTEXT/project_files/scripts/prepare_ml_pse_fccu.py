"""Prepare the main FCCU Dynamic NN dataset from ML-PSE raw CSV files.

The goal is not only to merge the source files, but to produce one modeling-ready
table aligned with the project document:

- dynamic product forecasting
- chronological train/validation/test splits
- explicit operating phases
- product-flow targets and yield-proxy targets in wt% of feed
- understandable column names for future analysis and model training
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def _find_project_root() -> Path:
    for path in Path(__file__).resolve().parents:
        if (path / "FCCU_DYNAMIC_NN_DATASET.csv").exists() and (path / "DATASET_CONTEXT").exists():
            return path
    raise RuntimeError("Could not find project root with FCCU_DYNAMIC_NN_DATASET.csv")


PROJECT_ROOT = _find_project_root()
DATASET_CONTEXT_DIR = PROJECT_ROOT / "DATASET_CONTEXT"
RAW_CSV_DIR = DATASET_CONTEXT_DIR / "data" / "raw" / "ml_pse_fccu" / "csv"
INTERIM_DIR = DATASET_CONTEXT_DIR / "data" / "interim" / "ml_pse_fccu"
HEADERED_DIR = INTERIM_DIR / "headered"
SOURCE_VARIABLE_DICTIONARY_PATH = INTERIM_DIR / "variable_dictionary.csv"
SCENARIO_METADATA_PATH = INTERIM_DIR / "scenario_metadata.csv"
ALL_SCENARIOS_PATH = PROJECT_ROOT / "FCCU_DYNAMIC_NN_DATASET.csv"
MANIFEST_PATH = INTERIM_DIR / "dataset_manifest.csv"
DATASET_DICTIONARY_PATH = INTERIM_DIR / "fccu_dynamic_nn_dataset_dictionary.csv"
MODELING_TARGETS_PATH = INTERIM_DIR / "modeling_targets.csv"
RECOMMENDED_FEATURES_PATH = INTERIM_DIR / "recommended_feature_columns.csv"

LB_PER_SECOND_TO_LB_PER_MINUTE = 60.0
LB_PER_SECOND_TO_METRIC_TONNE_PER_HOUR = 0.45359237 * 3600.0 / 1000.0

RENAME_MAP = {
    "F3": "feed_flow_F3_lb_s",
    "Tatm": "ambient_temperature_Tatm_degF",
    "T1": "feed_temperature_T1_degF",
    "P4": "reactor_pressure_P4_psia",
    "deltaP": "regenerator_minus_reactor_pressure_deltaP_psig",
    "P6": "regenerator_pressure_P6_psia",
    "Fair": "regenerator_total_air_flow_Fair_mol_s",
    "T3": "furnace_firebox_temperature_T3_degF",
    "T2": "reactor_inlet_feed_temperature_T2_degF",
    "Tr": "reactor_riser_temperature_Tr_degF",
    "Treg": "regenerator_temperature_Treg_degF",
    "Lsp": "standpipe_catalyst_level_Lsp_ft",
    "Tcyc": "stack_gas_temperature_Tcyc_degF",
    "Tcyc_Treg": "stack_gas_minus_regenerator_temperature_Tcyc_Treg_degF",
    "Cco_g": "stack_gas_co_Cco_g_ppm",
    "Co2_g": "stack_gas_o2_Co2_g_mol_pct",
    "P5": "fractionator_overhead_pressure_P5_psia",
    "V4": "valve_fractionator_pressure_V4_pct",
    "V6": "valve_regenerator_temperature_V6_pct",
    "V7": "valve_regenerator_pressure_V7_pct",
    "V3": "valve_reactor_inventory_V3_pct",
    "V1": "valve_preheated_feed_temperature_V1_pct",
    "V2": "valve_reactor_temperature_V2_pct",
    "Frgc": "regenerated_catalyst_flow_Frgc_lb_min",
    "Fsc": "spent_catalyst_flow_Fsc_lb_min",
    "ACAB": "combustion_air_blower_current_ACAB_amp",
    "AWGC": "wet_gas_compressor_current_AWGC_amp",
    "F5": "furnace_fuel_flow_F5_scf_min",
    "F7": "combustion_air_flow_F7_lb_min",
    "Fsg": "stack_gas_flow_Fsg_mol_min",
    "FV11": "wet_gas_compressor_suction_valve_flow_FV11_mol_min",
    "P1": "cab_suction_pressure_P1_psia",
    "P2": "cab_discharge_pressure_P2_psia",
    "FLPG": "lpg_flow_FLPG_lb_min",
    "FLN": "light_naphtha_flow_FLN_lb_min",
    "FHN": "heavy_naphtha_flow_FHN_lb_min",
    "FLCO": "light_cycle_oil_flow_FLCO_lb_min",
    "Fslurry": "slurry_flow_Fslurry_lb_min",
    "Freflux": "reflux_flow_Freflux_lb_min",
    "Tfra": "fractionator_overhead_temperature_Tfra_degF",
    "T10": "fractionator_mid_temperature_T10_degF",
    "T20": "fractionator_bottom_temperature_T20_degF",
    "V9": "valve_accumulator_level_V9_pct",
    "V8": "valve_fractionator_temperature_V8_pct",
    "V10": "valve_heavy_naphtha_temperature_V10_pct",
    "V11": "valve_light_naphtha_temperature_V11_pct",
}

FEATURE_COLUMNS = [
    "ambient_temperature_Tatm_degF",
    "feed_temperature_T1_degF",
    "reactor_inlet_feed_temperature_T2_degF",
    "reactor_riser_temperature_Tr_degF",
    "regenerator_temperature_Treg_degF",
    "reactor_pressure_P4_psia",
    "regenerator_minus_reactor_pressure_deltaP_psig",
    "regenerator_pressure_P6_psia",
    "regenerator_total_air_flow_Fair_mol_s",
    "furnace_firebox_temperature_T3_degF",
    "standpipe_catalyst_level_Lsp_ft",
    "stack_gas_temperature_Tcyc_degF",
    "stack_gas_minus_regenerator_temperature_Tcyc_Treg_degF",
    "stack_gas_co_Cco_g_ppm",
    "stack_gas_o2_Co2_g_mol_pct",
    "fractionator_overhead_pressure_P5_psia",
    "regenerated_catalyst_flow_Frgc_lb_min",
    "spent_catalyst_flow_Fsc_lb_min",
    "combustion_air_blower_current_ACAB_amp",
    "wet_gas_compressor_current_AWGC_amp",
    "furnace_fuel_flow_F5_scf_min",
    "combustion_air_flow_F7_lb_min",
    "wet_gas_compressor_suction_valve_flow_FV11_mol_min",
    "cab_suction_pressure_P1_psia",
    "cab_discharge_pressure_P2_psia",
    "reflux_flow_Freflux_lb_min",
    "fractionator_overhead_temperature_Tfra_degF",
    "fractionator_mid_temperature_T10_degF",
    "fractionator_bottom_temperature_T20_degF",
    "valve_fractionator_pressure_V4_pct",
    "valve_regenerator_temperature_V6_pct",
    "valve_regenerator_pressure_V7_pct",
    "valve_reactor_inventory_V3_pct",
    "valve_preheated_feed_temperature_V1_pct",
    "valve_reactor_temperature_V2_pct",
    "valve_accumulator_level_V9_pct",
    "valve_fractionator_temperature_V8_pct",
    "valve_heavy_naphtha_temperature_V10_pct",
    "valve_light_naphtha_temperature_V11_pct",
]

TARGET_FLOW_COLUMNS = [
    "lpg_flow_FLPG_lb_min",
    "light_naphtha_flow_FLN_lb_min",
    "heavy_naphtha_flow_FHN_lb_min",
    "gasoline_proxy_flow_lb_min",
    "light_cycle_oil_flow_FLCO_lb_min",
    "slurry_flow_Fslurry_lb_min",
]

TARGET_YIELD_COLUMNS = [
    "lpg_wt_pct_of_feed",
    "light_naphtha_wt_pct_of_feed",
    "heavy_naphtha_wt_pct_of_feed",
    "gasoline_proxy_wt_pct_of_feed",
    "light_cycle_oil_wt_pct_of_feed",
    "slurry_wt_pct_of_feed",
]


def _fahrenheit_to_celsius(series: pd.Series) -> pd.Series:
    return (series - 32.0) * (5.0 / 9.0)


def _load_source_columns() -> list[str]:
    variables = pd.read_csv(SOURCE_VARIABLE_DICTIONARY_PATH).sort_values("csv_position")
    columns = variables["column_name"].tolist()
    if len(columns) != 47:
        raise ValueError(f"Expected 47 source columns, got {len(columns)}")
    if columns[0] != "time_min":
        raise ValueError("First source column must be time_min")
    return columns


def _assign_operating_phase(frame: pd.DataFrame, scenario: pd.Series) -> pd.DataFrame:
    frame = frame.copy()
    fault_start = scenario.get("fault_start_min")
    if scenario["condition"] == "normal":
        frame["operating_phase"] = "normal_operation"
    elif pd.notna(fault_start):
        frame["operating_phase"] = "pre_fault"
        frame.loc[frame["time_min"] >= float(fault_start), "operating_phase"] = "post_fault"
    else:
        frame["operating_phase"] = "faulty_unlabeled_transition"
    return frame


def _assign_split(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.sort_values("row_in_scenario").copy()
    n_rows = len(frame)
    train_end = n_rows * 70 // 100
    validation_end = train_end + n_rows * 15 // 100

    split = pd.Series("test", index=frame.index, dtype="object")
    split.iloc[:train_end] = "train"
    split.iloc[train_end:validation_end] = "validation"
    frame["dataset_split"] = split
    return frame


def _prefix_metadata(frame: pd.DataFrame, scenario: pd.Series) -> pd.DataFrame:
    frame = frame.copy()
    condition = scenario["condition"]
    fault_start = scenario.get("fault_start_min")
    fault_transition_end = scenario.get("fault_transition_end_min")
    fault_type = scenario.get("fault_type")
    if pd.isna(fault_type):
        fault_type = "none"

    frame.insert(0, "source_file", scenario["file_name"])
    frame.insert(0, "fault_type", fault_type)
    frame.insert(0, "condition", condition)
    frame.insert(0, "scenario_id", scenario["scenario_id"])
    frame.insert(4, "row_in_scenario", range(1, len(frame) + 1))
    frame.insert(5, "is_faulty_scenario", int(condition == "faulty"))
    frame.insert(6, "fault_start_min", fault_start)
    frame.insert(7, "fault_transition_end_min", fault_transition_end)

    if condition == "faulty" and pd.notna(fault_start):
        frame.insert(8, "is_after_fault_start", (frame["time_min"] >= float(fault_start)).astype(int))
        frame.insert(9, "time_since_fault_start_min", frame["time_min"] - float(fault_start))
    else:
        frame.insert(8, "is_after_fault_start", 0)
        frame.insert(9, "time_since_fault_start_min", float("nan"))

    frame = _assign_operating_phase(frame, scenario)
    frame = _assign_split(frame)
    frame.insert(8, "time_hour", frame["time_min"] / 60.0)
    frame.insert(9, "time_day", frame["time_min"] / 1440.0)
    frame.insert(
        14,
        "time_since_fault_start_hour",
        frame["time_since_fault_start_min"] / 60.0,
    )
    return frame


def _add_derived_columns(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()

    frame["feed_flow_lb_min"] = frame["feed_flow_F3_lb_s"] * LB_PER_SECOND_TO_LB_PER_MINUTE
    frame["feed_flow_tph"] = frame["feed_flow_F3_lb_s"] * LB_PER_SECOND_TO_METRIC_TONNE_PER_HOUR
    frame["reactor_riser_temperature_Tr_degC"] = _fahrenheit_to_celsius(
        frame["reactor_riser_temperature_Tr_degF"]
    )
    frame["gasoline_proxy_flow_lb_min"] = (
        frame["light_naphtha_flow_FLN_lb_min"] + frame["heavy_naphtha_flow_FHN_lb_min"]
    )
    frame["measured_products_flow_lb_min"] = (
        frame["lpg_flow_FLPG_lb_min"]
        + frame["light_naphtha_flow_FLN_lb_min"]
        + frame["heavy_naphtha_flow_FHN_lb_min"]
        + frame["light_cycle_oil_flow_FLCO_lb_min"]
        + frame["slurry_flow_Fslurry_lb_min"]
    )

    frame["lpg_wt_pct_of_feed"] = frame["lpg_flow_FLPG_lb_min"] / frame["feed_flow_lb_min"] * 100.0
    frame["light_naphtha_wt_pct_of_feed"] = (
        frame["light_naphtha_flow_FLN_lb_min"] / frame["feed_flow_lb_min"] * 100.0
    )
    frame["heavy_naphtha_wt_pct_of_feed"] = (
        frame["heavy_naphtha_flow_FHN_lb_min"] / frame["feed_flow_lb_min"] * 100.0
    )
    frame["gasoline_proxy_wt_pct_of_feed"] = (
        frame["gasoline_proxy_flow_lb_min"] / frame["feed_flow_lb_min"] * 100.0
    )
    frame["light_cycle_oil_wt_pct_of_feed"] = (
        frame["light_cycle_oil_flow_FLCO_lb_min"] / frame["feed_flow_lb_min"] * 100.0
    )
    frame["slurry_wt_pct_of_feed"] = (
        frame["slurry_flow_Fslurry_lb_min"] / frame["feed_flow_lb_min"] * 100.0
    )
    frame["measured_products_wt_pct_of_feed"] = (
        frame["measured_products_flow_lb_min"] / frame["feed_flow_lb_min"] * 100.0
    )

    return frame


def _final_column_order() -> list[str]:
    metadata_columns = [
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
    ]
    derived_columns = [
        "feed_flow_lb_min",
        "feed_flow_tph",
        "reactor_riser_temperature_Tr_degC",
        "gasoline_proxy_flow_lb_min",
        "measured_products_flow_lb_min",
        "lpg_wt_pct_of_feed",
        "light_naphtha_wt_pct_of_feed",
        "heavy_naphtha_wt_pct_of_feed",
        "gasoline_proxy_wt_pct_of_feed",
        "light_cycle_oil_wt_pct_of_feed",
        "slurry_wt_pct_of_feed",
        "measured_products_wt_pct_of_feed",
    ]
    renamed_source_columns = [
        "feed_flow_F3_lb_s",
        "ambient_temperature_Tatm_degF",
        "feed_temperature_T1_degF",
        "reactor_inlet_feed_temperature_T2_degF",
        "reactor_riser_temperature_Tr_degF",
        "regenerator_temperature_Treg_degF",
        "reactor_pressure_P4_psia",
        "regenerator_minus_reactor_pressure_deltaP_psig",
        "regenerator_pressure_P6_psia",
        "regenerator_total_air_flow_Fair_mol_s",
        "furnace_firebox_temperature_T3_degF",
        "standpipe_catalyst_level_Lsp_ft",
        "stack_gas_temperature_Tcyc_degF",
        "stack_gas_minus_regenerator_temperature_Tcyc_Treg_degF",
        "stack_gas_co_Cco_g_ppm",
        "stack_gas_o2_Co2_g_mol_pct",
        "fractionator_overhead_pressure_P5_psia",
        "regenerated_catalyst_flow_Frgc_lb_min",
        "spent_catalyst_flow_Fsc_lb_min",
        "combustion_air_blower_current_ACAB_amp",
        "wet_gas_compressor_current_AWGC_amp",
        "furnace_fuel_flow_F5_scf_min",
        "combustion_air_flow_F7_lb_min",
        "stack_gas_flow_Fsg_mol_min",
        "wet_gas_compressor_suction_valve_flow_FV11_mol_min",
        "cab_suction_pressure_P1_psia",
        "cab_discharge_pressure_P2_psia",
        "lpg_flow_FLPG_lb_min",
        "light_naphtha_flow_FLN_lb_min",
        "heavy_naphtha_flow_FHN_lb_min",
        "light_cycle_oil_flow_FLCO_lb_min",
        "slurry_flow_Fslurry_lb_min",
        "reflux_flow_Freflux_lb_min",
        "fractionator_overhead_temperature_Tfra_degF",
        "fractionator_mid_temperature_T10_degF",
        "fractionator_bottom_temperature_T20_degF",
        "valve_fractionator_pressure_V4_pct",
        "valve_regenerator_temperature_V6_pct",
        "valve_regenerator_pressure_V7_pct",
        "valve_reactor_inventory_V3_pct",
        "valve_preheated_feed_temperature_V1_pct",
        "valve_reactor_temperature_V2_pct",
        "valve_accumulator_level_V9_pct",
        "valve_fractionator_temperature_V8_pct",
        "valve_heavy_naphtha_temperature_V10_pct",
        "valve_light_naphtha_temperature_V11_pct",
    ]
    return metadata_columns + derived_columns + renamed_source_columns


def _build_dataset_dictionary() -> pd.DataFrame:
    source_dict = pd.read_csv(SOURCE_VARIABLE_DICTIONARY_PATH)
    source_rows = source_dict[source_dict["column_name"] != "time_min"].copy()
    source_rows["dataset_column"] = source_rows["column_name"].map(RENAME_MAP)
    source_rows["source_column"] = source_rows["column_name"]
    source_rows["column_group"] = source_rows["role_hint"]
    source_rows["recommended_role"] = "feature"
    source_rows.loc[
        source_rows["dataset_column"].isin(
            {
                "lpg_flow_FLPG_lb_min",
                "light_naphtha_flow_FLN_lb_min",
                "heavy_naphtha_flow_FHN_lb_min",
                "light_cycle_oil_flow_FLCO_lb_min",
                "slurry_flow_Fslurry_lb_min",
            }
        ),
        "recommended_role",
    ] = "target_flow_candidate"
    source_rows.loc[
        source_rows["dataset_column"].isin({"stack_gas_flow_Fsg_mol_min"}),
        "recommended_role",
    ] = "auxiliary_output"
    source_rows.loc[
        source_rows["dataset_column"].isin({"reflux_flow_Freflux_lb_min"}),
        "recommended_role",
    ] = "feature_recycle"

    source_rows = source_rows.rename(
        columns={
            "unit": "unit",
            "description": "description",
        }
    )
    source_rows = source_rows[
        ["dataset_column", "source_column", "unit", "column_group", "recommended_role", "description"]
    ]

    derived_rows = [
        {
            "dataset_column": "dataset_row_id",
            "source_column": "",
            "unit": "",
            "column_group": "metadata",
            "recommended_role": "metadata",
            "description": "Unique row identifier in the final main dataset.",
        },
        {
            "dataset_column": "scenario_id",
            "source_column": "",
            "unit": "",
            "column_group": "metadata",
            "recommended_role": "metadata",
            "description": "Scenario identifier derived from scenario metadata.",
        },
        {
            "dataset_column": "condition",
            "source_column": "",
            "unit": "",
            "column_group": "metadata",
            "recommended_role": "metadata",
            "description": "Scenario condition label, normal or faulty.",
        },
        {
            "dataset_column": "operating_phase",
            "source_column": "",
            "unit": "",
            "column_group": "metadata",
            "recommended_role": "metadata",
            "description": "normal_operation, pre_fault, or post_fault according to scenario timing.",
        },
        {
            "dataset_column": "dataset_split",
            "source_column": "",
            "unit": "",
            "column_group": "metadata",
            "recommended_role": "metadata",
            "description": "Chronological split per scenario: train, validation, or test.",
        },
        {
            "dataset_column": "fault_type",
            "source_column": "",
            "unit": "",
            "column_group": "metadata",
            "recommended_role": "metadata",
            "description": "Fault label from scenario metadata.",
        },
        {
            "dataset_column": "source_file",
            "source_column": "",
            "unit": "",
            "column_group": "metadata",
            "recommended_role": "metadata",
            "description": "Original raw CSV filename.",
        },
        {
            "dataset_column": "row_in_scenario",
            "source_column": "",
            "unit": "row_index",
            "column_group": "metadata",
            "recommended_role": "metadata",
            "description": "Chronological row number inside the scenario.",
        },
        {
            "dataset_column": "time_min",
            "source_column": "time_min",
            "unit": "min",
            "column_group": "metadata",
            "recommended_role": "time_index",
            "description": "Sampling time in minutes from the source dataset.",
        },
        {
            "dataset_column": "time_hour",
            "source_column": "time_min",
            "unit": "hour",
            "column_group": "metadata",
            "recommended_role": "time_index",
            "description": "Sampling time converted to hours.",
        },
        {
            "dataset_column": "time_day",
            "source_column": "time_min",
            "unit": "day",
            "column_group": "metadata",
            "recommended_role": "time_index",
            "description": "Sampling time converted to days.",
        },
        {
            "dataset_column": "is_faulty_scenario",
            "source_column": "",
            "unit": "binary",
            "column_group": "metadata",
            "recommended_role": "metadata",
            "description": "1 for faulty scenarios and 0 for normal scenarios.",
        },
        {
            "dataset_column": "fault_start_min",
            "source_column": "",
            "unit": "min",
            "column_group": "metadata",
            "recommended_role": "metadata",
            "description": "Fault start time in minutes when available.",
        },
        {
            "dataset_column": "fault_transition_end_min",
            "source_column": "",
            "unit": "min",
            "column_group": "metadata",
            "recommended_role": "metadata",
            "description": "Fault transition end time in minutes when available.",
        },
        {
            "dataset_column": "is_after_fault_start",
            "source_column": "",
            "unit": "binary",
            "column_group": "metadata",
            "recommended_role": "metadata",
            "description": "1 if the row is at or after the fault start time.",
        },
        {
            "dataset_column": "time_since_fault_start_min",
            "source_column": "",
            "unit": "min",
            "column_group": "metadata",
            "recommended_role": "metadata",
            "description": "Time relative to fault start in minutes; negative values are pre-fault rows.",
        },
        {
            "dataset_column": "time_since_fault_start_hour",
            "source_column": "",
            "unit": "hour",
            "column_group": "metadata",
            "recommended_role": "metadata",
            "description": "Time relative to fault start in hours.",
        },
        {
            "dataset_column": "feed_flow_lb_min",
            "source_column": "F3",
            "unit": "lb/min",
            "column_group": "derived_helper",
            "recommended_role": "helper",
            "description": "Fresh feed flow converted from lb/s to lb/min for yield calculations.",
        },
        {
            "dataset_column": "feed_flow_tph",
            "source_column": "F3",
            "unit": "metric_ton/h",
            "column_group": "derived_helper",
            "recommended_role": "helper",
            "description": "Fresh feed flow converted to metric tonnes per hour to align with the project document.",
        },
        {
            "dataset_column": "reactor_riser_temperature_Tr_degC",
            "source_column": "Tr",
            "unit": "degC",
            "column_group": "derived_helper",
            "recommended_role": "helper",
            "description": "Reactor riser temperature converted from degF to degC to align with the project document.",
        },
        {
            "dataset_column": "gasoline_proxy_flow_lb_min",
            "source_column": "FLN+FHN",
            "unit": "lb/min",
            "column_group": "target_flow",
            "recommended_role": "target_flow_candidate",
            "description": "Gasoline proxy flow defined as light naphtha plus heavy naphtha.",
        },
        {
            "dataset_column": "measured_products_flow_lb_min",
            "source_column": "FLPG+FLN+FHN+FLCO+Fslurry",
            "unit": "lb/min",
            "column_group": "derived_helper",
            "recommended_role": "helper",
            "description": "Sum of measured mass-based product streams available in the source dataset.",
        },
        {
            "dataset_column": "lpg_wt_pct_of_feed",
            "source_column": "FLPG/F3",
            "unit": "wt_pct_of_feed",
            "column_group": "target_yield_proxy",
            "recommended_role": "target_yield_candidate",
            "description": "LPG flow expressed as weight percent of feed using mass flow ratio.",
        },
        {
            "dataset_column": "light_naphtha_wt_pct_of_feed",
            "source_column": "FLN/F3",
            "unit": "wt_pct_of_feed",
            "column_group": "target_yield_proxy",
            "recommended_role": "target_yield_candidate",
            "description": "Light naphtha flow expressed as weight percent of feed.",
        },
        {
            "dataset_column": "heavy_naphtha_wt_pct_of_feed",
            "source_column": "FHN/F3",
            "unit": "wt_pct_of_feed",
            "column_group": "target_yield_proxy",
            "recommended_role": "target_yield_candidate",
            "description": "Heavy naphtha flow expressed as weight percent of feed.",
        },
        {
            "dataset_column": "gasoline_proxy_wt_pct_of_feed",
            "source_column": "(FLN+FHN)/F3",
            "unit": "wt_pct_of_feed",
            "column_group": "target_yield_proxy",
            "recommended_role": "target_yield_candidate",
            "description": "Gasoline proxy yield in wt% of feed using light plus heavy naphtha.",
        },
        {
            "dataset_column": "light_cycle_oil_wt_pct_of_feed",
            "source_column": "FLCO/F3",
            "unit": "wt_pct_of_feed",
            "column_group": "target_yield_proxy",
            "recommended_role": "target_yield_candidate",
            "description": "Light cycle oil flow expressed as weight percent of feed.",
        },
        {
            "dataset_column": "slurry_wt_pct_of_feed",
            "source_column": "Fslurry/F3",
            "unit": "wt_pct_of_feed",
            "column_group": "target_yield_proxy",
            "recommended_role": "target_yield_candidate",
            "description": "Slurry flow expressed as weight percent of feed.",
        },
        {
            "dataset_column": "measured_products_wt_pct_of_feed",
            "source_column": "(FLPG+FLN+FHN+FLCO+Fslurry)/F3",
            "unit": "wt_pct_of_feed",
            "column_group": "derived_helper",
            "recommended_role": "helper",
            "description": "Sum of measured product streams expressed as weight percent of feed.",
        },
    ]

    dictionary = pd.concat([pd.DataFrame(derived_rows), source_rows], ignore_index=True)
    order = {column: idx for idx, column in enumerate(_final_column_order())}
    dictionary["column_order"] = dictionary["dataset_column"].map(order)
    dictionary = dictionary.sort_values(["column_order", "dataset_column"]).drop(columns=["column_order"])
    return dictionary


def _build_modeling_targets() -> pd.DataFrame:
    rows = [
        {
            "column_name": "gasoline_proxy_wt_pct_of_feed",
            "unit": "wt_pct_of_feed",
            "target_family": "yield_proxy",
            "recommended_priority": "primary",
            "description": "Closest available proxy to gasoline yield in the project document.",
        },
        {
            "column_name": "light_cycle_oil_wt_pct_of_feed",
            "unit": "wt_pct_of_feed",
            "target_family": "yield_proxy",
            "recommended_priority": "primary",
            "description": "Closest available proxy to light gas oil yield using FLCO.",
        },
        {
            "column_name": "lpg_wt_pct_of_feed",
            "unit": "wt_pct_of_feed",
            "target_family": "yield_proxy",
            "recommended_priority": "secondary",
            "description": "LPG yield proxy in wt% of feed.",
        },
        {
            "column_name": "slurry_wt_pct_of_feed",
            "unit": "wt_pct_of_feed",
            "target_family": "yield_proxy",
            "recommended_priority": "secondary",
            "description": "Slurry yield proxy in wt% of feed.",
        },
        {
            "column_name": "light_naphtha_wt_pct_of_feed",
            "unit": "wt_pct_of_feed",
            "target_family": "yield_proxy_component",
            "recommended_priority": "secondary",
            "description": "Light naphtha contribution to the gasoline proxy target.",
        },
        {
            "column_name": "heavy_naphtha_wt_pct_of_feed",
            "unit": "wt_pct_of_feed",
            "target_family": "yield_proxy_component",
            "recommended_priority": "secondary",
            "description": "Heavy naphtha contribution to the gasoline proxy target.",
        },
        {
            "column_name": "gasoline_proxy_flow_lb_min",
            "unit": "lb/min",
            "target_family": "flow",
            "recommended_priority": "secondary",
            "description": "Gasoline proxy flow defined as FLN plus FHN.",
        },
        {
            "column_name": "light_cycle_oil_flow_FLCO_lb_min",
            "unit": "lb/min",
            "target_family": "flow",
            "recommended_priority": "secondary",
            "description": "Light cycle oil flow target available directly from the source dataset.",
        },
        {
            "column_name": "lpg_flow_FLPG_lb_min",
            "unit": "lb/min",
            "target_family": "flow",
            "recommended_priority": "secondary",
            "description": "LPG flow target available directly from the source dataset.",
        },
        {
            "column_name": "slurry_flow_Fslurry_lb_min",
            "unit": "lb/min",
            "target_family": "flow",
            "recommended_priority": "secondary",
            "description": "Slurry flow target available directly from the source dataset.",
        },
    ]
    return pd.DataFrame(rows)


def _build_recommended_features() -> pd.DataFrame:
    descriptions = {
        "ambient_temperature_Tatm_degF": "Ambient disturbance variable available in both normal scenarios.",
        "feed_temperature_T1_degF": "Fresh feed temperature before the reactor train.",
        "reactor_inlet_feed_temperature_T2_degF": "Temperature of feed entering the reactor.",
        "reactor_riser_temperature_Tr_degF": "Main reactor temperature proxy aligned with the project document.",
        "regenerator_temperature_Treg_degF": "Core FCCU thermal state variable.",
        "reactor_pressure_P4_psia": "Reactor pressure state.",
        "regenerator_minus_reactor_pressure_deltaP_psig": "Pressure difference related to circulation and fault scenarios.",
        "regenerator_pressure_P6_psia": "Regenerator pressure state.",
        "regenerator_total_air_flow_Fair_mol_s": "Total regenerator air flow.",
        "furnace_firebox_temperature_T3_degF": "Heating-system state variable.",
        "standpipe_catalyst_level_Lsp_ft": "Catalyst inventory proxy.",
        "stack_gas_temperature_Tcyc_degF": "Stack gas thermal response.",
        "stack_gas_minus_regenerator_temperature_Tcyc_Treg_degF": "Temperature-difference feature from source data.",
        "stack_gas_co_Cco_g_ppm": "Combustion quality indicator.",
        "stack_gas_o2_Co2_g_mol_pct": "Oxygen concentration indicator.",
        "fractionator_overhead_pressure_P5_psia": "Downstream fractionator pressure state.",
        "regenerated_catalyst_flow_Frgc_lb_min": "Regenerated catalyst circulation feature.",
        "spent_catalyst_flow_Fsc_lb_min": "Spent catalyst circulation feature.",
        "combustion_air_blower_current_ACAB_amp": "Equipment loading signal for the air blower.",
        "wet_gas_compressor_current_AWGC_amp": "Equipment loading signal for the wet gas compressor.",
        "furnace_fuel_flow_F5_scf_min": "Fuel input to the furnace.",
        "combustion_air_flow_F7_lb_min": "Combustion air flow input.",
        "wet_gas_compressor_suction_valve_flow_FV11_mol_min": "Wet gas compressor suction-valve flow measurement.",
        "cab_suction_pressure_P1_psia": "CAB suction pressure state.",
        "cab_discharge_pressure_P2_psia": "CAB discharge pressure state.",
        "reflux_flow_Freflux_lb_min": "Fractionator recycle stream useful for dynamic response modeling.",
        "fractionator_overhead_temperature_Tfra_degF": "Overhead fractionator temperature.",
        "fractionator_mid_temperature_T10_degF": "Mid-stage fractionator temperature.",
        "fractionator_bottom_temperature_T20_degF": "Bottom-stage fractionator temperature.",
        "valve_fractionator_pressure_V4_pct": "Pressure controller opening.",
        "valve_regenerator_temperature_V6_pct": "Regenerator temperature controller opening.",
        "valve_regenerator_pressure_V7_pct": "Regenerator pressure controller opening.",
        "valve_reactor_inventory_V3_pct": "Reactor inventory controller opening.",
        "valve_preheated_feed_temperature_V1_pct": "Preheated feed temperature controller opening.",
        "valve_reactor_temperature_V2_pct": "Reactor temperature controller opening.",
        "valve_accumulator_level_V9_pct": "Accumulator level controller opening.",
        "valve_fractionator_temperature_V8_pct": "Fractionator temperature controller opening.",
        "valve_heavy_naphtha_temperature_V10_pct": "Heavy naphtha temperature controller opening.",
        "valve_light_naphtha_temperature_V11_pct": "Light naphtha temperature controller opening.",
    }
    return pd.DataFrame(
        [
            {"column_name": column, "feature_group": "recommended_feature", "description": descriptions[column]}
            for column in FEATURE_COLUMNS
        ]
    )


def prepare() -> pd.DataFrame:
    source_columns = _load_source_columns()
    scenarios = pd.read_csv(SCENARIO_METADATA_PATH)
    HEADERED_DIR.mkdir(parents=True, exist_ok=True)

    prepared_frames: list[pd.DataFrame] = []
    manifest_rows: list[dict[str, object]] = []

    for _, scenario in scenarios.iterrows():
        raw_path = RAW_CSV_DIR / scenario["file_name"]
        if not raw_path.exists():
            raise FileNotFoundError(raw_path)

        frame = pd.read_csv(raw_path, header=None)
        if frame.shape[1] != len(source_columns):
            raise ValueError(
                f"{raw_path.name} has {frame.shape[1]} columns, expected {len(source_columns)}"
            )

        frame.columns = source_columns
        frame = _prefix_metadata(frame, scenario)
        frame = frame.rename(columns=RENAME_MAP)
        frame = _add_derived_columns(frame)

        output_path = HEADERED_DIR / f"{scenario['scenario_id']}.csv"
        prepared_frames.append(frame)

        split_counts = frame["dataset_split"].value_counts().to_dict()
        manifest_rows.append(
            {
                "scenario_id": scenario["scenario_id"],
                "source_file": scenario["file_name"],
                "condition": scenario["condition"],
                "rows": len(frame),
                "columns": len(_final_column_order()),
                "time_min_first": frame["time_min"].min(),
                "time_min_last": frame["time_min"].max(),
                "train_rows": split_counts.get("train", 0),
                "validation_rows": split_counts.get("validation", 0),
                "test_rows": split_counts.get("test", 0),
                "output_file": str(output_path.relative_to(PROJECT_ROOT)),
            }
        )

    all_scenarios = pd.concat(prepared_frames, ignore_index=True)
    all_scenarios.insert(0, "dataset_row_id", range(1, len(all_scenarios) + 1))
    all_scenarios = all_scenarios[_final_column_order()]

    for scenario_id, group in all_scenarios.groupby("scenario_id", sort=False):
        output_path = HEADERED_DIR / f"{scenario_id}.csv"
        group.to_csv(output_path, index=False)

    all_scenarios.to_csv(ALL_SCENARIOS_PATH, index=False)
    pd.DataFrame(manifest_rows).to_csv(MANIFEST_PATH, index=False)
    _build_dataset_dictionary().to_csv(DATASET_DICTIONARY_PATH, index=False)
    _build_modeling_targets().to_csv(MODELING_TARGETS_PATH, index=False)
    _build_recommended_features().to_csv(RECOMMENDED_FEATURES_PATH, index=False)
    return all_scenarios


if __name__ == "__main__":
    result = prepare()
    print(f"Wrote {ALL_SCENARIOS_PATH.relative_to(PROJECT_ROOT)}")
    print(f"Rows: {len(result)}")
    print(f"Columns: {result.shape[1]}")
