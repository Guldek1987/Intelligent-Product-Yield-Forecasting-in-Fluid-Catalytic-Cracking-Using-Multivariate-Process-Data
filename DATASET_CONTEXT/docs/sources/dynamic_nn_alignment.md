# Dynamic NN Alignment

This note explains how `FCCU_DYNAMIC_NN_DATASET.csv` maps the project document
`DYNAMIC NN for FCCU.docx` to the available open ML-PSE FCCU dataset.

## Project objective from the document

- build an adequate dynamic recurrent neural network for FCCU
- forecast product outputs over time
- use chronological train/validation/test splits

## Requested inputs in the document

The document highlights these conceptual inputs:

- reactor temperature
- feed flow rate
- catalyst activity
- feed composition such as API and sulfur content

## What is available in the open ML-PSE dataset

Directly available and already prepared in the main CSV:

- `reactor_riser_temperature_Tr_degF`
- `reactor_riser_temperature_Tr_degC`
- `feed_flow_F3_lb_s`
- `feed_flow_lb_min`
- `feed_flow_tph`
- regenerator, pressure, catalyst circulation, air-flow, fractionator, and control-valve signals

Not directly available in the open ML-PSE dataset:

- catalyst activity coefficient
- feed API gravity
- sulfur content
- direct coke yield
- direct dry-gas yield in wt%

## Target mapping used in the prepared dataset

The document asks for product yields such as gasoline, light gas oil, gas, and coke.
The closest measurable targets available in the source dataset are:

- `gasoline_proxy_wt_pct_of_feed` using `FLN + FHN`
- `light_cycle_oil_wt_pct_of_feed`
- `lpg_wt_pct_of_feed`
- `slurry_wt_pct_of_feed`

These are proxy targets, not exact refinery lab yields.

## Why the main dataset was prepared this way

`FCCU_DYNAMIC_NN_DATASET.csv` includes:

- readable column names
- source and scenario metadata
- `operating_phase`
- `dataset_split`
- product flow targets
- product yield proxies in `wt% of feed`

This makes the dataset usable for sequence modeling without having to repeat
manual preprocessing in every notebook or training script.

