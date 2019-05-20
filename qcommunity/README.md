# Workflows

Use at your own risk.

## To test the quality of a refinement

1. Update `modularity/single_level_refinement.py` 
2. Use `palmetto_run/run_refinement.sh` and `palmetto_run/refinement.pbs` to run with different parameters
3. Use `utils/explore_refinement_results.py` to compute statistics

## To generate training data for QAOA parameter optimization

1. Update `modularity/single_level_refinement_dump_boundary.py` 
2. Run it using  `palmetto_run/run_refinement_dump_boundary.sh` and `palmetto_run/refinement_dump_boundary.pbs` to generate problem parameters.
3. Run `utils/generate_data_for_heatmap_refinement.py` using `palmetto_run/run_generate_heatmaps.sh` and `palmetto_run/generate_heatmaps.pbs` to generate the full data.
4. Run `utils/aggregate_training_data.py` using `palmetto_run/aggregate_training.pbs` to aggregate the data produced in the previous step into whatever nice format you prefer. 

## To get a dictionary of guesses for bayesian

1. Run `utils/aggregate_training_data.py --best`
2. Put the generated pickle in `../bin` folder
3. Modify the filepath in `optimization/bayesian.py` TODO make filepath an optional parameter

## To benchmark different optimization algorithms

1. Add a new optimizer to `optimize.py`
2. Use `run_optimize.sh` and `optimize.pbs` to run benchmark on Palmetto
3. Use `utils/aggregate_optimization_results.py` to aggregate the results
4. Use `utils/explore_results/explore_optimize_results.py` to look at the results
