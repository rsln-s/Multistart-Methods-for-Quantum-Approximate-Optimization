#!/bin/bash


ansatz_depth=2
label=test_setup
run=2
seed=5
graph_generator_seed=3
#problem="maxcut --weighted --zero-tol"
#problem="maxcut --weighted"
#problem="maxcut"
problem="modularity"
#problem="maxcut --weighted --restart-local"
niter=1000

graph_gens=( get_random_partition_graph get_random_partition_graph get_random_partition_graph get_connected_caveman_graph get_connected_caveman_graph ) 
lefts=( 6 5 5 3 2)
rights=( 6 5 6 4 6)

for ((i=0;i<${#graph_gens[@]};++i)); do
    g="${graph_gens[i]}"
    l="${lefts[i]}"
    r="${rights[i]}"
    ./worst_edge_removed.py -g $g -l $l -r $r --seed $seed --graph-generator-seed 1
done
