#!/bin/bash

niter=10
ansatz_depths=( 1 2 4 )

problem="modularity9--restart-local"

graph_gens=( get_random_partition_graph get_random_partition_graph get_random_partition_graph get_connected_caveman_graph get_connected_caveman_graph get_connected_caveman_graph ) 
lefts=( 6 5 5 2 3 2)
rights=( 6 5 6 5 4 6)
graph_generator_seeds=( 1 1 1 1 1 1 )

echo "Starting parallel jobs"

parallel \
   --sshloginfile "$PBS_NODEFILE" \
   --jobs 1 \
   --linebuffer \
    """
    g={1}
    l={2}
    r={3}
    graph_generator_seed={4}
    remove_edge={5}
    seed={6}
    ansatz_depth={7}
    postfix='d' # workaround

    label=$label\$ansatz_depth\$postfix

    # environment-specific
    cd /home/rshaydu/quantum/qcommunity_multistart_public/qcommunity/optimization

    source ~/bash_setup/packages_qcommunity.sh
    export PATH="/home/rshaydu/soft/anaconda3/bin:$PATH"
    source activate qcommunity-multistart-public

    mkdir -p libe_stat_files
    problem='${problem//9/ }'

    python optimize.py -g \$g -l \$l -r \$r --method COBYLA_NLOPT --seed \$seed --graph-generator-seed \$graph_generator_seed --backend IBMQX --ansatz-depth \$ansatz_depth --ansatz QAOA --label \$label --niter $niter --verbose --problem \$problem

    python optimize.py -g \$g -l \$l -r \$r --method BOBYQA_NLOPT --seed \$seed --graph-generator-seed \$graph_generator_seed --backend IBMQX --ansatz-depth \$ansatz_depth --ansatz QAOA --label \$label --niter $niter --verbose --problem \$problem

    python optimize.py -g \$g -l \$l -r \$r --method BOBYQA_NLOPT --seed \$seed --graph-generator-seed \$graph_generator_seed --backend IBMQX --ansatz-depth \$ansatz_depth --ansatz QAOA --label \$label --niter $niter --verbose --problem \$problem
    python optimize.py -g \$g -l \$l -r \$r --method NELDERMEAD_NLOPT --seed \$seed --graph-generator-seed \$graph_generator_seed --backend IBMQX --ansatz-depth \$ansatz_depth --ansatz QAOA --label \$label --niter $niter --verbose --problem \$problem
    python optimize.py -g \$g -l \$l -r \$r --method NEWUOA_NLOPT --seed \$seed --graph-generator-seed \$graph_generator_seed --backend IBMQX --ansatz-depth \$ansatz_depth --ansatz QAOA --label \$label --niter $niter --verbose --problem \$problem
    python optimize.py -g \$g -l \$l -r \$r --method PRAXIS_NLOPT --seed \$seed --graph-generator-seed \$graph_generator_seed --backend IBMQX --ansatz-depth \$ansatz_depth --ansatz QAOA --label \$label --niter $niter --verbose --problem \$problem
    python optimize.py -g \$g -l \$l -r \$r --method SBPLX_NLOPT --seed \$seed --graph-generator-seed \$graph_generator_seed --backend IBMQX --ansatz-depth \$ansatz_depth --ansatz QAOA --label \$label --niter $niter --verbose --problem \$problem
    mpirun -np 2 python -m mpi4py optimize.py -g \$g -l \$l -r \$r --method libensemble --localopt-method LN_BOBYQA --mpi --seed \$seed --graph-generator-seed \$graph_generator_seed --backend IBMQX --ansatz-depth \$ansatz_depth --ansatz QAOA --label \$label --max-active-runs 1 --niter $niter --verbose --problem \$problem
    """ ::: "${graph_gens[@]}" :::+ "${lefts[@]}" :::+ "${rights[@]}" :::+ "${graph_generator_seeds[@]}" :::+ "${edges_to_remove[@]}" ::: $(seq 1 10) ::: "${ansatz_depths[@]}"
