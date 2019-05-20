### Reproducing the results of "Multistart Methods for Quantum Approximate Optimization"

To reproduce the results, run `optimize.py` with graphs passed as parameters.

Example:

`./optimize.py -g get_random_partition_graph -l 6 -r 5 --method BOBYQA_NLOPT --seed 1 --graph-generator-seed 1 --backend IBMQX --ansatz-depth 2 --ansatz QAOA --niter 10 --problem modularity`

Try running `./optimize.py -h` to see all available methods.

To run with APOSMM:

`mpirun -np 2 python -m mpi4py optimize.py -g get_random_partition_graph -l 6 -r 5 --method libensemble --localopt-method LN_BOBYQA --mpi --seed 1 --graph-generator-seed 1 --backend IBMQX --ansatz-depth 2 --ansatz QAOA --niter 10 --problem modularity`

See `example.sh` for an example setup for the full benchmark (requires gnu-parallel)
