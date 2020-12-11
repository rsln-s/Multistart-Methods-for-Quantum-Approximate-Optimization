### Installation

```
git clone git@github.com:rsln-s/Multistart-Methods-for-Quantum-Approximate-Optimization.git
cd Multistart-Methods-for-Quantum-Approximate-Optimization
pip install -e .
```

See `qcommunity/optimization/REPRODUCE.md` for instructions on how to reproduce the results presented in "Multistart Methods for Quantum Approximate Optimization" 

### Dependency considerations

#### libE

Library for managing ensemble-like collections of computations (includes APOSMM discussed in the paper). It installs automatically from `https://github.com/rsln-s/libensemble_var`. It requires MPI (`brew install open-mpi` on MacOS).

##### IBMQX backend

Backend to run code in Qiskit Aer simulator (includes a QAOA implementation) and it is automatically from `git@github.com:rsln-s/ibmqxbackend.git@v1.0-multistart`.
