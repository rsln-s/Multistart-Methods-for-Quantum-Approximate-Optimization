### Installation

```
git clone git@github.com:rsln-s/Multistart-Methods-for-Quantum-Approximate-Optimization.git
cd Multistart-Methods-for-Quantum-Approximate-Optimization
pip install -e .
```

See `qcommunity/optimization/REPRODUCE.md` for instructions on how to reproduce the results presented in "Multistart Methods for Quantum Approximate Optimization" 

### Dependencies

#### libE

Library for managing ensemble-like collections of computations (includes APOSMM discussed in the paper). Follow instructions on installation and dependencies in `https://github.com/rsln-s/libensemble_var`.

After you installed all dependencies, this should work:

```
git clone git@github.com:rsln-s/libensemble_var.git
cd libensemble_var
pip install -e .
```

##### IBMQX backend

Backend to run code in Qiskit Aer simulator (includes a QAOA implementation)

```
git clone git@github.com:rsln-s/ibmqxbackend.git
cd ibmqxbackend
git checkout tags/v1.0-multistart
pip install -e .
```
