from setuptools import setup

setup(
    name='qcommunity',
    description=
    'Quantum Local Search framework for graph modularity optimization',
    author='Ruslan Shaydulin',
    author_email='rshaydu@g.clemson.edu',
    packages=['qcommunity'],
    install_requires=[
        'qiskit', 'qiskit_aqua', 'networkx', 'numpy', 'matplotlib', 'joblib',
        'progressbar2', 'SALib', 'nlopt'
        'libensemble @ git+ssh://git@github.com/rsln-s/libensemble_var#egg=libensemble&sha1=0a0d1b2bbf10ef1d8cb596e7e401f9e78db01332',
        'ibmqxbackend @ git+ssh://git@github.com/rsln-s/ibmqxbackend@v1.0-multistart#egg=ibmqxbackend',
    ],
    zip_safe=False)
