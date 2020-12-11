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
    ],
    zip_safe=False)
