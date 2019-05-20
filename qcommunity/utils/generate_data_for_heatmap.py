#!/usr/bin/env python

# Generates data for 2D heatmap over betas and gammas for Modularity optimization problem

import networkx as nx
import numpy as np
import os
from itertools import product
from collections import defaultdict
import pickle
import argparse
from qcommunity.optimization.obj import get_obj_val, get_obj

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-l", type=int, default=3, help="number of vtx in the left (first) community")
    parser.add_argument("-r", type=int, default=3, help="number of vtx in the right (second) community")
    parser.add_argument("--samples", type=int, default=100, help="number of samples on each axis of the grid")
    parser.add_argument(
        "--graph-generator-seed",
        type=int,
        default=None,
        help="random seed, only used for graph generator")
    parser.add_argument("-g", "--graph-generator", type=str, default="get_barbell_graph", help="graph generator function")
    parser.add_argument(
        "--label",
        type=str,
        help=
        "description of this version of the script. The description is prepended to the filename, so it should not contain any spaces. Default: time stamp"
    )
    parser.add_argument(
        "--weighted", help="if raised, the graph will be randomly weighted", action="store_true")
    parser.add_argument(
        "--problem",
        type=str,
        default="modularity",
        choices=["modularity", "maxcut"],
        help="the problem to be solved on the graph")
    args = parser.parse_args()
    backend_params = {'depth': 1, 'var_form':'QAOA'}

    obj_val, num_parameters = get_obj_val(
        args.graph_generator,
        args.l,
        args.r,
        seed=args.graph_generator_seed,
        obj_params='ndarray',
        backend="IBMQX",
        backend_params=backend_params,
        problem_name=args.problem,
        return_x=False,
        weight=args.weighted) 

    if num_parameters != 2:
        raise ValueError("Do not support num_parameters={} yet!".format(num_parameters))

    if args.label:
        label = args.label
    else:
        import time
        label = time.strftime("%Y%m%d-%H%M%S")

    outname = "/zfs/safrolab/users/rshaydu/quantum/data/heatmaps_ibmqx/{}_{}_left_{}_right_{}_samples_{}_seed_{}.p".format(label, args.graph_generator, args.l, args.r, args.samples, args.graph_generator_seed)
    print(outname)
    if os.path.isfile(outname):
        print('Output file {} already exists! Our job here is done'.format(
            outname))
        sys.exit(0)

    betas = np.linspace(0,np.pi,num=args.samples,endpoint=True) 
    gammas = np.linspace(0,2*np.pi,num=args.samples,endpoint=True)
    res = defaultdict(dict)
    for beta, gamma in product(betas, gammas):
        x = np.array([beta,gamma])
        res[(beta,gamma)]['mean'] = obj_val(x)
        print("Computed modularities for ", beta, gamma, ":", res[(beta,gamma)])
    pickle.dump((res, betas, gammas), open(outname, "wb"))
    print("Dumped pickle to ", outname)
