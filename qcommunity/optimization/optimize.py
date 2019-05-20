#!/usr/bin/env python

# QAOA parameter optimization

# Example: mpirun -np 2 python -m mpi4py optimize.py -g get_connected_watts_strogatz_graph -l 12 -r 5 --method libensemble --mpi --backend IBMQX
# Example: ./optimize.py -g get_connected_watts_strogatz_graph -l 12 -r 5 --method neldermead
# Example: ./optimize.py -g get_connected_watts_strogatz_graph -l 12 -r 5 --method COBYLA --niter 100 --backend IBMQX

import pickle
import numpy as np
import os.path
import sys
import argparse
import warnings
import random
import logging
import nlopt
from operator import itemgetter
from SALib.sample import saltelli
from qcommunity.utils.ising_obj import ising_objective
import qcommunity.modularity.graphs as gm
from qcommunity.optimization.obj import get_obj_val, get_obj
from qcommunity.optimization.run_with_angles import run_angles, test_angles, run_and_get_best
import qcommunity.optimization.qaoa_libensemble as qaoa_libensemble
import qcommunity.optimization.learning as ml
import qcommunity.optimization.cobyla_nlopt as cobyla_nlopt
import qcommunity.optimization.bobyqa_nlopt as bobyqa_nlopt
import qcommunity.optimization.newuoa_nlopt as newuoa_nlopt
import qcommunity.optimization.praxis_nlopt as praxis_nlopt
import qcommunity.optimization.sbplx_nlopt as sbplx_nlopt
import qcommunity.optimization.neldermead_nlopt as neldermead_nlopt
import qcommunity.optimization.mlsl_nlopt as mlsl_nlopt

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-l",
        type=int,
        default=3,
        help="number of vtx in the left (first) community")
    parser.add_argument(
        "-r",
        type=int,
        default=3,
        help="number of vtx in the right (second) community")
    parser.add_argument("--seed", type=int, default=None, help="random seed")
    parser.add_argument(
        "--graph-generator-seed",
        type=int,
        default=None,
        help="random seed, only used for graph generator")
    parser.add_argument(
        "--sobol", help="use sobol sequence as the random sequence", action="store_true")
    parser.add_argument(
        "--niter", type=int, default=100, help="number of iterations")
    parser.add_argument(
        "--label",
        type=str,
        help=
        "description of this version of the script. The description is prepended to the filename, so it should not contain any spaces. Default: time stamp"
    )
    parser.add_argument(
        "-g",
        "--graph-generator",
        type=str,
        default="get_barbell_graph",
        help="graph generator function")
    parser.add_argument(
        "--method",
        type=str,
        default="neldermead",
        choices=[
            "libensemble", "COBYLA_NLOPT",
            "BOBYQA_NLOPT", "MLSL_NLOPT", "NELDERMEAD_NLOPT", "NEWUOA_NLOPT", 
            "PRAXIS_NLOPT", "SBPLX_NLOPT"
        ],
        help="optimization method")
    parser.add_argument(
        "--localopt-method",
        type=str,
        default="LN_BOBYQA",
        choices=["LN_BOBYQA", "LN_COBYLA"],
        help="libensemble local optimization method")
    parser.add_argument(
        "--backend",
        type=str,
        default="IBMQX",
        choices=["IBMQX"],
        help="backend simulator to be used")
    parser.add_argument(
        "--problem",
        type=str,
        default="modularity",
        choices=["modularity", "maxcut"],
        help="the problem to be solved on the graph")
    parser.add_argument(
        "--ansatz-depth", type=int, default=1, help="variational ansatz depth")
    parser.add_argument(
        "--ansatz",
        type=str,
        default="RYRZ",
        choices=["RYRZ", "QAOA"],
        help="ansatz (variational form) to be used")
    parser.add_argument(
        "--sample-points",
        type=str,
        help="path to the pickle with sample points (produced by get_optimal_sample_points")
    parser.add_argument(
        "--max-active-runs",
        type=int,
        default=10,
        help="maximal number of active runs in libensemble")
    parser.add_argument(
        "--noise", help="flag to add noise", action="store_true")
    parser.add_argument(
        "--weighted", help="if raised, the graph will be randomly weighted", action="store_true")
    parser.add_argument(
        "--save", help="flag to save results", action="store_true")
    parser.add_argument(
        "--mpi", help="note that optimize is run with mpi", action="store_true")
    parser.add_argument(
        "--verbose", help="sets logging level to INFO", action="store_true")
    parser.add_argument(
        "--restart-local", help="restarts local nlopt methods after convergence with a new initial point", 
        action="store_true")
    parser.add_argument(
        "--zero-tol", help="pass zero tolerance to local optimizers (only use for no restart comparison!)", 
        action="store_true")
    parser.add_argument(
        "--remove-edge", 
        help="remove random edge from the graph", 
        action="store",
        const=-1, # hack! '-1' means remove random edge
        default=None,
        nargs='?',
        type=int)
    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.INFO)
        
    init_points = int(args.niter * 0.1)  # for backwards compatibility
    n_iter = args.niter - init_points
    if args.backend == "IBMQX":
        backend_params = {'depth': args.ansatz_depth, 'var_form':args.ansatz, 'backend_device':None}
    else:
        raise ValueError("Illegal backend: {}".format(args.backend))

    if args.graph_generator_seed is None:
        graph_generator_seed = args.seed
    else:
        graph_generator_seed = args.graph_generator_seed

    print(args.seed, graph_generator_seed)

    np.random.seed(args.seed)
    random.seed(args.seed)

    main_proc = True
    if args.mpi:
        from mpi4py import MPI
        if MPI.COMM_WORLD.Get_rank() != 0:
            main_proc = False

    if args.label:
        label = args.label
    else:
        import time
        label = time.strftime("%Y%m%d-%H%M%S")

    if not args.save:
        warnings.warn("save flag not raised, the results will be thrown out!",
                      UserWarning)
    params = {
        'init_points': init_points,
        'n_iter': n_iter,
        'localopt_method': args.localopt_method,
        'max_active_runs': args.max_active_runs,
        'ansatz': args.ansatz,
        'ansatz_depth': args.ansatz_depth,
        'seed': args.seed
    }
    
    if args.zero_tol:
        params['xtol_rel'] = 0 
        params['ftol_rel'] = 0 
    else:
        params['xtol_rel'] = 1e-3 
        params['ftol_rel'] = 1e-2


    H = {
        'method': args.method,
        'problem': {
            'id':
                '{}_left_{}_right_{}_seed_{}'.format(args.graph_generator,
                                                     args.l, args.r, args.seed),
            'graph_generator':
                args.graph_generator,
            'left':
                args.l,
            'right':
                args.r,
            'seed':
                args.seed,
            'graph_generator_seed':
                graph_generator_seed
        }
    }

    obj_val, num_parameters, all_x, all_vals = get_obj_val(
    # obj_val, num_parameters, all_x, all_vals, all_samples_tuple, all_samples_energies = get_obj_val(
        args.graph_generator,
        args.l,
        args.r,
        seed=graph_generator_seed,
        obj_params='ndarray',
        sign=-1,
        backend=args.backend,
        backend_params=backend_params,
        problem_name=args.problem,
        return_x=1,
        weight=args.weighted,
        remove_edge=args.remove_edge)  # sign = -1 because all methods minimize

    outname = os.path.join(
        "/zfs/safrolab/users/rshaydu/quantum/data/for_jeff/", args.method,
        "{}_{}_l_{}_r_{}_nparam_{}_noise_{}_init_pts_{}_niter_{}_seed_{}_graph_gen_seed_{}_max_active_runs_{}_sobol_{}_lopt_method_{}_removeedge_{}.p"
        .format(label, args.graph_generator, args.l, args.r, num_parameters,
                args.noise, init_points, n_iter, args.seed,
                graph_generator_seed, args.max_active_runs, args.sobol, args.localopt_method, args.remove_edge))
    print(outname)
    if os.path.isfile(outname):
        print('Output file {} already exists! Our job here is done'.format(
            outname))
        sys.exit(0)

    if args.ansatz == 'RYRZ':
        if args.sobol:
            raise ValueError("Sobol is not implemented yet for RYRZ ansatz")
        elif args.sample_points:
            raise ValueError("Sample points are not tested yet for RYRZ ansatz")
        else:
            sample_points = np.split(
                np.random.uniform(-np.pi + 0.25, np.pi - 0.25,
                                  args.niter * num_parameters), args.niter)
    elif args.ansatz == 'QAOA':
        if args.sobol:
            problem = {
              'num_vars': 2 * args.ansatz_depth,
              'bounds': [[0.25, np.pi - 0.25], [0.25, 2*np.pi - 0.25]] * args.ansatz_depth
            }
            sample_points = saltelli.sample(problem, args.niter*10)
            sample_points = sample_points[args.seed % 10 : ]
        elif args.sample_points:
            best_pts, best_vals = pickle.load(open(args.sample_points, 'rb'))
            sample_points = best_pts
            ub = [np.pi - 0.25, 2*np.pi - 0.25] * args.ansatz_depth * args.niter
            lb = [0.25]  * 2 * args.ansatz_depth * args.niter
            sample_points.extend(np.split(np.random.uniform(lb,ub, 2 * args.ansatz_depth * args.niter), args.niter))
        else:
            ub = [np.pi - 0.25, 2*np.pi - 0.25] * args.ansatz_depth * args.niter
            lb = [0.25]  * 2 * args.ansatz_depth * args.niter
            sample_points = np.split(np.random.uniform(lb,ub, 2 * args.ansatz_depth * args.niter), args.niter)
    else:
        raise ValueError("Unsupported ansatz: {}".format(args.ansatz))

    params['sample_points'] = sample_points
    if args.method == 'libensemble':
        res_tuple = qaoa_libensemble.optimize_obj(obj_val, num_parameters,
                                                  params)
        # libensemble is kinda problematic
        if main_proc:
            # cannot simply put two things to the left of equality since None is not iterable
            res = res_tuple[0]
            persis_info = res_tuple[1]
            all_x = res['x']
            H['persis_info'] = persis_info
            all_vals = [{'mean': -x, 'max': None} for x in res['f']]
    elif args.method == 'MLSL_NLOPT':
        res = mlsl_nlopt.optimize_obj(obj_val, num_parameters, params)
    else:
        # assuming the method is local
        # for local methods, once they converge, restart until exhausting the eval limit
        while len(all_vals) < args.niter:
            print("Starting {}, current nevals {}".format(args.method, len(all_vals)))
            try:
                if args.method == 'COBYLA_NLOPT':
                    res = cobyla_nlopt.optimize_obj(obj_val, num_parameters, params)
                elif args.method == 'BOBYQA_NLOPT':
                    res = bobyqa_nlopt.optimize_obj(obj_val, num_parameters, params)
                elif args.method == 'NELDERMEAD_NLOPT':
                    res = neldermead_nlopt.optimize_obj(obj_val, num_parameters, params)
                elif args.method == 'NEWUOA_NLOPT':
                    res = newuoa_nlopt.optimize_obj(obj_val, num_parameters, params)
                elif args.method == 'PRAXIS_NLOPT':
                    res = praxis_nlopt.optimize_obj(obj_val, num_parameters, params)
                elif args.method == 'SBPLX_NLOPT':
                    res = sbplx_nlopt.optimize_obj(obj_val, num_parameters, params)
                else:
                    raise ValueError('Incorrect method: {}'.format(args.method))
            except (nlopt.RoundoffLimited, FloatingPointError) as e:
                print("Encountered {}, recovering results at iter {}".format(e, len(all_vals)))
                res = None
            assert len(all_x) == len(all_vals)
            params['sample_points'] = params['sample_points'][1:]
            if not args.restart_local:
                break

    # If in running the try-except run over the limit, truncate the result to just first niter
    if len(all_x) > args.niter or len(all_vals) > args.niter:
        all_x = all_x[:args.niter]
        all_vals = all_vals[:args.niter]

    if main_proc:
        print("Total nevals", len(all_x))
        H['x'] = all_x
        H['values'] = all_vals
        H['num_parameters'] = num_parameters
        H['raw_output'] = res
        print("energy from optimizer: {}".format(
            max([x['mean'] for x in H['values']])))

        if args.save:
            print("\n\n\nRun completed.\nSaving results to file: " + outname)
            params.update({
                'l': args.l,
                'r': args.r,
                'graph_generator': args.graph_generator,
                'seed': args.seed,
                'ansatz_depth': backend_params['depth'],
                'args':args
            })
            pickle.dump((H, params), open(outname, "wb"))
    if args.mpi:
        MPI.COMM_WORLD.Barrier()
