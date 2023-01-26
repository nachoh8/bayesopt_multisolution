import numpy as np
import bayesopt

import george
import logging
import sys
import argparse
import pickle
from scipy.optimize import minimize


"""
--------------------------------------------------------
Instructions on how to use this script

You need to install this from gpyopt_mirror repo.

Minimal dependencies (all from pip): GPy, numpy, scipy
then run the standard "python setup.py install" from this library.

In the main function, we are interested in these parameters

  dim: dimensionality of the problem
  max_iter: number of iterations
  kernel: this is the kernel used for Bayesian optimization

  These parameters affect the way we sample the target function:
    sampler: function sampled from a GP. Currently, to change the kernel, we have to edit GPSampler
    lenght: is the lenghtscale of the GP used generate "sampler"

  When we generate func2 (the function with outliers):
    outlier_ratio: ratio of outliers (0, 1)
    max_value, min_value: bounds of the uniform distribution for the outliers

  When we run the robustGP optimization:
    low_margin, top_margin: percentile to remove points from the top or the bottom of the distribution
    filtering_startup, filtering_interval: iterations at which the removal of outliers
        is produced. So far, 40-2 is the best strategy.

  The traces of each experiment (baseline, GP with outliers and our method are saved in the last line).
  The script plot_stuff in this folder can be used to plot this.
--------------------------------------------------------
"""

# This might not be needed
#logging.basicConfig(filename="sample.log", level=logging.INFO)
logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger("results")
logger.setLevel(logging.INFO)

def parse_args(raw_args):
    parser = argparse.ArgumentParser(
        description='Robust parsing script with stated inputs.',
    )

    parser.add_argument(
        '--dim',
        type=int,
        required=False,
        default=8,
        help='The dimension of the function to be optimized.',
    )

    parser.add_argument(
        '--max-iterations',
        type=int,
        required=False,
        default=100,
        help='Maximum number of iterations.',
    )

    parser.add_argument(
        '--num-trials',
        type=int,
        required=False,
        default=20,
        help='Number of trials for this given function.',
    )

    parser.add_argument(
        '--num-gp-points',
        type=int,
        required=False,
        default=10,
        help='Number of points at which the GP is defined.',
    )

    parser.add_argument(
        '--use-rq-kernel',
        action='store_true',
        required=False,
        help='Use RQ kernel for the benchmark or keep the Matern default'
    )

    parser.add_argument(
        '--input-samples',
        type=str,
        required=False,
        default=None,
        help='Pickle file with the samples to load'
    )

    parser.add_argument(
        '--output-samples',
        type=str,
        required=False,
        default=None,
        help='Pickle filename where samples will be saved'
    )

    parser.add_argument(
        '--experiment-output',
        type=str,
        required=False,
        default='./output/',
        help='folder in which results will be saved',
    )

    args = parser.parse_args(raw_args)
    return args


class GPSampler(object):
    def __init__(self, dim, lenght=0.01, num_points=10, use_rq_kernel=False, data=None):
        # For proper testing we should have different kernels for sampling and optimization
        # In-class funcion vs Out-of-class function

        if use_rq_kernel:
            self.kernel = george.kernels.RationalQuadraticKernel(0.3, metric=lenght, ndim=dim)
        else:
            self.kernel = george.kernels.Matern52Kernel(lenght, ndim=dim)

        self.gp = george.GP(self.kernel)

        if data is None or "X" not in data or "Y" not in data:
            self.X = np.random.rand(num_points, dim)

            self.gp.compute(self.X)
            self.Y = self.gp.sample(self.X)
        else:
            self.X = data["X"]
            self.gp.compute(self.X)
            self.Y = data["Y"]

    def call(self, X_test):
        try:
            mu, _ = self.gp.predict(self.Y, X_test.reshape(1, -1))
            return mu
        except Exception as e:
            import traceback
            traceback.print_exc()
            raise e


def create_sampler(dim, length, num_gp_points, use_rq_kernel=False, load_f=None, save_f=None):

    def load_pickle(filename):
        if filename is not None:
            try:
                with open(filename, 'rb') as handle:
                    return pickle.load(handle)
            except:
                pass
        return None

    def save_pickle(filename, data):
        if filename is not None:
            with open(filename, 'wb') as handle:
                pickle.dump(data, handle)

    data = load_pickle(load_f)
    sampler = GPSampler(dim, length, num_gp_points, use_rq_kernel=use_rq_kernel, data=data)
    data = {
        "X": sampler.X,
        "Y": sampler.Y,
    }
    save_pickle(save_f, data)
    return sampler


def run_experiment(exp_args, bopt_params={}):
    args = parse_args(exp_args)
    dim = args.dim
    exp_folder = args.experiment_output
    max_iter = args.max_iterations
    num_gp_points = args.num_gp_points
    num_trials = args.num_trials
    use_rq_kernel = args.use_rq_kernel

    # Problem
    lb = np.array([0.] * dim)
    ub = np.array([1.] * dim)
    length = 0.1

    #sampler = create_sampler(dim, length, num_gp_points, use_rq_kernel=use_rq_kernel,
    #                         load_f=args.input_samples, save_f=args.output_samples)

    for r in range(num_trials):
        # Optimizer
        np.random.seed(r)
        sampler = GPSampler(dim, length, num_gp_points, use_rq_kernel=use_rq_kernel, data=None)

        params = {}
        params['n_iterations'] = max_iter
        params['n_iter_relearn'] = 5
        params['n_init_samples'] = 10
        params['n_parallel_samples'] = 4
        params['crit_name'] = 'cEI'
        params['par_type'] = 'PAR_MCMC'
        params['load_save_flag'] = 2


        # Overwrite defaults
        params.update(bopt_params)
        params['random_seed'] = r
        
        extra = ''
        if use_rq_kernel:
            extra += 'rq' 

        strat = 'batch_sampling'
        if params['crit_name'] == 'cLCBexp' or params['crit_name'] == 'cEIexp':
            temp = 'default'
            if len(params['crit_params']) >= 2:
                temp = str(params['crit_params'][1])
            strat = 'fixedtemp_' + temp

        params['save_filename'] = exp_folder + '{strat}_{crit}_x{par}_{problem}_rep_{rep}.dat'.format(
            crit=params['crit_name'], par=params['n_parallel_samples'], rep=r,
            strat=strat, problem='gpsample' + str(dim) + extra
        )

        mvalue, x_out, error = bayesopt.optimize(sampler.call, dim, lb, ub, params)


if __name__ == "__main__":
    run_experiment(sys.argv[1:])

