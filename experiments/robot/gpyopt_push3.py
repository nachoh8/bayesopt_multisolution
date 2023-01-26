import numpy as np
import bayesopt


import traceback
import logging
import time
import argparse
from push_world import *
import sys



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

# This is to avoid annoyances from GPy optimizers
import ipdb
ipdb.set_trace = lambda: None

def parse_args(raw_args):
    parser = argparse.ArgumentParser(
        description='Robust parsing script with stated inputs.',
    )

    parser.add_argument(
        '--dim',
        type=int,
        required=False,
        default=3,
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
        '--outlier-ratio',
        type=float,
        required=False,
        default=0.2,
        help='What proportion of observations (randomly) should be outliers.',
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


class RobotSim(object):
    def __init__(self, dim):
        self.dim = dim
        self.random_object()

    def initialize(self):
        self.random_object()

    def random_object(self):
        self.gx = 10 * np.random.rand() - 5
        self.gy = 10 * np.random.rand() - 5
        self.gx_old = self.gx
        self.gy_old = self.gy

    def outlier_object(self, gx, gy):
        self.gx = gx
        self.gy = gy

    def recover_object(self):
        self.gx = self.gx_old
        self.gy = self.gy_old

    def call(self, X_test):
        rx = X_test[0]
        ry = X_test[1]
        gx = self.gx
        gy = self.gy
        simu_steps = int(X_test[2] * 10)
        if self.dim > 3:
            init_angle = X_test[3]
        else:
            init_angle = np.arctan(ry / rx)

        # set it to False if no gui needed
        world = b2WorldInterface(False)
        oshape, osize, ofriction, odensity, bfriction, hand_shape, hand_size  = 'circle', 1, 0.01, 0.05, 0.01, 'rectangle', (0.3,1) 
        thing,base = make_thing(500, 500, world, oshape, osize, ofriction, odensity, bfriction, (0,0))

        robot = end_effector(world, (rx,ry), base, init_angle, hand_shape, hand_size)
        if self.dim > 3:
            xvel = -rx
            yvel = -ry
            regu = np.linalg.norm([xvel,yvel])
            xvel = xvel / regu * 10
            yvel = yvel / regu * 10
            ret = simu_push2(world, thing, robot, base, xvel, yvel, simu_steps)
        else:
            ret = simu_push(world, thing, robot, base, simu_steps)

        ret = np.linalg.norm(np.array([gx, gy]) - ret)
        #print X_test, ret

        return ret


def run_optimization(func, bounds, model, kernel, max_iter=100, repetitions=10,
                     low_margin=5, top_margin=5, filtering_startup=5, filtering_interval=5):
    values = None
    times = []
    watchdog = 0

    for iteration in xrange(repetitions):
        np.random.seed(iteration)
        func.initialize()
        all_ok = False
        while not all_ok and watchdog < 3:
            try:
                # Never use normalize_Y. It breaks the robust part.
                myProblem = GPyOpt.methods.BayesianOptimization(func.call, bounds,
                                                                model_type=model,
                                                                normalize_Y=False,
                                                                kernel=kernel,
                                                                evaluations_file='results.out')
                myProblem.set_rejection_margins(low_margin, top_margin)  # Percentages
                myProblem.set_rejection_schedule(filtering_startup, filtering_interval)
                print "Running main loop"
                start = time.time()
                myProblem.run_optimization(max_iter)
                end = time.time()
                logger.info("Keeped values:")
                logger.info(myProblem.Y)

                print "Optimum:", myProblem.fx_opt
                if values is None:
                    values = myProblem.Y_best
                else:
                    values = np.vstack([values, myProblem.Y_best])

                times.append(end - start)

                #myProblem.plot_convergence()
                all_ok = True
            except Exception as e:
                print "Problem:", e
                watchdog += 1
        print "Iter:", iteration

    return values, times

def run_experiment(exp_args, bopt_params={}):
    args = parse_args(exp_args)
    dim = args.dim
    exp_folder = args.experiment_output
    if dim not in (3,4):
        raise RuntimeError("Incorrect dimension")
    
    max_iter = args.max_iterations
    outlier_ratio = args.outlier_ratio
    num_trials = args.num_trials


    lb = np.array([-5., -5. ,1.])
    ub = np.array([5., 5. ,30.])

    if dim > 3:
        lb = np.append(lb, 0)
        ub = np.append(ub, 2 * np.pi)



    sampler = RobotSim(dim)

    def func(*args, **kwargs):
        arr = []
        for i in range(50):
            try:
                v = sampler.call(*args, **kwargs)
                arr.append(v)
            except Exception as e:
                traceback.print_exc()
        return np.mean(arr)

    for r in range(num_trials):
        np.random.seed(r)
        sampler.initialize()

        # Optimizer
        params = {}
        params['n_iterations'] = max_iter
        params['n_iter_relearn'] = 5
        params['n_init_samples'] = 10
        params['n_parallel_samples'] = 4
        params['crit_name'] = 'cEI'
        params['par_type'] = 'PAR_MCMC'
        #params['kernel_name'] = 'kMaternISO5'
        #params['kernel_hp_std'] = [1.0] * dim
        params['load_save_flag'] = 2

        # Overwrite defaults
        params.update(bopt_params)
        params['random_seed'] = r

        strat = 'batch_sampling'
        if params['crit_name'] == 'cLCBexp' or params['crit_name'] == 'cEIexp':
            temp = 'default'
            if len(params['crit_params']) >= 2:
                temp = str(params['crit_params'][1])
            strat = 'fixedtemp_' + temp

        params['save_filename'] = exp_folder + '{strat}_{crit}_x{par}_{problem}_rep_{rep}.dat'.format(
            crit=params['crit_name'], par=params['n_parallel_samples'], rep=r,
            strat=strat, problem='robotpush'+str(dim)
        )

        mvalue, x_out, error = bayesopt.optimize(func, dim, lb, ub, params)

if __name__ == "__main__":
    run_experiment(sys.argv[1:])

    '''
    sampler = RobotSim(3)
    x = [0.1, 0.1, 20]

    lb = np.array([-5., -5. ,1.])
    ub = np.array([5., 5. ,30.])

    for s in range(20):
        np.random.seed(s)
        x[0] = np.random.uniform(lb[0], ub[0])
        x[1] = np.random.uniform(lb[1], ub[1])
        x[2] = np.random.uniform(lb[2], ub[2])

        np.random.seed(0)
        sampler.initialize()
        a = []
        for i in range(200):
            a.append(sampler.call(x))
        print(x, 'a =', np.mean(a), '+-', np.var(a))
    '''
