import numpy as np
import argparse
import sys
import traceback
import bayesopt

from kvae.vae import VAE

def parse_args(raw_args):
    parser = argparse.ArgumentParser(
        description='Robust parsing script with stated inputs.',
    )

    parser.add_argument(
        '--num-trials',
        type=int,
        required=False,
        default=20,
        help='Number of trials for this given function.',
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

def run_experiment(exp_args, bopt_params={}):
    args = parse_args(exp_args)
    num_trials = args.num_trials
    exp_folder = args.experiment_output

    n = 4
    lb = np.array((1e-6, 1e-10, 0.0, 10))
    ub = np.array((1e-2, 0.1, 0.1, 300))


    #from pympler.tracker import SummaryTracker
    #tracker = SummaryTracker()

    vae_obj = VAE()

    def opt_func(X):
        try:
            arr = np.zeros((1, 4))
            arr[0][0] = X[0]
            arr[0][1] = X[1]
            arr[0][2] = X[2]
            arr[0][3] = int(X[3])
            return vae_obj.call(arr)
        except Exception as e:
            print 'ERROR:', str(e)
            print 'INPUT:', X
            print(traceback.format_exc())

    for r in range(num_trials):
        params = {}
        params['n_iterations'] = 10
        params['n_iter_relearn'] = 5
        params['n_init_samples'] = 10
        params['n_parallel_samples'] = 4
        params['par_type'] = 'PAR_MCMC'
        params['load_save_flag'] = 2
        params['crit_name'] = 'cEI'

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
            strat=strat, problem='vaeexample'
        )

        mvalue, x_out, error = bayesopt.optimize(opt_func, n, lb, ub, params)

    #tracker.print_diff()

if __name__ == "__main__":
    run_experiment(sys.argv[1:])
