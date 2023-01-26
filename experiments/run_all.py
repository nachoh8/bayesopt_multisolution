from functools import partial
import itertools

import os
import errno

import gp_sample_optimization
import robot.gpyopt_push3
#import keras_example
#import vae_example
from functions import run_function
from functions import Michalewicz, Bohachevsky

def bind_problem(problem_f, args):
    return partial(problem_f, args.split(' '))

if __name__ == "__main__":
    # Global arguments
    num_trials = 10
    output_folder = '/home/javi/code/notebooks/output_02_25_sequential_comparison/'
    parallel = 1 #parallel = 10
    approx_iterations = 150

    num_trials_robotpush = 40

    # Create output folder
    try:
        os.makedirs(output_folder)
    except OSError as exc:
        if exc.errno == errno.EEXIST and os.path.isdir(output_folder):
            pass
        else:
            raise

    # Define all strategies
    all_strategies = {
        'n_parallel_samples': parallel,
        'n_iterations': approx_iterations/parallel,
        #'kernel_name': 'kSpartan',
    }

    mcmc_ei_params = {
        'par_type': 'PAR_MCMC',
        'crit_name': 'cEI'
    }

    mcmc_lcb_params = {
        'par_type': 'PAR_MCMC',
        'crit_name': 'cLCBnorm'
    }

    mcmc_poi_params = {
        'par_type': 'PAR_MCMC',
        'crit_name': 'cPOI'
    }

    thompson_params = {
        'par_type': 'PAR_THOMPSON',
        'crit_name': 'cThompsonSampling'
    }

    ei_temperature = 0.4
    mcmc_ei_exponential_params = {
        'par_type': 'PAR_MCMC',
        'crit_name': 'cEIexp',
        'crit_params': [1.0, ei_temperature]
    }

    lcb_temperature = 1.0
    mcmc_lcb_exponential_params = {
        'par_type': 'PAR_MCMC',
        'crit_name': 'cLCBexp',
        'crit_params': [1.0, lcb_temperature]
    }

    seq_ei_params = {
        'par_type': 'PAR_NONE',
        'crit_name': 'cEI',
    }


    # Define all problems

    # vaeexample = bind_problem(vae_example.run_experiment,
    #                         '--num-trials {} --experiment-output {}'.format(num_trials, output_folder))

    # kerasexample = bind_problem(keras_example.run_experiment,
    #                         '--num-trials {} --experiment-output {}'.format(num_trials, output_folder))


    def gpsample(dim, use_rq_kernel=False):
        filename = 'samples_{}{}.pickle'.format(dim, '_rq' if use_rq_kernel else '')
        args = '--num-trials {num_trials} --dim {dim} --input-samples {filename} --output-samples {filename} --experiment-output {exp_output} --num-gp-points {gp_points}'.format(
            num_trials=num_trials, dim=dim, filename=filename, exp_output=output_folder, gp_points=10
        )
        if use_rq_kernel:
            args += ' --use-rq-kernel'
        return bind_problem(gp_sample_optimization.run_experiment, args)

    def michalewicz(dim):
        def instantiateMicha(params):
            func = Michalewicz(dim)
            run_function(params, func, output_folder, num_trials)
        return instantiateMicha

    def bohachevsky():
        def instantiateBoha(params):
            func = Bohachevsky()
            run_function(params, func, output_folder, num_trials)
        return instantiateBoha
        

    def robotpush(dim):
        assert dim == 3 or dim == 4
        return bind_problem(robot.gpyopt_push3.run_experiment,
                            '--num-trials {} --dim {} --experiment-output {}'.format(num_trials_robotpush, dim, output_folder))

    
    strategies = [mcmc_lcb_params, mcmc_ei_params, mcmc_poi_params, thompson_params]
    problems = [gpsample(2), gpsample(2, True)]

    strategies = [seq_ei_params]
    problems = [bohachevsky()]

    # Testing burnouts
    '''burnouts = [10, 50, 100, 500, 1000, 1500, 2500, 3500]
    strategies = []
    for b in burnouts:
        strategies.append({
            'par_type': 'PAR_MCMC',
            'crit_name': 'cEI',
            'burnout_mcmc_parallel': b
        })
    problems = [michalewicz(8)]'''

    # Run all experiments
    for strat in strategies:
        strat.update(all_strategies)

    for problem, params in itertools.product(problems, strategies):
        problem(params)
