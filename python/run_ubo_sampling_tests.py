import bayesopt
import numpy as np
import sys
import itertools
import argparse

from ubo_sampling_functions.push_function_one import PushReward
from ubo_sampling_functions.functions import Michalewicz, GaussianMixture, Polynomial
from ubo_sampling_functions.robot.gpyopt_push3 import RobotSim
from ubo_sampling_functions.rover_function import RoverProblem
from rkhs import rkhs_synth


# Functions to read a dat file
def open_file(filename):
    with open(filename) as f:
        content = f.readlines()
    return [c.rstrip() for c in content]


def parse_value(value):
    if value.startswith('['):
        fields = value.split('](')
        size_str = fields[0][1:].split(',')
        arr_str = fields[1][:-1].split(',')
        arr_str = [a for a in arr_str if a != '']

        if int(size_str[0]) == 0:
            return np.array([])

        size = tuple([int(n) for n in size_str])
        arr = np.array([float(n) for n in arr_str])
        return arr.reshape(size)

    elif value.isdigit():
        return float(value)

    else:
        return value


def read_dat(filename):
    lines = open_file(filename)
    dat = {}
    for l in lines:
        fields = l.split('=')
        dat[fields[0]] = parse_value(fields[1])

    return dat


# Function to name the output file
def name_output(params, problem, rep, output_dir):
    base_template = output_dir + '{strat}_{crit}_x{par}_{problem}_rep_{rep}.dat'
    strat = 'sequential'
    par = str(params['n_parallel_samples'])
    crit = params['crit_name']

    if 'input_noise_variance' not in params:
        strat = 'default'

    if params['par_type'] == 'PAR_MCMC':
        strat = 'sampling'
    elif params['par_type'] == 'PAR_RANDOM':
        strat = 'random'

    return base_template.format(strat=strat, crit=crit, par=par, problem=problem, rep=rep)


class ExperimentRunner:
    def __init__(self, f, lb, ub, params, problem_name, reps_list, output_dir, input_var=None, num_samples=1000,
                 skip_opt=False, resume=False, safety_flag=False):
        self.f = f
        self.lb, self.ub = lb, ub
        self.params = params
        self.problem_name = problem_name
        self.reps_list = reps_list
        self.output_dir = output_dir
        self.num_samples = num_samples
        self.skip_opt = skip_opt
        self.resume = resume
        self.safety_flag = safety_flag
        self.input_var = input_var
        self.dims = len(ub)

    def contaminate(self, x, sample=None, is_input_norm=False):
        if self.input_var is None:
            return x

        if sample is None:
            sample = np.random.normal([0] * self.dims, np.sqrt(self.input_var))

        lower = np.array(self.lb)
        upper = np.array(self.ub)

        if is_input_norm:
            x_norm = np.array(x)
        else:
            x_norm = (np.array(x) - lower) / (upper - lower)

        x_contaminate = x_norm + sample
        x_denorm = x_contaminate * (upper - lower) + lower
        return x_denorm

    def evaluate(self, x):
        try:
            if self.safety_flag:
                y = self.f(x)
            else:
                x_noise = self.contaminate(x)
                y = self.f(x_noise)
            return y

        except Exception as error:
            print(error)

    def run(self):
        for i in self.reps_list:
            np.random.seed(seed=i+1000)
            filename = name_output(self.params, problem=self.problem_name, rep=i, output_dir=self.output_dir)

            # Run optimization
            if not self.skip_opt:
                self.params['save_filename'] = filename
                self.params['random_seed'] = i

                mvalue, x_out, error = bayesopt.optimize(self.evaluate, self.dims, self.lb, self.ub, self.params)

                # Add line to the dat file
                with open(filename, 'a') as file:
                    line = 'mOut=[{}]{}\n'.format(self.dims, tuple(x_out)).replace(" ", "")
                    file.write(line)

            # Postprocess results for plotting
            if self.input_var is not None:
                data = read_dat(filename)

                #print('Computing means and std of {}'.format(filename.split('/')[-1]))
                avg_y, std_y = self.compute_deviations_list(data['mX'], is_input_norm=True)
                avg_best, std_best = self.compute_deviations_list(data['mDebugBestPoints'])
                avg_out, std_out = self.compute_deviations(data['mOut'])

                if self.skip_opt:  # Remove already existing avg and std computations
                    fields = ['mAvgOut', 'mStdOut', 'mAvgY', 'mStdY', 'mDebugBestPointsAvgValues',
                              'mDebugBestPointsStdValues']
                    with open(filename, 'r') as file:
                        all_lines = file.readlines()
                    with open(filename, 'w') as file:
                        for line in all_lines:
                            if not any(line.startswith(v + '=') for v in fields):
                                file.write(line)

                # Add new avg and std computations
                with open(filename, 'a') as file:
                    self.write_value(file, 'mAvgOut', avg_out)
                    self.write_value(file, 'mStdOut', std_out)

                    self.write_list(file, 'mAvgY', avg_y)
                    self.write_list(file, 'mStdY', std_y)

                    self.write_list(file, 'mDebugBestPointsAvgValues', avg_best)
                    self.write_list(file, 'mDebugBestPointsStdValues', std_best)

    def compute_deviations_list(self, x_list, is_input_norm=False):
        avg_list = []
        std_list = []
        for x in x_list:
            avg, std = self.compute_deviations(x, is_input_norm=is_input_norm)
            avg_list.append(avg)
            std_list.append(std)

        return avg_list, std_list

    def compute_deviations(self, x, is_input_norm=False):
        samples = np.random.normal(np.zeros(self.dims), np.sqrt(self.input_var), (self.num_samples, self.dims))
        samples_values = [self.f(self.contaminate(x, samples[j, :], is_input_norm=is_input_norm)) for j in
                          range(self.num_samples)]
        return np.mean(samples_values), np.std(samples_values)

    def write_value(self, file, name, value):
        line = '{}={}\n'.format(name, value).replace(" ", "")
        file.write(line)

    def write_list(self, file, name, values):
        line = '{}=[{}]{}\n'.format(name, len(values), tuple(values)).replace(" ", "")
        file.write(line)


class ConfigurationManager:
    # Prepares all the resources (parameters, function ...) to allow execution of a experiment configuration
    def __init__(self, resource_folder, output_dir, skip_opt, safety_flag):
        self.resource_folder = resource_folder
        self.output_dir = output_dir
        self.skip_opt = skip_opt
        self.safety_flag = safety_flag

        self.num_experiments = None
        self.started_experiments = 0

    def default_params(self, input_var, ndim, iters=100, configuration=None):
        if configuration is None:
            configuration = {'strat': 'default', 'par': 1, 'crit': 'cEI'}

        strat = configuration['strat']

        params = {}
        params['n_iter_relearn'] = 1
        params['l_type'] = 'mcmc'
        params['noise'] = 1e-4
        params['force_jump'] = 3
        params['crit_name'] = configuration['crit']
        # params['sigma_s'] = 4
        params['verbose_level'] = -1
        params['load_save_flag'] = 2
        params['n_parallel_samples'] = 1
        params['par_type'] = 'PAR_NONE'
        params['init_method'] = 2  # 1-LHS, 2-Sobol

        params['n_iterations'] = iters
        params['n_init_samples'] = 20
        params['n_inner_iterations'] = 1200  # 500 #per dim
        params['burnout_mcmc_parallel'] = 1000

        # Spartan
        params['kernel_name'] = "kNonStMoving(kMaternARD5,kMaternARD5)"
        params['kernel_hp_mean'] = [0.5] * ndim + [1] * ndim + [1] * ndim
        params['kernel_hp_std'] = [0.2] * ndim + [5] * ndim + [5] * ndim
        params['kernel_domain_center'] = np.asarray([[0.5] * ndim, [0.5] * ndim])
        params['kernel_domain_std'] = np.asarray([[0.05] * ndim, [10] * ndim])

        # Unscented BO
        if strat != 'default':
            params['input_noise_variance'] = input_var

        # Crit sampling
        if strat == 'sampling':
            params['n_parallel_samples'] = configuration['par']
            params['n_iterations'] = params['n_iterations'] / configuration['par']
            params['par_type'] = 'PAR_MCMC'

        if strat == 'random':
            params['n_parallel_samples'] = configuration['par']
            params['n_iterations'] = params['n_iterations'] / configuration['par']
            params['par_type'] = 'PAR_RANDOM'

        return params

    def get_function(self, problem_name):
        f = None
        lb = None
        ub = None

        if problem_name == 'rkhs':
            def f(x):
                return -rkhs_synth(x)

            n = 1
            lb = np.zeros((n,))
            ub = np.ones((n,))

        elif problem_name == 'robotpush1':
            f = PushReward(use_dir=False, use_gui=False)
            lb = f.lb
            ub = f.ub

        elif problem_name == 'micha4':
            micha = Michalewicz(dim=4)

            def micha4_func(x):
                return micha.call(x) * 5.0

            f = micha4_func
            lb, ub = micha.bounds()

        elif problem_name == 'polynomial':
            poly = Polynomial()

            f = poly.call
            lb, ub = poly.bounds()

        elif problem_name == 'gmixture':
            gm = GaussianMixture()

            def gmixture_func(x):
                return gm.call(x) * 100.0

            f = gmixture_func
            lb, ub = gm.bounds()

        elif problem_name == 'robotpush3':
            rs = RobotSim(3)
            rs.gx = 4.0
            rs.gy = 1.5

            def robotpush3_func(x):
                return rs.call(x) * 5.0

            f = robotpush3_func
            lb = rs.lb
            ub = rs.ub

        elif problem_name == 'robotpush4':
            rs = RobotSim(4)
            rs.gx = 4.0
            rs.gy = 1.5

            def robotpush4_func(x):
                return rs.call(x) * 5.0

            f = robotpush4_func
            lb = rs.lb
            ub = rs.ub
        elif problem_name.startswith('rover4'):
            if problem_name == 'rover4':
                rover = RoverProblem(2, k=2, s=0.0, extra_domain=0.0)
            elif problem_name == 'rover4hill':
                rover = RoverProblem(2, k=2, s=0.0, extra_domain=0.0,
                                     img_map=self.resource_folder)
            else:
                rover = None

            f = rover
            lb = rover.lb
            ub = rover.ub

        if f is None:
            print('Error: function {} not found'.format(problem_name))

        return f, lb, ub

    def get_experiment(self, problem_name, conf):
        f, lb, ub = self.get_function(problem_name)
        params = None
        input_var = None

        # Function RKHS (1d)
        if problem_name == 'rkhs':
            input_var = 0.015 ** 2  # 0.02
            params = self.default_params(input_var, ndim=len(ub), iters=40, configuration=conf)
            params['n_init_samples'] = 5

        # Function Polynomial (2d)
        elif problem_name == 'polynomial':
            input_var = 0.1 ** 2  # 0.02
            params = self.default_params(input_var, ndim=len(ub), iters=60, configuration=conf)
            params['n_init_samples'] = 20  # Previous 40

        # Function GaussianMixture (2d)
        elif problem_name == 'gmixture':
            input_var = 0.065 ** 2
            params = self.default_params(input_var, ndim=len(ub), iters=40, configuration=conf)
            params['n_init_samples'] = 5  # Previous 20,40

        # Function Michalewicz (4d)
        elif problem_name == 'micha4':
            input_var = 0.05 ** 2  # 0.02
            params = self.default_params(input_var, ndim=len(ub), iters=40, configuration=conf)
            params['n_init_samples'] = 30

        # Function robotpush3 (3d):
        elif problem_name == 'robotpush3':
            input_var = 0.07 ** 2  # 0.02 ** 2
            params = self.default_params(input_var, ndim=len(ub), iters=40, configuration=conf)
            params['n_init_samples'] = 10

        # Function robotpush4 (4d):
        elif problem_name == 'robotpush4':
            input_var = 0.04 ** 2  # 0.01
            params = self.default_params(input_var, ndim=len(ub), iters=40, configuration=conf)
            params['n_init_samples'] = 10

        # Function rover4 (4d): default and hill_map
        elif problem_name.startswith('rover4'):
            input_var = 0.05 ** 2  # 0.02 ** 2
            #input_var = 0.015 ** 2

            params = self.default_params(input_var, ndim=len(ub), iters=40, configuration=conf)
            params['n_init_samples'] = 30

        return f, lb, ub, params, input_var

    def run_configuration(self, conf, problem_name, rep):
        single_rep = [rep]

        f, lb, ub, params, input_var = self.get_experiment(problem_name, conf)
        experiment = ExperimentRunner(f, lb, ub, params, problem_name, reps_list=single_rep, output_dir=self.output_dir,
                                      input_var=input_var, skip_opt=self.skip_opt, safety_flag=self.safety_flag)
        experiment.run()


def main():
    parser = argparse.ArgumentParser(description='Run input noise experiments.')
    parser.add_argument('output', type=str, help='set output directory')
    parser.add_argument('--reps', type=int, default=20, help='set number of repetitions')
    parser.add_argument('--start-reps', type=int, default=0, help='seed at which a experiment will start')
    parser.add_argument('--skip', action='store_true', help='skip optimization and only postprocess')
    parser.add_argument('--ray', action='store_true', help='activate ray parallelization of optimizations')
    parser.add_argument('--funcs', nargs='+', default=[])
    parser.add_argument('--priority', choices=['reps', 'funcs', 'configs'])
    parser.add_argument('--safety', action='store_true', help='removes noise from optimization')
    args = parser.parse_args()

    output_dir = args.output
    num_rep = args.reps
    start_rep = args.start_reps
    use_ray = args.ray
    skip_optimization = args.skip
    functions_list = args.funcs
    priority = args.priority
    safety_flag = args.safety

    if use_ray:
        import ray
        ray.init()

    # Put '/' at the end of directory
    if not output_dir.endswith('/'):
        output_dir += '/'

    if len(functions_list) == 0:
        functions_list = ['rkhs', 'gmixture', 'robotpush3', 'robotpush4', 'rover4hill', 'micha4', 'polynomial']

    # Configurations to run
    crit_name = 'cEI'
    config_list = [
        {'strat': 'default', 'par': 1, 'crit': crit_name},
        {'strat': 'sequential', 'par': 1, 'crit': crit_name},
        {'strat': 'sampling', 'par': 1, 'crit': crit_name},
    ]
    repetitions_list = range(start_rep, start_rep + num_rep)

    conf_manager = ConfigurationManager(resource_folder='./ubo_sampling_functions/hill_map.png',
                                        output_dir=output_dir,
                                        skip_opt=skip_optimization,
                                        safety_flag=safety_flag,)

    # Prioritize which experiments to run first
    if priority == 'funcs':
        prod_order = itertools.product(functions_list, repetitions_list, config_list)
        experiment_list = [(rep, func, conf) for func, rep, conf in prod_order]
    elif priority == 'confs':
        prod_order = itertools.product(config_list, functions_list, repetitions_list)
        experiment_list = [(rep, func, conf) for conf, rep, func in prod_order]
    else:
        prod_order = itertools.product(repetitions_list, functions_list, config_list)
        experiment_list = [(rep, func, conf) for rep, func, conf in prod_order]

    # Keeps track of number of started experiments and plots the current progress
    class ProgressManager(object):
        def __init__(self, total):
            self.counter = 0
            self.total = total

        def report_new_task(self, conf, func, rep):
            self.counter += 1
            self.print_progress(conf, func, rep, self.counter, self.total)

        def print_progress(self, configuration, problem_name, repetition, current, total):
            bar_format = '[{}/{}]'.format('{' + ':>' + str(len(str(total))) + '}', '{}')
            bar_string = bar_format.format(current, total)
            print(bar_string + ' {} {} {}'.format(configuration, problem_name, repetition))

    # Launch experiment tasks
    num_total_experiments = len(experiment_list)
    if use_ray:
        @ray.remote
        def run_experiment(conf, func, rep, prog_ref):
            prog_ref.report_new_task.remote(conf, func, rep)
            conf_manager.run_configuration(conf, func, rep)

        progress_mgr_class = ray.remote(ProgressManager)
        progress_ref = progress_mgr_class.remote(num_total_experiments)

        ray.put(conf_manager)
        ids = [run_experiment.remote(conf, func, rep, progress_ref) for rep, func, conf in experiment_list]
        ray.get(ids)
    else:
        def run_experiment(conf, func, rep, progress_mgr):
            progress_mgr.report_new_task(conf, func, rep)
            conf_manager.run_configuration(conf, func, rep)

        progress_manager = ProgressManager(num_total_experiments)
        [run_experiment(conf, func, rep, progress_manager) for rep, func, conf in experiment_list]


if __name__ == '__main__':
    main()
