import math
import bayesopt
import numpy as np

class Michalewicz(object):
    def __init__(self, dim, m=10.0):
        self.dim = dim
        self.m = m

    def problem_name(self):
        return 'micha' + str(self.dim)

    def bounds(self):
        lb = np.array([0.0] * self.dim)
        ub = np.array([math.pi] * self.dim)
        return (lb, ub)

    def call(self, X):
        assert len(X) == self.dim
        try:            
            accum = 0.0
            for i in range(self.dim):
                ii = float(i+1)
                accum += math.sin(X[i]) * math.sin(ii * X[i] * X[i] / math.pi) ** (2*self.m)
            return -accum

        except Exception as e:
            print 'ERROR:', str(e)

class Bohachevsky(object):
    def __init__(self):
        self.dim = 2

    def problem_name(self):
        return 'bohachevsky'

    def bounds(self):
        lb = np.array([-100.0, -100.0])
        ub = np.array([100.0, 100.0])
        return (lb, ub)

    def call(self, X):
        assert len(X) == self.dim
        try:    
            x0 = X[0]-0.1
            x1 = X[1]-0.1
            
            y = 0.7 + x0**2 + 2.0*x1**2
            y -= 0.3*np.cos(3*np.pi*x0)
            y -= 0.4*np.cos(4*np.pi*x1)
            return y

        except Exception as e:
            print 'ERROR:', str(e)

def run_function(bopt_params={}, func=None, exp_folder='./output/', num_trials=10):
    assert func is not None
    
    # Problem
    lb, ub = func.bounds()

    for r in range(num_trials):
        # Optimizer
        params = {}
        params['n_iterations'] = 75
        params['n_iter_relearn'] = 5
        params['n_init_samples'] = 10
        params['n_parallel_samples'] = 4
        params['crit_name'] = 'cEI'
        params['par_type'] = 'PAR_MCMC'
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

        if params['par_type'] == 'PAR_NONE':
            strat = 'bo_sequential'

        params['save_filename'] = exp_folder + '{strat}_{crit}_x{par}_{problem}_rep_{rep}.dat'.format(
            crit=params['crit_name'], par=params['n_parallel_samples'], rep=r,
            strat=strat, problem=func.problem_name() # +'burnout'+str(params['burnout_mcmc_parallel'])
        )

        mvalue, x_out, error = bayesopt.optimize(func.call, func.dim, lb, ub, params)


if __name__ == "__main__":
    run_experiment(sys.argv[1:])