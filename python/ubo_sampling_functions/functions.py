import math
import numpy as np
from scipy.stats import multivariate_normal

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
            print('ERROR:' + str(e))

class GaussianMixture(object):
    def __init__(self):
        self.dim = 2
        self.lb = np.zeros(self.dim)
        self.ub = np.ones(self.dim)

        self.mu_list = [
            [1.5, 0.0], 
            [3.0, 3.0], 
            [4.0, -1.0], 
            [0.0, -2.0], 
            [2.0, 2.0], 
            [2.0, -1.0],
        ]

        #print([((mu[0] +1) /7, (mu[1]+4)/8) for mu in self.mu_list])

        self.cov_list = [
            [[1.0, 0.0],  [0.0, 1.0]],
            [[5.0, 0.0],  [0.0, 1.0]],
            [[2.0, -0.5], [0.5, 4.0]],
            [[0.3, 0],    [0, 0.3]],
            [[10.0, 0],   [0, 10.0]],
            [[5.0, 0],    [0, 5.0]],
        ]

        self.p_list = [
            0.2,
            0.5,
            0.5,
            0.1,
            0.5,
            0.5,
        ]
        
    def reverse_norm(self, x):
        x[0] = (x[0]+1)/7.0
        x[1] = (x[1]+4)/8.0
        return x

    def problem_name(self):
        return 'gmixture'


    def bounds(self):
        return (self.lb, self.ub)

    def call(self, X):
        assert len(X) == self.dim

        xmod = [0, 0]
        
        xmod[0] = 1-X[0]-0.1 +0.03
        xmod[1] = X[1] +0.03
        
        xmod[0] = xmod[0] * 7 - 1
        xmod[1] = xmod[1] * 8 - 4

        try:
            y = 0.0
            for i in range(len(self.p_list)):
                y += self.p_list[i] * multivariate_normal.pdf(xmod, self.mu_list[i], cov=self.cov_list[i])

            return -y

        except Exception as e:
            print('ERROR:' + str(e))

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
            print('ERROR:' + str(e))


class Polynomial(object):
    def __init__(self):
        self.dim = 2
        self.optimum = [2.82, 4.0]

    def problem_name(selfs):
        return 'polynomial'

    def bounds(selfs):
        lb = np.array([-0.95, -0.45])
        ub = np.array([3.2, 4.4])
        return(lb, ub)

    def call(self, X):
        assert len(X) == self.dim
        try:
            x = X[0]
            y = X[1]
            value = -2*x**6 + 12.2*x**5 - 21.2*x**4 - 6.2*x + 6.4*x**3 + 4.7*x**2 - y**6 + 11*y**5 \
                    - 43.3*y**4 + 10*y + 74.8*y**3 - 56.9*y**2 + 4.1*x*y + 0.1*y**2*x**2 - 0.4*y**2*x - 0.4*x**2*y
            return -value

        except Exception as e:
            print('ERROR:' + str(e))

def run_function(bopt_params={}, func=None, exp_folder='./output/', num_trials=10):
    import bayesopt

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
