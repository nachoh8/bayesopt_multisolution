import bayesopt
import numpy as np

def rastrigin(x):
    print 'X:',x
    res1 = 20 + x[0]**2 - 10 * np.cos(2 * np.pi * x[0])
    res2 = x[1]**2 - 10 * np.cos(2 * np.pi * x[1])
    return res1+res2

result = []

for ii in range(20):
    params = {}
    params['random_seed'] = ii
    params['n_iterations'] = 100
    params['n_iter_relearn'] = 1
    params['l_type'] = 'L_MCMC'
    params['kernel_name'] = 'kMaternARD5'
    params['n_init_samples'] = 10
    params['noise'] = 1e-10
    params['force_jump'] = 5

    print "Callback implementation"

    n = 2                     # n dimensions
    lb = np.ones((n,)) * -5.11
    ub = np.ones((n,)) *  5.11

    mvalue, x_out, error = bayesopt.optimize(rastrigin, n, lb, ub, params)
    result.append(mvalue)

print result
res = np.array(result)
print np.mean(res), np.std(res)
