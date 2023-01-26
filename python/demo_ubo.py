#!/usr/bin/env python
# -------------------------------------------------------------------------
#    This file is part of BayesOpt, an efficient C++ library for
#    Bayesian optimization.
#
#    Copyright (C) 2011-2015 Ruben Martinez-Cantin <rmcantin@unizar.es>
#
#    BayesOpt is free software: you can redistribute it and/or modify it
#    under the terms of the GNU Affero General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    BayesOpt is distributed in the hope that it will be useful, but
#    WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU Affero General Public License for more details.
#
#    You should have received a copy of the GNU Affero General Public License
#    along with BayesOpt.  If not, see <http://www.gnu.org/licenses/>.
# ------------------------------------------------------------------------

import bayesopt
import numpy as np
from rkhs import rkhs_synth

# Python3 compat
if hasattr(__builtins__, 'raw_input'):
    input = raw_input

def inverse_rkhs(x):
    return -rkhs_synth(x)

rkhs_array = np.frompyfunc(rkhs_synth, 1, 1)

# Let's define the parameters
# For different options: see parameters.h and cpp
# If a parameter is not define, it will be automatically set
# to a default value.
input_var = 0.01 * 0.01
params = {}
params['n_iterations'] = 50
params['n_iter_relearn'] = 1
params['l_type'] = 'mcmc'
params['n_init_samples'] = 15
params['noise'] = 1e-4
params['force_jump'] = 3
#params['sigma_s'] = 4
params['verbose_level'] = -1
params['input_noise_variance'] = input_var

# Spartan
params['kernel_name'] = "kNonStMoving(kMaternARD5,kMaternARD5)"
params['kernel_hp_mean'] = [0.5, 1, 1]
params['kernel_hp_std'] = [0.2, 5, 5]
params['kernel_domain_center'] = np.asarray([[0.5], [0.5]])
params['kernel_domain_std'] = np.asarray([[0.05], [10]])

n = 1                     # n dimensions
lb = np.zeros((n,))
ub = np.ones((n,))

rep = 20
averages = np.zeros(rep)


for i in range(rep):
    mvalue, x_out, error = bayesopt.optimize(inverse_rkhs, n, lb, ub, params)
    averages[i] = np.mean(rkhs_array(np.random.normal(x_out[0], np.sqrt(input_var), 10000)))
    print("rep {}: {} -> {} | ave: {}".format(i, x_out, mvalue, averages[i]))
    

print("Final:", np.mean(averages))


# print("Result", mvalue, "at", x_out)
# print("Average", np.mean(rkhs_array(np.random.normal(x_out[0], np.sqrt(input_var), 100))))
