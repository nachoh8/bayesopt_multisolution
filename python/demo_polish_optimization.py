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
from bayesoptmodule import BayesOptContinuous
import numpy as np

from time import clock

class Rosenbrock(BayesOptContinuous):
    def sqr(self,x):
        return x*x

    def evaluateSample(self,Xin):
        dim = len(Xin)
        res = 0.0
        
        for i in range(dim-1):
            sqr = self.sqr
        
            xi = Xin[i]
            xn = Xin[i+1]
            res += 100*sqr(sqr(xi)-xn) + sqr(xi-1)
        
        return res


# Let's define the parameters
params = {} 
params['n_iterations'] = 80
params['n_init_samples'] = 20
params['n_final_samples'] = 20

params['noise'] = 1e-7
params['surr_name'] = "sStudentTProcessJef"
params['n_iter_relearn'] = 1

n = 5                     # n dimensions

bo_test = Rosenbrock(n)
bo_test.parameters = params
bo_test.lower_bound = np.ones((n,))*-5
bo_test.upper_bound = np.ones((n,))*10

start = clock()
mvalue, x_out, error = bo_test.optimize()

print "Result", x_out
print "Seconds", clock() - start

