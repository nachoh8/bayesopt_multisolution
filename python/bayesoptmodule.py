#!/usr/bin/env python
## \file bayesoptmodule.py \brief BayesOpt wrapper for Python interface (OOP)

# ----------------------------------------------------------------------------
#    This file is part of BayesOpt, an efficient C++ library for 
#    Bayesian optimization.
#
#    Copyright (C) 2011-2015 Ruben Martinez-Cantin <rmcantin@unizar.es>
#
#    BayesOpt is free software: you can redistribute it and/or modify
#    it under the terms of the GNU Affero General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    BayesOpt is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU Affero General Public License for more details.
#
#    You should have received a copy of the GNU Affero General Public License
#    along with BayesOpt. If not, see <http://www.gnu.org/licenses/>.
# ----------------------------------------------------------------------------

import numpy as np
import bayesopt as bo
import sys

## @addtogroup BayesOptPython
# Python Module for BayesOpt.
#
# Check test.py and test2.py with examples about how to run this code.
# Those files also provides examples about how to run the optimization
# using a callback pattern. (see bayesopt.pyx)

## Python Module for BayesOptContinuous
#
# Python module to get run BayesOpt library in a OO pattern.
# The objective module should inherit this one and override evaluateSample.
class BayesOptContinuous(object):
    
    ## Let's define the parameters.
    #
    # For different options: see parameters.h and parameters.cpp .
    # If a parameter is not defined, it will be automatically set
    # to a default value.
    def __init__(self,n_dim):
        ## Library parameters 
        self.params = {}
        ## n dimensions
        self.n_dim = n_dim
        ## Lower bounds
        self.lb = np.zeros((self.n_dim,))
        ## Upper bounds
        self.ub = np.ones((self.n_dim,))
        ## Python inner error while evaluating samples
        self.inner_error = None

        ##File for logging
        self._logfile = None

    @property
    def parameters(self):
        return self.params

    @parameters.setter
    def parameters(self,params):
        self.params = params

    @property
    def lower_bound(self):
        return self.lb

    @lower_bound.setter
    def lower_bound(self,lb):
        self.lb = lb

    @property
    def upper_bound(self):
        return self.ub

    @upper_bound.setter
    def upper_bound(self,ub):
        self.ub = ub

    @property
    def logfile(self):
        return self._logfile

    @logfile.setter
    def logfile(self,name):
        self._logfile = name

            
    ## Catches python exceptions on evaluteSample()
    # Saves exception to reraise it later at optimize()
    def tryEvaluateSample(self, x_in):
        try:
            return self.evaluateSample(x_in)
        except Exception:
            self.inner_error = sys.exc_info()
            raise
    
    def tryEvaluateSamples(self, x_in):
        try:
            return self.evaluateSamples(x_in)
        except Exception:
            self.inner_error = sys.exc_info()
            raise


    ## Function for testing.
    # It should be overriden.
    def evaluateSample(self, x_in):
        raise NotImplementedError("Please Implement this method")
    
    def evaluateSamples(self, x_in):
        raise NotImplementedError("Please Implement this method")

    ## Main function. Starts the optimization process.
    def optimize(self):
        try:
            if "par_type" in self.params and (self.params["par_type"] == "PAR_BBO" or self.params["par_type"] == "PAR_MCMC"):
                min_val, x_out, error = bo.optimize_parallel(self.tryEvaluateSamples, 
                                                self.n_dim,
                                                self.lb, self.ub,
                                                self.params)
            else:
                if self._logfile is None:
                    min_val, x_out, error = bo.optimize(self.tryEvaluateSample, 
                                                    self.n_dim,
                                                        self.lb, self.ub,
                                                        self.params)
                else:
                    min_val, x_out, error = bo.optimize_log(self.evaluateSample, self.n_dim,
                                                            self.lb, self.ub, self._logfile,
                                                            self.params)
        except Exception:
            if self.inner_error is not None:
                raise Exception(self.inner_error[0], self.inner_error[1], self.inner_error[2])
            else:
                raise
        return min_val, x_out, error

## Python Module for BayesOptDiscrete
#
# Python module to get run BayesOpt library in a OO pattern.
# The objective module should inherit this one and override evaluateSample.
class BayesOptDiscrete:
    
    ## Let's define the parameters.
    #
    # For different options: see parameters.h and parameters.cpp .
    # If a parameter is not define, it will be automatically set
    # to a default value.
    def __init__(self, x_set=None, n_dim=None, n_samples=None):
        ## Library parameters 
        self.params = {} #bo.initialize_params()
        self.x_set = x_set
        self.inner_error = None
        if self.x_set is None:
            ## Set of discrete points
            if n_dim is None or n_samples is None:
                raise ValueError
            else:
                self.x_set = np.random.rand(n_samples, n_dim)

    @property
    def parameters(self):
        return self.params

    @parameters.setter
    def parameters(self,params):
        self.params = params

    ## Catches python exceptions on evaluteSample()
    # Saves exception to reraise it later at optimize()
    def tryEvaluateSample(self, x_in):
        try:
            return self.evaluateSample(x_in)
        except Exception:
            self.inner_error = sys.exc_info()
            raise
                    
    ## Function for testing.
    # It should be overriden.
    def evaluateSample(self, x_in):
        raise NotImplementedError("Please Implement this method")

    ## Main function. Starts the optimization process.
    def optimize(self):
        try:            
            min_val, x_out, error = bo.optimize_discrete(
                                                    self.tryEvaluateSample,
                                                    self.x_set,
                                                    self.params)
        except Exception:
            if self.inner_error is not None:
                raise Exception(self.inner_error[0], self.inner_error[1], self.inner_error[2])
            else:
                raise
        return min_val, x_out, error


## Python Module for BayesOptCategorical
#
# Python module to get run BayesOpt library in a OO pattern.
# The objective module should inherit this one and override evaluateSample.
class BayesOptCategorical:
    
    ## Let's define the parameters.
    #
    # For different options: see parameters.h and parameters.cpp .
    # If a parameter is not define, it will be automatically set
    # to a default value.
    def __init__(self, categories=None):
        ## Library parameters 
        self.params = {}
        self.categories = categories
        self.inner_error = None

    @property
    def parameters(self):
        return self.params

    @parameters.setter
    def parameters(self,params):
        self.params = params

    ## Catches python exceptions on evaluteSample()
    # Saves exception to reraise it later at optimize()
    def tryEvaluateSample(self, x_in):
        try:
            return self.evaluateSample(x_in)
        except Exception:
            self.inner_error = sys.exc_info()
            raise
                    
    ## Function for testing.
    # It should be overriden.
    def evaluateSample(self, x_in):
        raise NotImplementedError("Please Implement this method")
    
    ## Main function. Starts the optimization process.
    def optimize(self):
        try:
            min_val, x_out, error = bo.optimize_categorical(
                                                        self.tryEvaluateSample,
                                                        self.categories,
                                                        self.params)
        except Exception:
            if self.inner_error is not None:
                raise Exception(self.inner_error[0], self.inner_error[1], self.inner_error[2])
            else:
                raise
        return min_val, x_out, error


if __name__ == "__main__":
    BO = BayesOptContinuous()
    __value__, __x__, __err__ = BO.optimize()
    print("Result", __x__)
