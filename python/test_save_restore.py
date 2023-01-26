#!/usr/bin/env python

"""This file is used for manual testing purposes. It allows you to check the
   save/restore functionality step by step.
   
   Execute a few steps and then exit execution anytime (kill the process).
   Then execute again loading the dat file generated in the previous execution.
   The objective is to see how optimization loads/saves the state in each step.
   
   You can use the following command to see how the dat file updates:
    >> watch -n 1 cat test_save_restore.dat
    
   Test cases that should be checked:
   - Values printed should be the same that the values of the dat file.
   - If saving, before the first step, initial x points should be generated.
   - If loading a state that was stopped during initial samples, then the
     non evaluated initial samples should be generated on each step
   - If loading a state that was stopped, execution should continue onwards.
"""

import bayesopt
from bayesoptmodule import BayesOptContinuous
import numpy as np

from time import clock

# Function for testing.
def testfunc(Xin):
    try:
        input("Press enter to continue")
    except SyntaxError:
        pass
    
    total = 5.0
    for value in Xin:
        total = total + (value -0.33)*(value-0.33)
        
    print "x=", Xin
    print "y=", total

    return total

# Class for OO testing.
class BayesOptTest(BayesOptContinuous):
    def evaluateSample(self,Xin):
        return testfunc(Xin)


# Ask user for load_save_flag value
print "Provide a load_save_flag: 0=nothing, 1=load, 2=save, 3=load and save"
load_save_flag = input(">>")

params = {} 
params['n_iterations'] = 5
params['n_init_samples'] = 5
params['crit_name'] = "cSum(cEI,cDistance)"
params['crit_params'] = [1, 0.5]
params['kernel_name'] = "kMaternISO3"
params['load_save_flag'] = load_save_flag
params['save_filename'] = 'test_save_restore.dat'
params['load_filename'] = 'test_save_restore.dat'
print "Callback implementation"

n = 2                     # n dimensions
lb = np.zeros((n,))
ub = np.ones((n,))

print "OO implementation"
bo_test = BayesOptTest(n)
bo_test.parameters = params
bo_test.lower_bound = lb
bo_test.upper_bound = ub

start = clock()
mvalue, x_out, error = bo_test.optimize()

print "Result", x_out
print "Seconds", clock() - start

