#!/usr/bin/env python

# ----------------------------------------------------------------------------
#    This file is part of BayesOpt, an efficient C++ library for 
#    Bayesian optimization.
#
#    Copyright (C) 2011-2015 Ruben Martinez-Cantin <rmcantin@unizar.es>
#
#    BayesOpt is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    BayesOpt is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with BayesOpt. If not, see <http://www.gnu.org/licenses/>.
# ----------------------------------------------------------------------------

import argparse
import cPickle

import time

import numpy as np
import bayesopt
from bayesoptmodule import BayesOptContinuous
from MountainCarEnvironment import MCEnvironment
from CartPoleEnvironmentGN2 import CartPoleEnvironment
#from AcrobotEnvironmentG import AcrobotEnvironment

from joblib import Parallel, delayed

RNG = 1
np.random.seed(RNG)


class NNPolicy:
    def __init__(self, xinit):
        pass
    
    def update(self, w):
        # We map the weights in the [-50, 50]
        self.w = np.tan((np.pi-0.0001)*w-np.pi/2)


    def policy(self,x):
        n = len(x)
        sumx = np.dot(self.w[0:n],x)
        return np.tanh(self.w[-1]*sumx)
        # sumx1 = np.dot(self.w[0:n],x)
        # sumx2 = np.dot(self.w[n:-2],x)
        # return np.tanh(np.tanh(self.w[-2]*sumx1) + np.tanh(self.w[-1]*sumx2)) 
        

class MCPolicy(NNPolicy):
    def __init__(self):
        # self.dim = 10
        self.dim = 5
        self.w = None

    def start(self,xinit):
        self.v_ant = xinit[1]

                
    def action(self, x):
        if self.w is None:
            raise ValueError('You need to update the policy first!')
        pos = x[0]
        vel = x[1]
        
        acc = vel - self.v_ant
        self.v_ant = vel
        
        return self.policy(np.array([1,pos,vel,acc]))

    
class CPPolicy(NNPolicy):
    def __init__(self):
        # self.dim = 16
        self.dim = 8
        self.w = None

    def start(self,xinit):
        self.v_ant = xinit[1]
        self.v_antt= xinit[3]

            
    def action(self, x):
        if self.w is None:
            raise ValueError('You need to update the policy first!')
        pos = x[0]
        vel = x[1]

        post = x[2]
        velt = x[3]
                
        acc = vel - self.v_ant
        v_ant = vel

        acct = velt - self.v_antt
        self.v_antt = velt
                
        action = self.policy(np.array([1,pos,vel,acc,post,velt,acct]))
        return action

    
class BayesOptRL(BayesOptContinuous):

    def __init__(self,Environment,Policy):
        self.Environment = Environment
        s                = self.Environment.GetInitialState()
        self.policy      = Policy
        self.policy.start(s)
        self.n           = self.policy.dim
        super(BayesOptRL, self).__init__(self.n)

        
    def evaluateSample(self,weights):
        s                = self.Environment.GetInitialState()
        steps            = 0
        total_reward     = 0
        r                = 0

        self.policy.update(weights)

        maxsteps = 1000
        
        for i in range(1,maxsteps+1):

            # selects an action
            a      = self.policy.action(s)
            
            # do the selected action and get the next car state
            s      = self.Environment.DoAction(a,s)

            # if self.Environment.graphs:
            #     print "Distance", self.Environment.goalPos - s[0]
            
            # observe the reward at state xp and the final state flag
            r,isfinal    = self.Environment.GetReward(s)
            total_reward = total_reward + r
            
            #increment the step counter.
            steps = steps + 1

            # if reachs the goal breaks the episode
            if isfinal==True:
                break

        return -total_reward

    def plotSample(self,weights):
        self.Environment.graphs = True
        self.Environment.initGraphs()
        return self.evaluateSample(weights)
    

def single_optimization(args,repetition):
    if args.model == "mc":
        logfilename = "mountaincar_"
        Env = MCEnvironment()
        Pol = MCPolicy()
    elif args.model == "cp":
        logfilename = "cartpole_"
        Env = CartPoleEnvironment()
        Pol = CPPolicy()
    else:
        logfilename = "acrobot_"
        Env = AcrobotEnvironment()
        Pol = CPPolicy()

    
    if args.active_coef is not None:
        logfilename += "active%03d_" %(args.active_coef*1000)

    logfilename += args.simtype+str(repetition).zfill(2)+".log"

    BOMC = BayesOptRL(Env,Pol)
    n = BOMC.n
        
    BOMC.lower_bound = np.zeros((n,))
    BOMC.upper_bound = np.ones((n,))

    params = {}
    params['n_iterations'] = args.iters
    params['n_iter_relearn'] = 1
    params['noise'] = 1e-10
    params['l_type'] = 'L_MCMC'
    params['random_seed'] = repetition

    if args.active_coef is not None:
        params['active_hyperparams_coef'] = args.active_coef

    elif args.simtype == "nst":
        params['kernel_name'] = "kNonStMoving(kMaternARD5,kMaternARD5)"
        domain_center = np.zeros((2,n))
        domain_std = np.zeros((2,n))
        kernel_mean = np.zeros(3*n)
        kernel_std = np.zeros(3*n)
        for i in range(n):
            domain_center[:,i] = [0.5, 0.5]
            domain_std[:,i] = [0.01, 10]
            kernel_mean[i] = 0.5
            kernel_mean[i+n] = 1
            kernel_mean[i+2*n] = 1
            kernel_std[i] = 0.2
            kernel_std[i+n] = 5
            kernel_std[i+2*n] = 5
            

        params['kernel_domain_center'] = domain_center
        params['kernel_domain_std'] = domain_std
        params['kernel_hp_mean'] = kernel_mean
        params['kernel_hp_std'] = kernel_std

    elif args.simtype == "warp":
        params['kernel_name'] = "kWarp(kMaternARD5)"
        kernel_mean = np.zeros(3*n)
        kernel_std = np.zeros(3*n)
        for i in range(n):
            kernel_mean[i] = 2
            kernel_mean[i+n] = 2
            kernel_mean[i+2*n] = 1
            kernel_std[i] = 0.5
            kernel_std[i+n] = 0.5
            kernel_std[i+2*n] = 10

        params['kernel_hp_mean'] = kernel_mean
        params['kernel_hp_std'] = kernel_std

            
    elif args.simtype == "spsa":
        params['n_iterations'] = args.iters/2;
        params['use_spsa'] = 1

    BOMC.params = params
    BOMC.logfile = logfilename
        
    mvalue, x_out, error = BOMC.optimize()

    #BOMC.plotSample(x_out)
    
    print "Result", mvalue, "at", x_out






def main(args):
    Parallel(n_jobs=args.jobs)(delayed (single_optimization)(args, repetition) for repetition in range(args.init_rep, args.init_rep+args.n_simulations))


if __name__ == "__main__":
    prog = "python surrogate_datasets"
    parser = argparse.ArgumentParser(description="Bayesian Optimization for the Mountain Car", prog=prog)

    # IPC infos
    parser.add_argument('-i','--iterations', dest="iters", required=False,
                        default=200, type=int, help="number of iterations")
    parser.add_argument('-j','--jobs', dest="jobs", required=False,
                        default=1, type=int, help="number of parallel jobs")
    parser.add_argument('-a','--active', dest="active_coef", required=False,
                        default=None, type=float, help="active hyperparameter learning coeficient")
    parser.add_argument('-s','--start', dest="init_rep", required=False,
                        default=10, type=int, help="set initial seed")
    parser.add_argument('-r','--repetitions', dest="n_simulations", required=False,
                        default=1, type=int,
                        help="set number of tests with consecutive"
                        "seeds after the initial seed")
    parser.add_argument('-t','--type', dest="simtype", required=False,
                        default="standard",
                        choices=["standard", "nst",
                                 "warp","spsa"],                        
                        help= "standard, nst, warp, spsa")
    parser.add_argument('-e','--environment', dest="model", required=False,
                        default="ab",
                        choices=["mc", "cp", "ab"],                        
                        help= "Mountain Car, Cart and Pole, Acrobot.")
    parser.add_argument('-v','--verbose', dest="verbose", required=False,
                        default=1, type=int,
                        help="verbose level: 0 (warning), 1 (info), 2 (debug)")
    
    args = parser.parse_args()
    main(args)
