#!/usr/bin/env python
# -------------------------------------------------------------------------
#    This file is part of BayesOpt, an efficient C++ library for
#    Bayesian optimization.
#
#    Copyright (C) 2011-2014 Ruben Martinez-Cantin <rmcantin@unizar.es>
#
#    BayesOpt is free software: you can redistribute it and/or modify it
#    under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    BayesOpt is distributed in the hope that it will be useful, but
#    WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with BayesOpt.  If not, see <http://www.gnu.org/licenses/>.
# ------------------------------------------------------------------------

import math
import numpy as np
import bayesopt
import scipy.optimize
from bayesoptmodule import BayesOptContinuous
import matplotlib.pyplot as plt

def eggholder_modified(x):
    y = -(x[1] + 47) * \
        np.sin(np.sqrt(abs(x[1] + x[0] / 2 + 47))) - \
        x[0] * np.sin(np.sqrt(abs(x[0] - (x[1] + 47))))
    return -y

class EggHolder(BayesOptContinuous):
    def __init__(self):
        super(EggHolder,self).__init__(2)
        self._save = True
        self.xplot = []
        self.yplot = []
        
    # This is not the standard test for the eggoholder
    def evaluateSample(self,x):
        y = -eggholder_modified(x)
        if self._save:
            self.yplot.append(y)
            self.xplot.append(x)
        return -y

    @property
    def save(self):
        return self._save

    @save.setter
    def save(self,save):
        self._save = save

        
def bayesopt_test():

    egg = EggHolder()
    
    # Initialize the parameters by default
    params = {} 
    params['n_iterations']   = 4
    params['random_seed']    = 100
    params['n_init_samples'] = 10
    params['n_iter_relearn'] = 1
    params['noise'] = 1e-6
    params['l_type'] = 'mcmc'
    #params['surr_name'] = "sStudentTProcessNIG"
    #params['crit_name'] = "cMI"

    egg.params = params
    
    dim = 2
    egg.lower_bound = np.ones((dim,))*-100
    egg.upper_bound = np.ones((dim,))*100

    mvalue, x_out, error = egg.optimize()

    print "Result", mvalue, x_out
    
    return egg

def evol_test():
    egg2 = EggHolder()
    fpoint = egg2.evaluateSample
    print "Testing Evolutionary"
    scipy.optimize.differential_evolution(fpoint,[(-100,100), (-100,100)],maxiter=4,seed=100,polish=False)
    return egg2

from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter

def plot_obj_func(obj, name, res=64):
    obj.save = False

    xplot = np.asarray(obj.xplot)
    yplot = np.asarray(obj.yplot)

    x = np.linspace(-100,100, res )
    y = np.linspace(-100,100, res )
    X, Y = np.meshgrid(x, y)
    Z = np.zeros((res, res))
    
    for i, p1 in enumerate(x):
        for j, p2 in enumerate(y):
            Z[j,i] = -obj.evaluateSample(np.asarray([p1,p2]))

    
    #ax_3d = fig.add_subplot(131, projection='3d')
    fig2 = plt.figure()
    ax_3d = fig2.add_subplot(111,projection='3d')
    ax_3d.plot(xplot[:,0],xplot[:,1],yplot,'k*')
    ax_3d.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)
    

    fig2.savefig(name+'.png',bbox_inches='tight')

    fig = plt.figure(figsize=(24, 7))

    ax_contour = fig.add_subplot(132)
    ax_contour.contour(X, Y, Z)
    ax_contour.plot(xplot[:,0],xplot[:,1],'*')
    
    ax_best = fig.add_subplot(133)
    ax_best.plot(yplot, 'b*')
    ax_best.set_xlabel("Number of Evaluations")
    ax_best.set_ylabel("Value Observed")

    plt.show()    

#egg =  bayesopt_test()
#plot_obj_func(egg,'bo_egg')
egg2 =  evol_test()
plot_obj_func(egg2,'dif_egg')

