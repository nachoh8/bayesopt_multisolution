#! /usr/bin/env python
import argparse
import cPickle
import bayesopt
import numpy

from joblib import Parallel, delayed

RNG = 1
numpy.random.seed(RNG)

from Surrogates.RegressionModels import GradientBoosting, KNN, LassoRegression, \
     LinearRegression, NuSupportVectorRegression, RidgeRegression, \
     SupportVectorRegression, RandomForest, GaussianProcess

# ArcGP, Fastrf, GaussianProcess

import Surrogates.DataExtraction.handle_configurations
import Surrogates.DataExtraction.configuration_space

def single_optimization(args,repetition):
    if args.problem == "onlineLDA":
        aux = ""
        aux2 = ""
        n = 3                     # n dimensions
    elif args.problem == "logreg":
        aux = "/nocv"
        aux2 = "_nocv"
        n = 4                     # n dimensions
    elif args.problem == "hpnnet":
        aux = "/nocv_mrbi"
        aux2 = "_nocv_mrbi"
        n = 14                     # n dimensions
    elif args.problem == "hpdbnet":
        aux = "/convex"
        #n = 3                     # n dimensions

    data_name = '/home/rmcantin/code/SurrogateBenchmarks/datasets/'+args.problem+aux+'/models/ENCODED_' + args.problem + aux2 + '_all_GradientBoosting'
    logfilename = args.problem + "_"

    if args.active_coef is not None:
        logfilename += "active%03d_" %(args.active_coef*1000)

    logfilename += args.simtype+str(repetition).zfill(2)+".log"

    print data_name
    # Read stuff we need to build input arrays
    fh = open(data_name, 'r')
    surrogate = cPickle.load(fh)

    def test_value_onlinelda(x):
        input_array = [0, int(x[1]*1023 + 1), int(x[2]*16383 + 1), x[0]/2.0 + 0.5]
        ans = surrogate.predict(input_array)
        return ans

    def test_value_logreg(x):
        input_array = [0, x[0]*1996 + 4, x[1]*1980 + 20, x[2], x[3]*10]
        ans = surrogate.predict(input_array)
        return ans

    def test_value_hpnnet(x):
        mins=[-24.3136667826,0.200000000001,0,
              -20.7232658368,0,0,0,
              4.60517018600,0,
              2.77258872225,0,0,
              0.500000000000001,-14.7247599296]
        maxs=[-0.145952402016,1.9999999999999, 2,
              -6.90775527899,3,2,2,
              9.21034037197,2,
              6.9314718055,4,2,
              0.999999999999999,8.79818982746]
        sxc=[]
            
        for xv,mi,ma in zip(x,mins,maxs):
            sxc.append(xv*(ma-mi)+mi)

        for i in [2,4,5,6,8,10,11]:
            sxc[i]=int(abs(sxc[i]-1e-10)) # super dirty hack!
            
        input_array = [0]+sxc
        ans = surrogate.predict(input_array)
        return ans

    def test_value_hpdbnet(x):
        ans = surrogate.predict(input_array)
        return ans

        
    lb = numpy.zeros((n,))
    ub = numpy.ones((n,))


    params = {}
    params['n_iterations'] = args.iters
    params['n_iter_relearn'] = 1
    params['noise'] = 1e-10
    params['l_type'] = 'L_MCMC'
    params['random_seed'] = repetition

    if args.active_coef is not None:
        params['active_hyperparams_coef'] = args.active_coef

    if args.simtype == "nonstationary":
        params['kernel_name'],"kNonSt(kMaternARD5,kMaternARD5)"
        domain_center = []
        domain_std = []
        for i in range(n):
            domain_center.append([0.5, 0.5])
            domain_std.append([10, 0.05])

        params['kernel_domain_center'] = domain_center
        params['kernel_domain_std'] = domain_std

    elif args.simtype == "moving":
        params['kernel_name'] = "kNonStMoving(kMaternARD5,kMaternARD5)"
        domain_center = numpy.zeros((2,n))
        domain_std = numpy.zeros((2,n))
        kernel_mean = numpy.zeros(3*n)
        kernel_std = numpy.zeros(3*n)
        for i in range(n):
            domain_center[:,i] = [0.5, 0.5]
            domain_std[:,i] = [0.05, 10]
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

    elif args.simtype == "warped":
        params['kernel_name'] = "kWarp(kMaternARD5)"
        kernel_mean = numpy.zeros(3*n)
        kernel_std = numpy.zeros(3*n)
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

    if args.problem == "onlineLDA":
        test_value = test_value_onlinelda
    elif args.problem == "logreg":
        test_value = test_value_logreg
    elif args.problem == "hpnnet":
        test_value = test_value_hpnnet
    elif args.problem == "hpdbnet":
        test_value = test_value_hpdbnet

        
    mvalue, x_out, error = bayesopt.optimize_log(test_value, n, lb, ub,
                                                 logfilename, params)

    print "Result", mvalue, "at", x_out
    fh.close()


def main(args):
    Parallel(n_jobs=args.jobs)(delayed (single_optimization)(args, repetition) for repetition in range(args.init_rep, args.init_rep+args.n_simulations))


if __name__ == "__main__":
    prog = "python surrogate_datasets"
    parser = argparse.ArgumentParser(description="Bayesian Optimization test based on surrogate datasets", prog=prog)

    # IPC infos
    parser.add_argument('-i','--iterations', dest="iters", required=False,
                        default=200, type=int, help="number of iterations")
    parser.add_argument('-j','--jobs', dest="jobs", required=False,
                        default=1, type=int, help="number of parallel jobs")
    parser.add_argument('-p','--problem', dest="problem", required=False,
                        default='onlineLDA', choices=["onlineLDA","logreg","hpnnet","hpdbnet"])
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
                        choices=["standard","nonstationary","moving",
                                 "warped","spsa"],                        
                        help= "standard, nonstationary,"
                        "moving, warped, spsa")
    parser.add_argument('-v','--verbose', dest="verbose", required=False,
                        default=1, type=int,
                        help="verbose level: 0 (warning), 1 (info), 2 (debug)")
    
    args = parser.parse_args()
    main(args)
