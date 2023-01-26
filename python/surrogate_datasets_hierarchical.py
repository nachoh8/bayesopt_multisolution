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
    if args.problem == "hpnnet":
        aux = "nocv_mrbi"
        n_cat = 7
        n_cont = 7
        #n = 3                     # n dimensions
    elif args.problem == "hpdbnet":
        aux = "mrbi"
        n_cat = 17
        n_cont = 19
        #n = 3                     # n dimensions

    data_name = '/home/rmcantin/code/SurrogateBenchmarks/datasets/'+args.problem+"/"+aux+'/models/ENCODED_' + args.problem +"_"+aux+'_all_GradientBoosting'
    logfilename = args.problem + "_"

    if args.active_coef is not None:
        logfilename += "active_"

    print data_name
    # Read stuff we need to build input arrays
    fh = open(data_name, 'r')
    surrogate = cPickle.load(fh)

    def test_value_hpdbnet(x):
        mins=[-3.0, -6.26016873291, -7.58562839264, 2.30258509299, -10.605170186,
              -14.7251883931, -20.462669088, 0.0, 0.0, 0.0,
              -33.5133245231, 4.60517018599,-15.8275163738, 4.85203026392, 4.85203026392,
              4.85203026392, 2.30258509299, 2.30258509299, 0.5]
        maxs=[3.22714454233, 4.01569873618, 3.0, 9.21034037198, 2.14701998895,
              2.30749104611, 2.49695414127, 7.31322038709, 7.60090245954, 8.00636756765,
              -2.33130904097, 9.21034037198, 9.63583601271, 8.31776616672, 8.31776616672,
              8.31776616672, 9.21034037198, 9.21034037198, 1.0]

        sxc = []
        for xv,mi,ma in zip(x,mins,maxs):
            sxc.append(xv*(ma-mi)+mi)

        def test_value_hpdbnet_cat(xc):            
            input_array = [0, xc[0], xc[1], sxc[0], sxc[1],  sxc[2], xc[2], xc[3], xc[4], xc[5],
                           sxc[3], sxc[4], sxc[5], sxc[6], sxc[7], sxc[8], sxc[9],
                            xc[6], xc[7], xc[8], sxc[10], xc[9], xc[10], sxc[11], sxc[12], sxc[13],
                            sxc[14], sxc[15], sxc[16], sxc[17], xc[11], xc[12], xc[13], xc[14],
                            sxc[18], xc[15], xc[16]]
            ans = surrogate.predict(input_array)
            return ans

        params_cat = {}
        params_cat['n_iterations'] = args.iters
        params_cat['n_init_samples'] = 5
        params_cat['n_iter_relearn'] = 1
        params_cat['noise'] = 1e-10
        params_cat['l_type'] = 'L_FIXED'
        params_cat['random_seed'] = repetition
        params_cat['kernel_name'] = "kHamming"
        params_cat['kernel_hp_mean'] = [10]
        params_cat['kernel_hp_mean'] = [0.1]
    
        categories = numpy.array([2,10, 4,2,2,2, 2,2,2, 2,2, 10,10,4,2, 2,2 ],dtype=int)

        mvalue, x_out, error = bayesopt.optimize_categorical(test_value_hpdbnet_cat,
                                                             categories,
                                                             params_cat)
        return mvalue

    
    def test_value_hpnnet(x):
        sxc = []
        
        mins=[-24.3136667827,0.2,-20.7232658369,4.60517018599,2.77258872224,0.5,-14.7247599297]
        maxs=[-0.145952402014,2.0,-6.90775527898,9.21034037198,6.9314718056,1.0,8.79818982747]
        for xv,mi,ma in zip(x,mins,maxs):
            sxc.append(xv*(ma-mi)+mi)

        
        def test_value_hpnnet_cat(xc):            
            input_array = [0, sxc[0], sxc[1], xc[0], sxc[2], xc[1], xc[2], xc[3],
                           sxc[3], xc[4], sxc[4], xc[5], xc[6], sxc[5], sxc[6]]
            ans = surrogate.predict(input_array)
            return ans

        params_cat = {}
        params_cat['n_iterations'] = args.iters
        params_cat['n_init_samples'] = 5
        params_cat['n_iter_relearn'] = 1
        params_cat['noise'] = 1e-10
        params_cat['l_type'] = 'L_FIXED'
        params_cat['random_seed'] = repetition
        params_cat['kernel_name'] = "kHamming"
        params_cat['kernel_hp_mean'] = [10]
        params_cat['kernel_hp_mean'] = [0.1]
    
        categories = numpy.array([2, 3, 2, 2, 2, 4, 2],dtype=int)

        mvalue, x_out, error = bayesopt.optimize_categorical(test_value_hpnnet_cat,
                                                             categories,
                                                             params_cat)
        return mvalue

  
    lb = numpy.zeros((n_cont,))
    ub = numpy.ones((n_cont,))


    params = {}
    params['n_iterations'] = args.iters
    params['n_init_samples'] = 2
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
        for i in range(n_cont):
            domain_center.append([0.5, 0.5])
            domain_std.append([10, 0.05])

        params['kernel_domain_center'] = domain_center
        params['kernel_domain_std'] = domain_std

    elif args.simtype == "moving":
        params['kernel_name'] = "kNonStMoving(kMaternARD5,kMaternARD5)"
        domain_center = numpy.zeros((2,n_cont))
        domain_std = numpy.zeros((2,n_cont))
        kernel_mean = numpy.zeros(3*n_cont)
        kernel_std = numpy.zeros(3*n_cont)
        for i in range(n_cont):
            domain_center[:,i] = [0.5, 0.5]
            domain_std[:,i] = [0.05, 10]
            kernel_mean[i] = 0.5
            kernel_mean[i+n_cont] = 1
            kernel_mean[i+2*n_cont] = 1
            kernel_std[i] = 0.2
            kernel_std[i+n_cont] = 5
            kernel_std[i+2*n_cont] = 5

        params['kernel_domain_center'] = domain_center
        params['kernel_domain_std'] = domain_std
        params['kernel_hp_mean'] = kernel_mean
        params['kernel_hp_std'] = kernel_std

    elif args.simtype == "warped":
        params['kernel_name'] = "kWarp(kMaternARD5)"
        kernel_mean = numpy.zeros(3*n_cont)
        kernel_std = numpy.zeros(3*n_cont)
        for i in range(n_cont):
            kernel_mean[i] = 2
            kernel_mean[i+n_cont] = 2
            kernel_mean[i+2*n_cont] = 1
            kernel_std[i] = 0.5
            kernel_std[i+n_cont] = 0.5
            kernel_std[i+2*n_cont] = 10

        params['kernel_hp_mean'] = kernel_mean
        params['kernel_hp_std'] = kernel_std


    elif args.simtype == "spsa":
        params['n_iterations'] = args.iters/2;
        params['use_spsa'] = 1

    logfilename += args.simtype+str(repetition).zfill(2)+".log"

    if args.problem == "hpnnet":
        test_function = test_value_hpnnet
    elif args.problem == "hpdbnet":
        test_function = test_value_hpdbnet

    mvalue, x_out, error = bayesopt.optimize_log(test_function, n_cont, lb, ub,
                                                 logfilename, params)

    
    print "Result", mvalue, "at", x_out
    fh.close()


def main(args):
    Parallel(n_jobs=args.jobs)(delayed (single_optimization)(args, repetition) for repetition in range(args.init_rep, args.init_rep+args.n_simulations))
    #[single_optimization(args, repetition) for repetition in range(args.init_rep, args.init_rep+args.n_simulations)]


if __name__ == "__main__":
    prog = "python surrogate_datasets"
    parser = argparse.ArgumentParser(description="Bayesian Optimization test based on surrogate datasets", prog=prog)

    # IPC infos
    parser.add_argument('-i','--iterations', dest="iters", required=False,
                        default=10, type=int, help="number of iterations")
    parser.add_argument('-i2','--iterations2', dest="iters2", required=False,
                        default=5, type=int, help="number of iterations")
    parser.add_argument('-j','--jobs', dest="jobs", required=False,
                        default=1, type=int, help="number of parallel jobs")
    parser.add_argument('-p','--problem', dest="problem", required=False,
                        default='hpnnet', choices=["hpnnet","hpdbnet"])
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
