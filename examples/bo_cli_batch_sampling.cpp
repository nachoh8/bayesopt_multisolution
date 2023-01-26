/*
-------------------------------------------------------------------------
   This file is part of BayesOpt, an efficient C++ library for
   Bayesian optimization.

   Copyright (C) 2011-2013 Ruben Martinez-Cantin <rmcantin@unizar.es>

   BayesOpt is free software: you can redistribute it and/or modify it
   under the terms of the GNU General Public License as published by
   the Free Software Foundation, either version 3 of the License, or
   (at your option) any later version.

   BayesOpt is distributed in the hope that it will be useful, but
   WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU General Public License for more details.

   You should have received a copy of the GNU General Public License
   along with BayesOpt.  If not, see <http://www.gnu.org/licenses/>.
------------------------------------------------------------------------
*/

#define _USE_MATH_DEFINES
#include <cmath>
#include <ctime>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <boost/program_options.hpp>

#include "testfunctions.hpp"

namespace po = boost::program_options;


std::string filename(const std::string &name, const std::string &type, size_t num)
{
  std::stringstream ss;
  ss << std::setw(2) << std::setfill('0') << num;
  std::string filename = name+type+ss.str()+".log";
  return filename;
}

int main(int nargs, char *args[])
{
  size_t init_rep = 10;
  size_t n_simulations = 1;
  size_t n_iter = 25;
  size_t dim = 2;
  std::string simtype = "standard";
  std::string surrtype = "gp";
  std::string problem_name = "branin";
  std::string robtype = "no";
  int verbose = 1;
  bool active = false;
  double active_coef = 0.1;
  int parallel;
  int burnout;
  std::string exp_conf;
  std::string exp_criteria;
  double temperature;
  std::string output_folder;
  double input_noise;

  try
    {
      po::options_description desc("BayesOpt test function optimization.");
      desc.add_options()
      ("help,h", "this help message")
      ("iterations,i", po::value<size_t>(&n_iter)->default_value(25), "number of iterations")
      ("active,a", po::value<double>(&active_coef), "coeficient for active hyperparameters")
      ("start,s", po::value<size_t>(&init_rep)->default_value(10), "set initial seed")
      ("repetitions,r", po::value<size_t>(&n_simulations)->default_value(1),
      "set number of tests with consecutive seeds after the initial seed")
      ("function,f", po::value<std::string>(&problem_name)->default_value("branin"),
      "exp2d,easom,branin,ackley2,egg,camelback,micha,rosen2,schubert,hart6")
      ("type,t", po::value<std::string>(&simtype)->default_value("standard"),
      "standard, nonstationary, spartan, moving, warped, spsa")
      ("surrogate,g", po::value<std::string>(&surrtype)->default_value("gp"),
      "gp,jef,nig,inf")
      ("verbose,v", po::value<int>(&verbose)->default_value(1),
      "verbose level: 0 (warning), 1 (info), 2 (debug)")
      ("parallel,p", po::value<int>(&parallel)->default_value(1),
      "parallel samples with criteria sampling")
      ("burnout", po::value<int>(&burnout)->default_value(2500),
      "burn-out period of mcmc for the criteria sampling")
      ("temperature", po::value<double>(&temperature)->default_value(1.0),
      "temperature for the criteria sampling")
      ("configuration", po::value<std::string>(&exp_conf)->default_value("batch_sampling"),
      "configuration set for the batch experiment")
      ("criteria", po::value<std::string>(&exp_criteria)->default_value("cEI"),
      "criteria for the batch experiment")
      ("robust,b", po::value<std::string>(&robtype)->default_value("no"),
      "no,ut")
      ("output_folder", po::value<std::string>(&output_folder)->default_value("./output/"),
      "folder where the output dat files will be saved")
      ("input_noise", po::value<double>(&input_noise)->default_value(-1.0))
      ;

      po::variables_map vm;
      po::store(po::parse_command_line(nargs, args, desc), vm);
      po::notify(vm);

      if (vm.count("help"))
        {
          std::cout << desc << "\n";
          return 0;
        }

      if (vm.count("active"))
        {
          std::cout << "active coefficient" << active_coef << std::endl;
          active = true;
        }
    }
  catch(std::exception& e)
    {
      std::cerr << "error: " << e.what() << "\n";
      return -1;
    }
  catch(...)
    {
      std::cerr << "Exception of unknown type!\n";
      return -1;
    }

  // The other problems are 2D
  if (problem_name == "micha") dim = 10;
  else if (problem_name == "hart6") dim = 6;    

  #pragma omp parallel for
  for(size_t repetition = init_rep; repetition < init_rep+n_simulations; repetition++)
  {
    bopt_params par = initialize_parameters_to_default();
    par.n_iterations = n_iter;
    par.noise = 1e-10;
    par.verbose_level = verbose;
    par.l_type = L_MCMC; //L_EMPIRICAL;
    par.n_iter_relearn = 1;
    par.n_parallel_samples = parallel;
    par.par_type = PAR_MCMC;
    par.load_save_flag = 2;     
    par.random_seed = repetition;
    par.burnout_mcmc_parallel = burnout;
    par.input_noise_variance = input_noise;

    par.l_all = false;
    par.force_jump = 5;

    std::string criteria_name = exp_criteria;
    
    std::stringstream ss; ss << std::setw(2) << std::setfill('0') << repetition;
    std::stringstream sspar; sspar << parallel;
    std::stringstream sstemp; sstemp << temperature;

    // Setup of the Configurations
    bool use_boltzmann = false;
    bool use_normalize = false;
    std::string output_prefix = exp_conf;

    if (exp_conf == "batch_sampling"){
      par.par_type = PAR_MCMC;
    }
    else if (exp_conf == "rejection_sampling"){
      par.par_type = PAR_REJ_SAMPLING;
    }
    else if (exp_conf == "parallel_tempering"){
      par.par_type = PAR_PAR_TEMPERING;
    }
    else if (exp_conf == "thompson_sampling"){
      par.par_type = PAR_THOMPSON;
      exp_criteria = "cThompsonSampling";
      output_prefix = "batch_sampling";
    }
    else if (exp_conf == "fixedtemp"){
      par.par_type = PAR_MCMC;
      use_boltzmann = true;
      use_normalize = exp_criteria == "cEI";

      output_prefix = exp_conf + "_" + sstemp.str();
    }
    else if (exp_conf == "sequential"){
      par.par_type = PAR_NONE;
      output_prefix = "bo_sequential";
      par.n_parallel_samples = 1;
    }
    else if (exp_conf == "random"){
      par.par_type = PAR_RANDOM;
      exp_criteria = "cRandom";
      output_prefix = "random_search";
    }

    set_criteria(&par, exp_criteria.c_str());
        
    if(exp_criteria == "cPseudoBayesian")
    {
      par.n_crit_params = 6;
      par.crit_params[0] = 1.0; // (ignore, shared parameter with any EI)
      par.crit_params[1] = 0.05; // EI temperature
      par.crit_params[2] = 20; // ES mNumCandidates
      par.crit_params[3] = 100; // ES mNumGPSamples
      par.crit_params[4] = 10; // ES mNumSamplesY
      par.crit_params[5] = 100; // ES mNumTrialPoints

      set_criteria(&par, "cProd(cEIexp, cEntropySearch)");
    }
    else if(exp_criteria == "cEntropySearch"){
      par.n_crit_params = 4;
      par.crit_params[0] = 50; // ES mNumCandidates
      par.crit_params[1] = 1000; // ES mNumGPSamples
      par.crit_params[2] = 100; // ES mNumSamplesY
      par.crit_params[3] = 500; // ES mNumTrialPoints
    }

    // Set up function
    if (active)
      {
	      par.active_hyperparams_coef = active_coef;
      }

    if (robtype == "ut")
      {
        par.input_noise_variance = 0.01 * 0.01;
      }

    if (surrtype == "jef")
      {
	      set_surrogate(&par, "sStudentTProcessJef");
      }
    else if (surrtype == "nig")
      {
	      set_surrogate(&par, "sStudentTProcessNIG");
      }
    else if (surrtype == "inf")
      {
        set_surrogate(&par, "sStudentTProcessNIG");
        par.alpha = 10;
        par.beta = 10;
      }

    if (simtype == "nonstationary")
      {
        set_kernel(&par,"kNonSt(kMaternARD5,kMaternARD5)");
        for(size_t i= 0; i< dim*2; ++i)
          {
            par.kernel.domain_center[0][i] = 0.5;
            par.kernel.domain_std[0][i]    = 100;
            par.kernel.domain_center[1][i] = 0.5;
            par.kernel.domain_std[1][i]    = 1;
          }

        par.kernel.n_basis_kernels     = 2;
        par.kernel.n_dim_domain        = dim;
            }
    else if (simtype == "moving")
      {
        set_kernel(&par,"kNonStMoving(kMaternARD5,kMaternARD5)");
        for(size_t i= 0; i< dim; ++i)
          {
            par.kernel.hp_mean[i] = 0.5; par.kernel.hp_std[i] = 0.2;
            par.kernel.hp_mean[i+dim] = 1; par.kernel.hp_std[i+dim] = 5;
            par.kernel.hp_mean[i+2*dim] = 1; par.kernel.hp_std[i+2*dim] = 5;
            par.kernel.n_hp = 3*dim;


            par.kernel.domain_center[1][i] = 0.5;
            par.kernel.domain_std[1][i]    = 10;
            par.kernel.domain_center[0][i] = 0.5;
            par.kernel.domain_std[0][i]    = 0.05;
          }

        par.kernel.n_basis_kernels     = 2;
        par.kernel.n_dim_domain        = dim;
      }
    else if (simtype == "spartan")
      {
        set_kernel(&par,"kNonStMoving(kMaternARD5,kMaternARD5)");
        for(size_t i= 0; i< dim; ++i)
          {
            par.kernel.hp_mean[i] = 0.5; par.kernel.hp_std[i] = 0.2;
            par.kernel.hp_mean[i+dim] = 1; par.kernel.hp_std[i+dim] = 5;
            par.kernel.hp_mean[i+2*dim] = 1; par.kernel.hp_std[i+2*dim] = 5;
            par.kernel.n_hp = 3*dim;


            par.kernel.domain_center[1][i] = 0.5;
            par.kernel.domain_std[1][i]    = 10;
            par.kernel.domain_center[0][i] = 0.5;
            par.kernel.domain_std[0][i]    = 0.05;
          }

        par.kernel.n_basis_kernels     = 2;
        par.kernel.n_dim_domain        = dim;
      }
    else if (simtype == "warped")
      {
        set_kernel(&par,"kWarp(kMaternARD5)");
        for(size_t i= 0; i< dim*2; ++i)
          {
            // Betacdf parameters for sigmoid function
            par.kernel.hp_mean[i] = 2;
            par.kernel.hp_std[i] = 0.5;
          }
        for(size_t i= dim*2; i< dim*3; ++i)
          {
            // MarternARD5 parameters
            par.kernel.hp_mean[i] = 1;
            par.kernel.hp_std[i] = 10;
          }
        par.kernel.n_hp = dim*3;

      }
    else if (simtype == "spsa")
      {
        par.n_iterations = n_iter/2;
        par.use_spsa = 1;
      }
    else if (simtype != "standard")
      {
      	throw std::runtime_error("Unsuported option: "+simtype);
      }

    // Ouput filename
    std::string savefile = output_folder+output_prefix+"_"+exp_criteria+"_x"+sspar.str()+"_" + problem_name + "_rep_" + ss.str() + ".dat";
    strcpy(par.save_filename, savefile.c_str());
    par.load_save_flag = 2;
    
    // Create optimizer
    boost::scoped_ptr<bayesopt::ContinuousModel> optimizer;

    if (problem_name == "exp2d")           optimizer.reset(new Exponential2D(par));
    else if (problem_name == "easom")      optimizer.reset(new Easom(par));
    else if (problem_name == "branin")     optimizer.reset(new BraninNormalized(par));
    else if (problem_name == "camelback")  optimizer.reset(new ExampleCamelback(par));
    else if (problem_name == "rosen2")     optimizer.reset(new Rosenbrock(dim,par));
    else if (problem_name == "schubert")   optimizer.reset(new Schubert(par));
    else if (problem_name == "hart6")      optimizer.reset(new ExampleHartmann6(par));
    else if (problem_name == "ackley2")
      {
        // We do this to access getUpperBound method
        AckleyTest* ackley2 = new AckleyTest(par,dim);
        optimizer.reset(ackley2);
        optimizer->setBoundingBox(-ackley2->getUpperBound(),ackley2->getUpperBound());
      }
    else if (problem_name == "egg")
      {
        // We do this to access getUpperBound method
        Eggholder* egg = new Eggholder(par);
        optimizer.reset(egg);
        optimizer->setBoundingBox(-egg->getUpperBound(), egg->getUpperBound());
      }
    else if (problem_name == "micha")
      {
        optimizer.reset(new ExampleMichalewicz(dim,par));
        zvectord lb(dim);
        svectord ub(dim,M_PI);
        optimizer->setBoundingBox(lb,ub);
      }
    else
      {
	      throw std::runtime_error("Problem undefined!");
      }

                        
    // Set up the optimization
    optimizer->setNormalizeCriteria(use_normalize);
    optimizer->setBoltzmannPolicy(use_boltzmann, temperature);

    optimizer->initializeOptimization();

    for (size_t ii = 0; ii < par.n_iterations; ++ii)
      {
	      optimizer->stepOptimization();
      }
  }

  return 0;
}
