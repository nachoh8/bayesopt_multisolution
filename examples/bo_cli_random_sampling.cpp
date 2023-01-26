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
  int verbose = 1;
  bool active = false;
  double active_coef = 0.1;

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
    par.l_type = L_EMPIRICAL;
    par.n_iter_relearn = 1;
    par.n_parallel_samples = 4;
    par.par_type = PAR_RANDOM;
    par.load_save_flag = 2;
    
    std::stringstream ss;
    ss << std::setw(2) << std::setfill('0') << repetition;
    std::string savefile = "output/random_sampling_" + problem_name + "_rep_" + ss.str() + ".dat";
    strcpy(par.save_filename, savefile.c_str());
    
    // posterior criteria:  
    /*
    set_criteria(&par, "cProd(cEIexp, cEntropySearch)");
        
    par.n_crit_params = 2+4;
    par.crit_params[0] = 1.0; // (ignore, shared parameter with any EI)
    par.crit_params[1] = 0.05; // EI temperature
    par.crit_params[2] = 20; // ES mNumCandidates
    par.crit_params[3] = 100; // ES mNumGPSamples
    par.crit_params[4] = 10; // ES mNumSamplesY
    par.crit_params[5] = 100; // ES mNumTrialPoints
    
    */
      
    set_criteria(&par, "cEI");

    par.random_seed = repetition;

    if (active)
      {
	par.active_hyperparams_coef = active_coef;
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

    // Optimization
    std::ofstream log;
    std::ofstream timelog;

    //Initialization
    boost::scoped_ptr<bayesopt::ContinuousModel> optimizer;

    // TODO(Javier): mcmc criteria sampling, LCB as negative
    //set_criteria(&par, "cLCB");

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
	optimizer->setBoundingBox(-ackley2->getUpperBound(),
			        ackley2->getUpperBound());

      }
    else if (problem_name == "egg")
      {
	// We do this to access getUpperBound method
	Eggholder* egg = new Eggholder(par);
	optimizer.reset(egg);
	optimizer->setBoundingBox(-egg->getUpperBound(),
				  egg->getUpperBound());
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

    std::string name = problem_name;
    if (active)
      {
	std::stringstream ss;
	double foo = active_coef * 1000.0;
	int num = static_cast<int>(active_coef*1000);
	ss << std::setw(3) << std::setfill('0') << num;
	name += "_active"+ss.str();
      }
    // TODO(Javier): mcmc criteria sampling, LCB as negative
    optimizer->setNormalizeCriteria(false);
    name = "random_sampling_" + name;

    log.open(filename(name+"_",simtype,repetition).c_str());
    timelog.open(filename("time_"+name+"_",simtype,repetition).c_str());

    std::clock_t curr_t;
    std::clock_t prev_t = clock();

    optimizer->initializeOptimization();

    for (size_t ii = 0; ii < par.n_iterations; ++ii)
      {
	optimizer->stepOptimization();

	curr_t = clock();
	timelog << ii << ","
		<< static_cast<double>(curr_t - prev_t) / CLOCKS_PER_SEC
		<< std::endl;
	prev_t = curr_t;

	// Results
	vectord result = optimizer->getFinalResult();
	log << ii << ";";
	log << optimizer->evaluateSample(result) << ";";
	log << result << std::endl;
      }

    //optimizer->printOptimal();

    timelog.close();
    log.close();
  }

  return 0;
}
