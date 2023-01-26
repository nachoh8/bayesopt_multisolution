/*
-------------------------------------------------------------------------
   This file is part of BayesOpt, an efficient C++ library for
   Bayesian optimization.

   Copyright (C) 2011-2015 Ruben Martinez-Cantin <rmcantin@unizar.es>

   BayesOpt is free software: you can redistribute it and/or modify it
   under the terms of the GNU Affero General Public License as published by
   the Free Software Foundation, either version 3 of the License, or
   (at your option) any later version.

   BayesOpt is distributed in the hope that it will be useful, but
   WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU Affero General Public License for more details.

   You should have received a copy of the GNU Affero General Public License
   along with BayesOpt.  If not, see <http://www.gnu.org/licenses/>.
------------------------------------------------------------------------
*/
#include <iostream>
#include "bayesopt/parameters.h"     // c parameters structs
#include "bayesopt/parameters.hpp"   // c++ parameters classes
#include "ublas_extra.hpp"  // array2vector()

/*-----------------------------------------------------------*/
/* Default parameters                                        */
/*-----------------------------------------------------------*/

/* Nonparametric process "parameters" */
const std::string KERNEL_NAME = "kMaternARD5";
const double KERNEL_THETA    = 1.0;
const double KERNEL_SIGMA    = 10.0;
const std::string MEAN_NAME = "mConst";
const double MEAN_MU         = 1.0;
const double MEAN_SIGMA      = 1000.0;
const double PRIOR_ALPHA     = 1.0;
const double PRIOR_BETA      = 1.0;
const double DEFAULT_SIGMA   = 1.0;
const double DEFAULT_NOISE   = 1e-6;

/* Algorithm parameters */
const size_t DEFAULT_ITERATIONS         = 190;
const size_t DEFAULT_INIT_SAMPLES       = 10;
const size_t DEFAULT_ITERATIONS_RELEARN = 50;
const size_t DEFAULT_PARALLEL_SAMPLES = 1;     /**< Only if sampling is allowed */
const size_t DEFAULT_INNER_EVALUATIONS  = 500; /**< Used per dimmension */
const size_t DEFAULT_FINAL_SAMPLES      = 0;   /**< Only in continuous */
const size_t DEFAULT_BURNOUT_MCMC_PARALLEL = 2500;

/* Logging and Files */
const size_t DEFAULT_VERBOSE           = 1;
const std::string LOG_FILENAME         = "bayesopt.log";
const std::string SAVE_FILENAME        = "bayesopt.dat";
const std::string LOAD_FILENAME        = "bayesopt.dat";

const std::string SURR_NAME            = "sGaussianProcess";
const std::string CRIT_NAME            = "cEI";

parallel_type str2par(const char* name)
{
  if      (!strcmp(name,"PAR_NONE")     || !strcmp(name,"none"))        return PAR_NONE;
  else if (!strcmp(name,"PAR_RANDOM")   || !strcmp(name,"random"))      return PAR_RANDOM;
  else if (!strcmp(name,"PAR_MCMC")     || !strcmp(name,"mcmc"))        return PAR_MCMC;
  else if (!strcmp(name,"PAR_THOMPSON") || !strcmp(name,"thompson"))    return PAR_THOMPSON;
  else if (!strcmp(name,"PAR_REJ_SAMPLING") || !strcmp(name,"rej_sampling"))    return PAR_REJ_SAMPLING;
  else if (!strcmp(name,"PAR_PAR_TEMPERING") || !strcmp(name,"par_tempering"))    return PAR_PAR_TEMPERING;
  else if (!strcmp(name,"PAR_BBO") || !strcmp(name,"par_bbo"))    return PAR_BBO;
  else return PAR_ERROR;
}

const char* par2str(parallel_type name)
{
  switch(name)
    {
    case PAR_NONE:      return "PAR_NONE";
    case PAR_RANDOM:    return "PAR_RANDOM";
    case PAR_MCMC:      return "PAR_MCMC";
    case PAR_THOMPSON:  return "PAR_THOMPSON";
    case PAR_REJ_SAMPLING:  return "PAR_REJ_SAMPLING";
    case PAR_PAR_TEMPERING:  return "PAR_PAR_TEMPERING";
    case PAR_BBO:  return "PAR_BBO";
    case PAR_ERROR:
    default: return "ERROR!";
    }
}

learning_type str2learn(const char* name)
{
  if      (!strcmp(name,"L_FIXED")     || !strcmp(name,"fixed"))     return L_FIXED;
  else if (!strcmp(name,"L_EMPIRICAL") || !strcmp(name,"empirical")) return L_EMPIRICAL;
  else if (!strcmp(name,"L_DISCRETE")  || !strcmp(name,"discrete"))  return L_DISCRETE;
  else if (!strcmp(name,"L_MCMC")      || !strcmp(name,"mcmc"))      return L_MCMC;
  else return L_ERROR;
}

const char* learn2str(learning_type name)
{
  switch(name)
    {
    case L_FIXED:     return "L_FIXED";
    case L_EMPIRICAL: return "L_EMPIRICAL";
    case L_DISCRETE:  return "L_DISCRETE";
    case L_MCMC:      return "L_MCMC";
    case L_ERROR:
    default: return "ERROR!";
    }
}

score_type str2score(const char* name)
{
  if      (!strcmp(name,"SC_MTL")   || !strcmp(name,"mtl"))   return SC_MTL;
  else if (!strcmp(name,"SC_ML")    || !strcmp(name,"ml"))    return SC_ML;
  else if (!strcmp(name,"SC_MAP")   || !strcmp(name,"map"))   return SC_MAP;
  else if (!strcmp(name,"SC_LOOCV") || !strcmp(name,"loocv")) return SC_LOOCV;
  else return SC_ERROR;
}

const char* score2str(score_type name)
{
  switch(name)
    {
    case SC_MTL:   return "SC_MTL";
    case SC_ML:    return "SC_ML";
    case SC_MAP:   return "SC_MAP";
    case SC_LOOCV: return "SC_LOOCV";
    case SC_ERROR:
    default: return "ERROR!";
    }
}

void set_kernel(bopt_params* params, const char* name)
{
  strcpy(params->kernel.name, name);
};

void set_mean(bopt_params* params, const char* name)
{
  strcpy(params->mean.name, name);
};

void set_criteria(bopt_params* params, const char* name)
{
  strcpy(params->crit_name, name);
};

void set_surrogate(bopt_params* params, const char* name)
{
  strcpy(params->surr_name, name);
};

void set_log_file(bopt_params* params, const char* name)
{
  strcpy(params->log_filename, name);
};

void set_load_file(bopt_params* params, const char* name)
{
  strcpy(params->load_filename, name);
};

void set_save_file(bopt_params* params, const char* name)
{
  strcpy(params->save_filename, name);
};

void set_parallel(bopt_params* params, const char* name)
{
  params->par_type = str2par(name);
};

void set_learning(bopt_params* params, const char* name)
{
  params->l_type = str2learn(name);
};

void set_score(bopt_params* params, const char* name)
{
  params->sc_type = str2score(name);
};

bopt_params initialize_parameters_to_default(void)
{
  //bayesopt::Parameters par;
  //return par.generate_bopt_params();
  kernel_parameters kernel;
  kernel.name = new char[128];
  strcpy(kernel.name,KERNEL_NAME.c_str());

  kernel.hp_mean[0] = KERNEL_THETA;
  kernel.hp_std[0]  = KERNEL_SIGMA;
  kernel.n_hp       = 1;

  kernel.domain_center[0][0] = 0.5;
  kernel.domain_std[0][0]    = 0.5;
  kernel.n_basis_kernels     = 1;
  kernel.n_dim_domain        = 1;

  mean_parameters mean;
  mean.name = new char[128];
  strcpy(mean.name,MEAN_NAME.c_str());

  mean.coef_mean[0] = MEAN_MU;
  mean.coef_std[0]  = MEAN_SIGMA;
  mean.n_coef       = 1;


  bopt_params params;

  params.n_iterations       = DEFAULT_ITERATIONS;
  params.n_inner_iterations = DEFAULT_INNER_EVALUATIONS;
  params.n_init_samples     = DEFAULT_INIT_SAMPLES;
  params.n_iter_relearn     = DEFAULT_ITERATIONS_RELEARN;
  params.n_final_samples    = DEFAULT_FINAL_SAMPLES;
  params.n_parallel_samples = DEFAULT_PARALLEL_SAMPLES;
  params.burnout_mcmc_parallel = DEFAULT_BURNOUT_MCMC_PARALLEL;

  params.init_method      =  1;
  params.random_seed      = -1;

  params.verbose_level = DEFAULT_VERBOSE;
  params.log_filename  = new char[128];
  strcpy(params.log_filename,LOG_FILENAME.c_str());

  params.load_save_flag = 0;
  params.load_filename = new char[128];
  strcpy(params.load_filename,LOAD_FILENAME.c_str());
  params.save_filename = new char[128];
  strcpy(params.save_filename,SAVE_FILENAME.c_str());

  params.surr_name = new char[128];
  //  strcpy(params.surr_name,"sStudentTProcessNIG");
  strcpy(params.surr_name,SURR_NAME.c_str());

  params.sigma_s = DEFAULT_SIGMA;
  params.noise   = DEFAULT_NOISE;
  params.alpha   = PRIOR_ALPHA;
  params.beta    = PRIOR_BETA;

  params.fixed_dataset_size = 0;
  params.num_solutions = 1;
  params.diversity_level = 0.0;

  params.l_all   = 0;
  params.l_type  = L_EMPIRICAL;
  params.sc_type = SC_MAP;
  params.par_type = PAR_NONE;

  params.use_spsa = 0;
  params.epsilon = 0.0;
  params.force_jump = 20;
  params.active_hyperparams_coef = 0.0; //0.01;

  params.crit_name = new char[128];
  strcpy(params.crit_name,CRIT_NAME.c_str());
  params.n_crit_params = 0;

  // Following the recomendations of Rudolph van der Merwe's thesis
  params.unscented_alpha = 1;
  params.unscented_beta = 2;
  params.unscented_kappa = 0;
  params.input_noise_variance = -1;
  params.use_unscented_criterion = 0;

  params.kernel = kernel;
  params.mean = mean;

  return params;
}

/**
 * Namespace of the library interface
 */
namespace bayesopt {
    /*
     * KernelParameters Class
     */
  KernelParameters::KernelParameters():
    hp_mean(1), hp_std(1), domain_center(1, vectord(1)), domain_std(1, vectord(1))
  {
    // Set default values
    name = KERNEL_NAME;
    hp_mean(0) = KERNEL_THETA;
    hp_std(0) = KERNEL_SIGMA;
    domain_center[0][0] = 0.5;
    domain_std[0][0] = 0.5;
  }

  /*
   * MeanParameters Class
   */
  MeanParameters::MeanParameters():
    coef_mean(1), coef_std(1)
  {
    // Set default values
    name = MEAN_NAME;
    coef_mean(0) = MEAN_MU;
    coef_std(0) = MEAN_SIGMA;
  }

  /*
   * Parameters Class
   */
  Parameters::Parameters():
    kernel(), mean(), crit_params()
  {
    // Set default values
    bopt_params c_params = initialize_parameters_to_default();
    init(c_params);
  }

  Parameters::Parameters(bopt_params c_params):
    kernel(), mean(), crit_params()
  {
    init(c_params);    
  }
  void Parameters::init(bopt_params c_params)
  {
    n_iterations = c_params.n_iterations;
    n_inner_iterations = c_params.n_inner_iterations;
    n_init_samples = c_params.n_init_samples;
    n_iter_relearn = c_params.n_iter_relearn;
    n_final_samples = c_params.n_final_samples;

    n_parallel_samples = c_params.n_parallel_samples;
    burnout_mcmc_parallel = c_params.burnout_mcmc_parallel;
        
    init_method = c_params.init_method;
    random_seed = c_params.random_seed;

    verbose_level = c_params.verbose_level;

    log_filename = std::string(c_params.log_filename);
    load_save_flag = c_params.load_save_flag;
    load_filename = std::string(c_params.load_filename);
    save_filename = std::string(c_params.save_filename);

    surr_name = std::string(c_params.surr_name);
    sigma_s = c_params.sigma_s;

    noise = c_params.noise;
    alpha = c_params.alpha;
    beta = c_params.beta;

    fixed_dataset_size = c_params.fixed_dataset_size;
    
    num_solutions = c_params.num_solutions;
    diversity_level = c_params.diversity_level;

    sc_type = c_params.sc_type;
    l_type = c_params.l_type;
    par_type = c_params.par_type;

    l_all = c_params.l_all;

    use_spsa = c_params.use_spsa;
    epsilon = c_params.epsilon;
    force_jump = c_params.force_jump;

    active_hyperparams_coef = c_params.active_hyperparams_coef;

    kernel.name = std::string(c_params.kernel.name);
    kernel.hp_mean = bayesopt::utils::array2vector(c_params.kernel.hp_mean,
                                                   c_params.kernel.n_hp);
    kernel.hp_std = bayesopt::utils::array2vector(c_params.kernel.hp_std,
                                                  c_params.kernel.n_hp);

    kernel.domain_center = vecOfvec(c_params.kernel.n_basis_kernels, 
                                    vectord(c_params.kernel.n_dim_domain));
    kernel.domain_std = vecOfvec(c_params.kernel.n_basis_kernels, 
                                 vectord(c_params.kernel.n_dim_domain));
    for(size_t i=0; i<kernel.domain_center.size(); i++){
      for(size_t j=0; j<kernel.domain_center[0].size(); j++){
        kernel.domain_center[i][j] = c_params.kernel.domain_center[i][j];
        kernel.domain_std[i][j] = c_params.kernel.domain_std[i][j];
      }
    }

    mean.name = std::string(c_params.mean.name);
    mean.coef_mean = bayesopt::utils::array2vector(c_params.mean.coef_mean,
                                                   c_params.mean.n_coef);
    mean.coef_std = bayesopt::utils::array2vector(c_params.mean.coef_std,
                                                  c_params.mean.n_coef);

    crit_name = c_params.crit_name;
    crit_params = bayesopt::utils::array2vector(c_params.crit_params,
                                                c_params.n_crit_params);

    // Following the recomendations of Rudolph van der Merwe's thesis
    unscented_alpha = c_params.unscented_alpha;
    unscented_beta  = c_params.unscented_beta;
    unscented_kappa = c_params.unscented_kappa;
    input_noise_variance = c_params.input_noise_variance;
    use_unscented_criterion = c_params.use_unscented_criterion;

  }

  bopt_params Parameters::generate_bopt_params()
  {
    bopt_params c_params = initialize_parameters_to_default();
    c_params.n_iterations = n_iterations;
    c_params.n_inner_iterations = n_inner_iterations;
    c_params.n_init_samples = n_init_samples;
    c_params.n_iter_relearn = n_iter_relearn;
    c_params.n_final_samples = n_final_samples;

    c_params.n_parallel_samples = n_parallel_samples;
    c_params.burnout_mcmc_parallel = burnout_mcmc_parallel;
      
    c_params.par_type = par_type;

    c_params.init_method = init_method;
    c_params.random_seed = random_seed;

    c_params.verbose_level = verbose_level;

    strcpy(c_params.log_filename, log_filename.c_str());
    c_params.load_save_flag = load_save_flag;
    strcpy(c_params.load_filename, load_filename.c_str());
    strcpy(c_params.save_filename, save_filename.c_str());

    strcpy(c_params.surr_name, surr_name.c_str());
    c_params.sigma_s = sigma_s;

    c_params.noise = noise;
    c_params.alpha = alpha;
    c_params.beta = beta;

    c_params.fixed_dataset_size = fixed_dataset_size;

    c_params.num_solutions = num_solutions;
    c_params.diversity_level = diversity_level;

    c_params.sc_type = sc_type;

    c_params.l_type = l_type;

    c_params.l_all = l_all;

    c_params.use_spsa = use_spsa;
    c_params.epsilon = epsilon;
    c_params.force_jump = force_jump;

    c_params.active_hyperparams_coef = active_hyperparams_coef;

    strcpy(c_params.kernel.name, kernel.name.c_str());
    //TODO (Javier): Should it be necessary to check size?
    for(size_t i=0; i<kernel.hp_mean.size(); i++){
      c_params.kernel.hp_mean[i] = kernel.hp_mean[i];
    }
    for(size_t i=0; i<kernel.hp_std.size(); i++){
      c_params.kernel.hp_std[i] = kernel.hp_std[i];
    }
    c_params.kernel.n_hp = kernel.hp_std.size();
    
    c_params.kernel.n_basis_kernels = kernel.domain_center.size();
    if(c_params.kernel.n_basis_kernels > 0){
      c_params.kernel.n_dim_domain = kernel.domain_center[0].size();
    }
    else{
      c_params.kernel.n_dim_domain = 0;
    }


    for(size_t i=0; i<kernel.domain_center.size(); i++){
      for(size_t j=0; j<kernel.domain_center[0].size(); j++){
        c_params.kernel.domain_center[i][j] = kernel.domain_center[i][j];
        c_params.kernel.domain_std[i][j] = kernel.domain_std[i][j];
      }
    }


    strcpy(c_params.mean.name, mean.name.c_str());
    for(size_t i=0; i<mean.coef_mean.size(); i++){
      c_params.mean.coef_mean[i] = mean.coef_mean[i];
    }
    for(size_t i=0; i<mean.coef_std.size(); i++){
      c_params.mean.coef_std[i] = mean.coef_std[i];
    }
    c_params.mean.n_coef = mean.coef_std.size();

    strcpy(c_params.crit_name, crit_name.c_str());
    for(size_t i=0; i<crit_params.size(); i++){
      c_params.crit_params[i] = crit_params[i];
    }
    c_params.n_crit_params = crit_params.size();

    // Following the recomendations of Rudolph van der Merwe's thesis
    c_params.unscented_alpha = unscented_alpha;
    c_params.unscented_beta  = unscented_beta;
    c_params.unscented_kappa = unscented_kappa;
    c_params.input_noise_variance = input_noise_variance;
    c_params.use_unscented_criterion = use_unscented_criterion;

    return c_params;
  }

  void Parameters::bostrdup (char* d, const char *s) {
    // Warning. This memory is not freed!
    d = new char[strlen (s) + 1];   // Space for length plus nul
    if (d == NULL) d = NULL;          // No memory
    strcpy (d,s);                        // Copy the characters
  }
    
  void Parameters::set_parallel(std::string name){
    par_type = str2par(name.c_str());
  }
    
  std::string Parameters::get_parallel(){
    return std::string(par2str(par_type));
  }

  void Parameters::set_learning(std::string name){
    l_type = str2learn(name.c_str());
  }
  std::string Parameters::get_learning(){
    return std::string(learn2str(l_type));
  }

  void Parameters::set_score(std::string name){
    sc_type = str2score(name.c_str());
  }
  std::string Parameters::get_score(){
    return std::string(score2str(sc_type));
  }

}//namespace bayesopt
