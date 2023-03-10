/**  \file parameters.h \brief Parameter definitions. */
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


#ifndef __BOPT_PARAMETERS_H__
#define __BOPT_PARAMETERS_H__

#include <string.h>

/* WINDOWS DLLs stuff */
#if defined (BAYESOPT_DLL) && (defined(_WIN32) || defined(__WIN32__)) && !defined(__LCC__)
  #if defined(bayesopt_EXPORTS)
    #define  BAYESOPT_API __declspec(dllexport)
  #else
    #define  BAYESOPT_API __declspec(dllimport)
  #endif /* MyLibrary_EXPORTS */
#else /* defined (_WIN32) */
 #define BAYESOPT_API
#endif

#ifdef __cplusplus
extern "C" {
#endif

  /*-----------------------------------------------------------*/
  /* Configuration parameters                                  */
  /*-----------------------------------------------------------*/
  typedef enum {
    PAR_NONE,
    PAR_RANDOM,
    PAR_MCMC,
    PAR_THOMPSON,
    PAR_REJ_SAMPLING,
    PAR_PAR_TEMPERING,
    PAR_BBO,
    PAR_ERROR = -1
  } parallel_type;
  
  typedef enum {
    L_FIXED,
    L_EMPIRICAL,
    L_DISCRETE,
    L_MCMC,
    L_ERROR = -1
  } learning_type;

  typedef enum {
    SC_MTL,
    SC_ML,
    SC_MAP,
    SC_LOOCV,
    SC_ERROR = -1
  } score_type;


  /** Kernel configuration parameters */
  typedef struct {
    char*  name;                 /**< Name of the kernel function */
    double hp_mean[128];         /**< Kernel hyperparameters prior (mean, log space) */
    double hp_std[128];          /**< Kernel hyperparameters prior (st dev, log space) */
    size_t n_hp;                 /**< Number of kernel hyperparameters */
        /* Domain of kernel for nonstationary kernels (linear
       combination). The domain is represented as a independent Normal
       distribution. */
    double domain_center[128][128]; /**< Center of kernel domain for nonst. kernels */
    double domain_std[128][128];    /**< Std of kernel domain for nonst.kernels */
    size_t n_basis_kernels;         /**< Number of stationary kernels that are 
				         combined. 1st index of domain_* arrays */
    size_t n_dim_domain;            /**< Number of domain terms. It should be 
				         the number of problem dimmensions. 
				         2nd index of domain_* arrays . */
  } kernel_parameters;

  typedef struct {
    char* name;                  /**< Name of the mean function */
    double coef_mean[128];       /**< Basis function coefficients (mean) */
    double coef_std[128];        /**< Basis function coefficients (std) */
    size_t n_coef;               /**< Number of mean funct. hyperparameters */
  } mean_parameters;

  /** \brief Configuration parameters
   *  @see \ref reference for a full description of the parameters
   */
  typedef struct {
    size_t n_iterations;         /**< Maximum BayesOpt evaluations (budget) */
    size_t n_inner_iterations;   /**< Maximum inner optimizer evaluations */
    size_t n_init_samples;       /**< Number of samples before optimization */
    size_t n_iter_relearn;       /**< Number of samples before relearn kernel */
    size_t n_final_samples;      /**< Number of samples after optimization (only continuous).
				      Refines the solution with a local optimization */
    size_t n_parallel_samples;   /**< Number of samples (only if allowed) */

    /** Sampling method for initial set 1-LHS, 2-Sobol (if available),
     *  other value-uniformly distributed */
    size_t init_method;
    int random_seed;             /**< >=0 -> Fixed seed, <0 -> Time based (variable). */

    int verbose_level;           /**< <=0-Error,0-Warning,1-Info,2-Debug -> stdout
				      >=6-Error,3-Warning,4-Info,5-Debug -> logfile*/
    char* log_filename;          /**< Log file path (if applicable) */

    size_t load_save_flag;       /**< 1-Load data,2-Save data,
				      3-Load and save data. */
    char* load_filename;          /**< Init data file path (if applicable) */
    char* save_filename;          /**< Sava data file path (if applicable) */

    char* surr_name;             /**< Name of the surrogate function */
    double sigma_s;              /**< Signal variance (if known).
				    Used in GaussianProcess and GaussianProcessNormal */
    double noise;                /**< Variance of observation noise (and nugget) */

    double alpha;                /**< Inverse Gamma prior for signal var.
				    Used in StudentTProcessNIG */
    double beta;                 /**< Inverse Gamma prior for signal var.
				    Used in StudentTProcessNIG */

    size_t fixed_dataset_size; // maximum size of the dataset, 0 to disable

    size_t num_solutions; // number of diverse solutions
    double diversity_level; // minimum diversity level between solutions

    score_type sc_type;          /**< Score type for kernel hyperparameters (ML,MAP,etc) */
    learning_type l_type;        /**< Type of learning for the kernel params */
    parallel_type par_type;      /**< Type of parallelization used  */
    size_t burnout_mcmc_parallel; /**< Number of burnout on mcmc parallel sampling per dimension */

    int l_all;                   /**< Learn all hyperparameters or only kernel */

    size_t use_spsa;             /**< Using SPSA trick to approximate gradients */ 
    double epsilon;              /**< For epsilon-greedy exploration */
    size_t force_jump;           /**< If >0, and the difference between two
				    consecutive observations is pure noise,
				    for n consecutive steps, force a random
				    jump. Avoid getting stuck if model is bad
				    and there is few data, however, it might
				    reduce the accuracy. */
    double active_hyperparams_coef; /**< Coefficient for active
				       hyperparameter learning */

    kernel_parameters kernel;    /**< Kernel parameters */
    mean_parameters mean;        /**< Mean (parametric function) parameters */

    char* crit_name;             /**< Name of the criterion */
    double crit_params[128];     /**< Criterion hyperparameters (if needed) */
    size_t n_crit_params;        /**< Number of criterion hyperparameters */

    // Following the recomendations of Rudolph van der Merwe's thesis
    double unscented_alpha;
    double unscented_beta;
    double unscented_kappa;
    double input_noise_variance;
    size_t use_unscented_criterion;

  } bopt_params;

  /*-----------------------------------------------------------*/
  /* These functions are added to simplify wrapping code       */
  /*-----------------------------------------------------------*/
  BAYESOPT_API parallel_type str2par(const char* name);
  BAYESOPT_API const char* par2str(parallel_type name);
  
  BAYESOPT_API learning_type str2learn(const char* name);
  BAYESOPT_API const char* learn2str(learning_type name);

  BAYESOPT_API score_type str2score(const char* name);
  BAYESOPT_API const char* score2str(score_type name);

  BAYESOPT_API void set_kernel(bopt_params* params, const char* name);
  BAYESOPT_API void set_mean(bopt_params* params, const char* name);
  BAYESOPT_API void set_criteria(bopt_params* params, const char* name);
  BAYESOPT_API void set_surrogate(bopt_params* params, const char* name);
  BAYESOPT_API void set_log_file(bopt_params* params, const char* name);
  BAYESOPT_API void set_load_file(bopt_params* params, const char* name);
  BAYESOPT_API void set_save_file(bopt_params* params, const char* name);
  BAYESOPT_API void set_learning(bopt_params* params, const char* name);
  BAYESOPT_API void set_score(bopt_params* params, const char* name);
  BAYESOPT_API void set_parallel(bopt_params* params, const char* name);

  BAYESOPT_API bopt_params initialize_parameters_to_default(void);

#ifdef __cplusplus
}
#endif


#endif
