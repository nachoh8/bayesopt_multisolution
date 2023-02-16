/*
-----------------------------------------------------------------------------
   This file is part of BayesOptimization, an efficient C++ library for 
   Bayesian optimization.

   Copyright (C) 2011-2015 Ruben Martinez-Cantin <rmcantin@unizar.es>
 
   BayesOptimization is free software: you can redistribute it and/or modify
   it under the terms of the GNU Affero General Public License as published by
   the Free Software Foundation, either version 3 of the License, or
   (at your option) any later version.

   BayesOptimization is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU Affero General Public License for more details.

   You should have received a copy of the GNU Affero General Public License
   along with BayesOptimization.  If not, see <http://www.gnu.org/licenses/>.
-----------------------------------------------------------------------------
*/
#include <ctime>
#include <fstream>

#include "bayesopt/bayesopt.h"
#include "bayesopt/bayesopt.hpp"      

#include "log.hpp"
#include "ublas_extra.hpp"
#include "specialtypes.hpp"

static const int BAYESOPT_FAILURE = -1; /* generic failure code */
static const int BAYESOPT_INVALID_ARGS = -2;
static const int BAYESOPT_OUT_OF_MEMORY = -3;
static const int BAYESOPT_RUNTIME_ERROR = -4;

/**
 * \brief Version of ContinuousModel for the C wrapper
 */
class CContinuousModel: public bayesopt::ContinuousModel 
{
 public:

  CContinuousModel(size_t dim, bopt_params params):
    ContinuousModel(dim,params)  {}; 

  virtual ~CContinuousModel(){};

  double evaluateSample( const vectord &Xi ) 
  {
    int n = static_cast<int>(Xi.size());
    return  mF(n,&Xi[0],NULL,mOtherData);
  };

  void set_eval_funct(eval_func f)
  {  mF = f; }


  void save_other_data(void* other_data)
  {  mOtherData = other_data; }
 
protected:
  void* mOtherData;
  eval_func mF;
};

/**
 * \brief Version of Parallel ContinuousModel for the C wrapper
 */
class CContinuousParallelModel: public bayesopt::ContinuousModel 
{
 public:

  CContinuousParallelModel(size_t dim, bopt_params params):
    ContinuousModel(dim,params)  {}; 

  virtual ~CContinuousParallelModel(){};

  double evaluateSample( const vectord &Xi ) 
  {
    const size_t ndim = Xi.size();

    vectord yNext(1);
    mF(ndim, 1, &Xi[0], &yNext[0], NULL, mOtherData);
    return  yNext[0];
  };

  vectord evaluateSamples( const vecOfvec &Xs ) 
  {
    size_t nQs = Xs.size();
    size_t ndim = Xs[0].size();
    int n = static_cast<int>(nQs*ndim);

    vectord vQs(n);
    for (size_t i = 0; i < nQs; i++) {
      for (size_t j = 0; j < ndim; j++) {
        vQs[i*ndim+j] = Xs[i][j];
      } 
    }

    vectord yNexts(nQs);
    mF(ndim, nQs, &vQs[0], &yNexts[0], NULL, mOtherData);

    return yNexts;
  };

  void set_eval_funct(multi_eval_func f)
  {  mF = f; }


  void save_other_data(void* other_data)
  {  mOtherData = other_data; }
 
protected:
  void* mOtherData;
  multi_eval_func mF;
};

/**
 * \brief Version of DiscreteModel for the C wrapper
 */
class CDiscreteModel: public bayesopt::DiscreteModel
{
 public:

  CDiscreteModel(const vecOfvec &validX, bopt_params params):
    DiscreteModel(validX, params)
  {}; 

  CDiscreteModel(const vectori &categories, bopt_params params):
    DiscreteModel(categories, params)
  {}; 

  double evaluateSample( const vectord &Xi ) 
  {
    int n = static_cast<int>(Xi.size());
    return  mF(n,&Xi[0],NULL,mOtherData);
  };

  vectord evaluateSamples( const vecOfvec &Xs ) 
  {

    return  zvectord(1);
  };

  void set_eval_funct(eval_func f)
  {  mF = f; }


  void save_other_data(void* other_data)
  {  mOtherData = other_data; }
 
protected:
  void* mOtherData;
  eval_func mF;
};

int bayes_optimization(int nDim, eval_func f, void* f_data,
		       const double *lb, const double *ub,
		       double *x, double *minf, bopt_params parameters)
{
  vectord result(nDim);

  vectord lowerBound = bayesopt::utils::array2vector(lb,nDim); 
  vectord upperBound = bayesopt::utils::array2vector(ub,nDim); 

  try 
    {
      CContinuousModel optimizer(nDim, parameters);

      optimizer.set_eval_funct(f);
      optimizer.save_other_data(f_data);
      optimizer.setBoundingBox(lowerBound,upperBound);

      optimizer.optimize(result);
      std::copy(result.begin(), result.end(), x);

      *minf = optimizer.getValueAtMinimum();
    }
  catch (std::bad_alloc& e)
    {
      FILE_LOG(logERROR) << e.what(); 
      return  BAYESOPT_OUT_OF_MEMORY; 
    }
  catch (std::invalid_argument& e)
    { 
      FILE_LOG(logERROR) << e.what(); 
      return BAYESOPT_INVALID_ARGS; 
    }
  catch (std::runtime_error& e)
    { 
      FILE_LOG(logERROR) << e.what(); 
      return BAYESOPT_RUNTIME_ERROR;
    }
  catch (...)
    { 
      FILE_LOG(logERROR) << "Unknown error";
      return BAYESOPT_FAILURE; 
    }
  return 0; /* everything ok*/
};

int parallel_bayes_optimization(int nDim, multi_eval_func f, void* f_data,
		       const double *lb, const double *ub,
		       double *x, double *minf, bopt_params parameters)
{
  vectord result(nDim);

  vectord lowerBound = bayesopt::utils::array2vector(lb,nDim); 
  vectord upperBound = bayesopt::utils::array2vector(ub,nDim); 

  try 
    {
      CContinuousParallelModel optimizer(nDim, parameters);

      optimizer.set_eval_funct(f);
      optimizer.save_other_data(f_data);
      optimizer.setBoundingBox(lowerBound,upperBound);

      optimizer.optimize(result);
      std::copy(result.begin(), result.end(), x);

      *minf = optimizer.getValueAtMinimum();
    }
  catch (std::bad_alloc& e)
    {
      FILE_LOG(logERROR) << e.what(); 
      return  BAYESOPT_OUT_OF_MEMORY; 
    }
  catch (std::invalid_argument& e)
    { 
      FILE_LOG(logERROR) << e.what(); 
      return BAYESOPT_INVALID_ARGS; 
    }
  catch (std::runtime_error& e)
    { 
      FILE_LOG(logERROR) << e.what(); 
      return BAYESOPT_RUNTIME_ERROR;
    }
  catch (...)
    { 
      FILE_LOG(logERROR) << "Unknown error";
      return BAYESOPT_FAILURE; 
    }
  return 0; /* everything ok*/
};


int parallel_multisolution_bayes_optimization(int nDim, multi_eval_func f, void* f_data,
				      const double *lb, const double *ub,
				      double *x, double *minf, double* solutions, double* values,
				      bopt_params parameters)
{
  vectord result(nDim);

  vectord lowerBound = bayesopt::utils::array2vector(lb,nDim); 
  vectord upperBound = bayesopt::utils::array2vector(ub,nDim); 

  try 
    {
      CContinuousParallelModel optimizer(nDim, parameters);

      optimizer.set_eval_funct(f);
      optimizer.save_other_data(f_data);
      optimizer.setBoundingBox(lowerBound,upperBound);

      optimizer.optimize(result);
      std::copy(result.begin(), result.end(), x);

      *minf = optimizer.getValueAtMinimum();
      
      vecOfvec _solutions;
      vectord _values;
      optimizer.getFinalResults(_solutions, _values);

      vectord _plain_solutions(_solutions.size() * nDim);
      for (size_t i = 0; i < _solutions.size(); i++) {
        for (size_t j = 0; j < nDim; j++) {
          _plain_solutions[i*nDim+j] = _solutions[i][j];
        } 
      }

      std::copy(_plain_solutions.begin(), _plain_solutions.end(), solutions);
      std::copy(_values.begin(), _values.end(), values);
    }
  catch (std::bad_alloc& e)
    {
      FILE_LOG(logERROR) << e.what(); 
      return  BAYESOPT_OUT_OF_MEMORY; 
    }
  catch (std::invalid_argument& e)
    { 
      FILE_LOG(logERROR) << e.what(); 
      return BAYESOPT_INVALID_ARGS; 
    }
  catch (std::runtime_error& e)
    { 
      FILE_LOG(logERROR) << e.what(); 
      return BAYESOPT_RUNTIME_ERROR;
    }
  catch (...)
    { 
      FILE_LOG(logERROR) << "Unknown error";
      return BAYESOPT_FAILURE; 
    }
  return 0; /* everything ok*/
}

int bayes_optimization_log(int nDim, eval_func f, void* f_data,
			   const double *lb, const double *ub,
			   double *x, double *minf, 
			   const char* filename, 
			   bopt_params parameters)
{
  vectord result(nDim);

  vectord lowerBound = bayesopt::utils::array2vector(lb,nDim); 
  vectord upperBound = bayesopt::utils::array2vector(ub,nDim); 

  try 
    {
      std::ofstream log;
      std::ofstream timelog;
      std::string timefilename = "time_";
      timefilename.append(filename);

      log.open(filename);
      timelog.open(timefilename.c_str());


      CContinuousModel optimizer(nDim, parameters);

      optimizer.set_eval_funct(f);
      optimizer.save_other_data(f_data);
      optimizer.setBoundingBox(lowerBound,upperBound);

      std::clock_t curr_t;
      std::clock_t prev_t = clock();

      optimizer.initializeOptimization();

      for (size_t ii = 0; ii < parameters.n_iterations; ++ii)
	{      
	  optimizer.stepOptimization();

	  curr_t = clock();
	  timelog << ii << ","
		  << static_cast<double>(curr_t - prev_t) / CLOCKS_PER_SEC 
		  << std::endl;
	  prev_t = curr_t;

	  // Results
	  result = optimizer.getFinalResult();
	  log << ii << ";";
	  log << optimizer.evaluateSample(result) << ";";
	  log << result << std::endl;
	}

      std::copy(result.begin(), result.end(), x);
      *minf = optimizer.getValueAtMinimum();

      timelog.close();
      log.close();
    }
  catch (std::bad_alloc& e)
    {
      FILE_LOG(logERROR) << e.what(); 
      return  BAYESOPT_OUT_OF_MEMORY; 
    }
  catch (std::invalid_argument& e)
    { 
      FILE_LOG(logERROR) << e.what(); 
      return BAYESOPT_INVALID_ARGS; 
    }
  catch (std::runtime_error& e)
    { 
      FILE_LOG(logERROR) << e.what(); 
      return BAYESOPT_RUNTIME_ERROR;
    }
  catch (...)
    { 
      FILE_LOG(logERROR) << "Unknown error";
      return BAYESOPT_FAILURE; 
    }
  return 0; /* everything ok*/
};


int bayes_optimization_disc(int nDim, eval_func f, void* f_data,
			    double *valid_x, size_t n_points,
			    double *x, double *minf, bopt_params parameters)
{
  vectord result(nDim);
  vectord input(nDim);
  vecOfvec xSet;

  for(size_t i = 0; i<n_points;++i)
    {
      for(int j = 0; j<nDim; ++j)
	{
	 input(j) = valid_x[i*nDim+j]; 
	}
      xSet.push_back(input);
    }

  if(parameters.n_init_samples > n_points)
    {
      parameters.n_init_samples = n_points;
      parameters.n_iterations = 0;
    }

  try
    {
      CDiscreteModel optimizer(xSet,parameters);
      
      optimizer.set_eval_funct(f);
      optimizer.save_other_data(f_data);
      optimizer.optimize(result);

      std::copy(result.begin(), result.end(), x);

      *minf = optimizer.getValueAtMinimum();
    }
  catch (std::bad_alloc& e)
    {
      FILE_LOG(logERROR) << e.what(); 
      return  BAYESOPT_OUT_OF_MEMORY; 
    }
  catch (std::invalid_argument& e)
    { 
      FILE_LOG(logERROR) << e.what(); 
      return BAYESOPT_INVALID_ARGS; 
    }
  catch (std::runtime_error& e)
    { 
      FILE_LOG(logERROR) << e.what(); 
      return BAYESOPT_RUNTIME_ERROR;
    }
  catch (...)
    { 
      FILE_LOG(logERROR) << "Unknown error";
      return BAYESOPT_FAILURE; 
    }

  return 0; /* everything ok*/
}


int bayes_optimization_categorical(int nDim, eval_func f, void* f_data,
				   int *categories, double *x, 
				   double *minf, bopt_params parameters)
{
  vectord result(nDim);
  vectori cat(nDim);

  std::copy(categories,categories+nDim,cat.begin());

  try
    {
      CDiscreteModel optimizer(cat,parameters);
      
      optimizer.set_eval_funct(f);
      optimizer.save_other_data(f_data);
      optimizer.optimize(result);

      std::copy(result.begin(), result.end(), x);

      *minf = optimizer.getValueAtMinimum();
    }
  catch (std::bad_alloc& e)
    {
      FILE_LOG(logERROR) << e.what(); 
      return  BAYESOPT_OUT_OF_MEMORY; 
    }
  catch (std::invalid_argument& e)
    { 
      FILE_LOG(logERROR) << e.what(); 
      return BAYESOPT_INVALID_ARGS; 
    }
  catch (std::runtime_error& e)
    { 
      FILE_LOG(logERROR) << e.what(); 
      return BAYESOPT_RUNTIME_ERROR;
    }
  catch (...)
    { 
      FILE_LOG(logERROR) << "Unknown error";
      return BAYESOPT_FAILURE; 
    }

  return 0; /* everything ok*/
}
