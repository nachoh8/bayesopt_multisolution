/** \file gaussian_process_ml.hpp
    \brief Gaussian process with ML parameters */
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


#ifndef __GAUSSIAN_PROCESS_ML_HPP__
#define __GAUSSIAN_PROCESS_ML_HPP__

#include "distributions/gauss_distribution.hpp"
#include "surrogates/gaussian_process_hierarchical.hpp"


namespace bayesopt
{
  
  /** \addtogroup NonParametricProcesses */
  /**@{*/

  /**
   * \brief Gaussian process with ML parameters 
   */
  class GaussianProcessML: public HierarchicalGaussianProcess
  {
  public:
    GaussianProcessML(size_t dim, Parameters params, const Dataset* data, 			 
		      MeanModel& mean,
		      randEngine& eng);
    virtual ~GaussianProcessML();

    /** 
     * \brief Function that returns the prediction of the GP for a query point
     * in the hypercube [0,1].
     * 
     * @param query in the hypercube [0,1] to evaluate the Gaussian process
     * @return pointer to the probability distribution.
     */	
    ProbabilityDistribution* prediction(const vectord &query);

  private:

    /** 
     * \brief Computes the negative log likelihood and its gradient of
     * the data. In this case, it is equivalent to the
     * negativeTotalLogLikelihood
     * @return value negative log likelihood
     */
    double negativeLogLikelihood();

    /** Precompute some values of the prediction that do not depends on
     * the query
     */
    void precomputePrediction();

  private:
    vectord mWML;           //!< GP ML parameters
    
    /// Precomputed GP prediction operations
    vectord mAlphaF;
    matrixd mKF, mL2;

    GaussianDistribution* d_;      //!< Predictive distributions
  };

  /**@}*/

} //namespace bayesopt

#endif
