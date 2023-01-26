/**  \file criteria_lp.hpp \brief Local Penalization based criteria */
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

#ifndef  _CRITERIA_LP_HPP_
#define  _CRITERIA_LP_HPP_

#include <memory>
#include <string>
#include <iostream> // TODO: remove
#include <cmath>
#include <boost/numeric/ublas/assignment.hpp>
#include <boost/numeric/ublas/vector_proxy.hpp>
#include <boost/math/distributions/normal.hpp>

#include "criteria/criteria_functors.hpp"
#include "lhs.hpp"
#include "inneroptimization.hpp"

namespace ublas = boost::numeric::ublas;

namespace bayesopt
{
  class EstimateLCallback: public RBOptimizable
  {
  public:
    explicit EstimateLCallback(NonParametricProcess *mProc, randEngine* mtRandom)
      :mProc(mProc), mtRandom(mtRandom) {};
    
    double evaluate(const vectord &query)
    {
      return aprox_L(query);
    }

    /**
     * @brief Estimate the Lipschitz constant of f by taking maximizing the norm of the expectation of the gradient of *f*.
     * 
     * @return double 
     */
    double estimateL(const size_t nDim) {
      if (! cOptimizer) {
        cOptimizer.reset(new NLOPT_Optimization(this, nDim));
        cOptimizer->setAlgorithm(BOBYQA);
        cOptimizer->setMaxEvals(NUM_SAMPLES*2);
      }

      double minG = DBL_MAX;
      vectord x0(nDim);
      vecOfvec Xmodel = mProc->getData()->mX;
      for (size_t i = 0; i < Xmodel.size(); i++) {
        vectord v(nDim);
        std::copy(Xmodel[i].begin(), Xmodel[i].end(), v.begin());

        double gPred = aprox_L(v);
        if (gPred < minG) {
          minG = gPred;
          x0 = vectord(v);
        }
      }

      matrixd samples(NUM_SAMPLES, nDim);
      utils::uniformSampling(samples, *mtRandom);
      for (size_t i = 0; i < NUM_SAMPLES; i++) {
        vectord v(nDim);
        ublas::matrix_row<matrixd> mr (samples, i); 
        std::copy(mr.begin(), mr.end(), v.begin());

        double gPred = aprox_L(v);
        if (gPred < minG) {
          minG = gPred;
          x0 = vectord(v);
        }
      }
      
      double L = cOptimizer->run(x0);

      L = -L;
      if (L < 1e-7) { // avoid problems in cases in which the model is flat.
        return 10;
      }

      return L;
    }

  private:
    double aprox_L(const vectord& x) {
      const int d = x.size();
      vectord g(d); // gradient
      for (int i = 0; i < d; i++) {
        vectord e(d, 0.0);
        e[i] = epsilon;

        double vx_plus = mProc->prediction(x + e)->mean();
        double vx_minus = mProc->prediction(x - e)->mean();

        g[i] = (vx_plus - vx_minus) / (2.0 * epsilon);
      }

      double L = ublas::norm_2(g);
      return -L; // to minimize
    }

    const int NUM_SAMPLES = 200;

    const double epsilon = 1e-7;

    NonParametricProcess* mProc;
    randEngine* mtRandom;
    boost::scoped_ptr<NLOPT_Optimization> cOptimizer;
  };

  /**\addtogroup CriteriaFunctions  */
  //@{

  /// Local Penalization criterion by \cite [Javier González et al., 2015]
  class LocalPenalization : public Criteria
  {
  public:
    virtual ~LocalPenalization(){};
    void init(NonParametricProcess* proc)
    { 
      mProc = proc;
    };

    void setParameters(const vectord &params)
    {
      using boost::numeric::ublas::subrange;

      size_t totalParams = nParameters();
      size_t nParams = params.size();
      if (totalParams != nParams) {
        std::string msg = "Wrong number of criteria parameters: " + std::to_string(totalParams) + " received " + std::to_string(nParams);
        FILE_LOG(logERROR) << msg;
        throw std::invalid_argument(msg);
      }

      int batchSize = static_cast<int>(params(0));
      if (batchSize < 2) {
        std::string msg = "Batch size must be at least 2";
        FILE_LOG(logERROR) << msg;
        throw std::invalid_argument(msg);
      }

      int transform = static_cast<int>(params(1));
      if (transform < 0 || transform > 1) {
        std::string msg = "Transform function must be {0: None, 1: Softplus}";
        FILE_LOG(logERROR) << msg;
        throw std::invalid_argument(msg);
      }

      mCriteria->setParameters(subrange(params, 2, nParams));

      X_batch = vecOfvec(batchSize - 1);
        
      r = vectord(X_batch.size());
      s = vectord(X_batch.size());

      transformation = transform;

      mLCallback.reset(new EstimateLCallback(mProc, mtRandom));
    };

    size_t nParameters() {return 2 + mCriteria->nParameters();};

    virtual void setRandomEngine(randEngine& eng){
      mtRandom = &eng;
      mCriteria->setRandomEngine(eng);
    }

    void pushCriteria(Criteria* crit)
    {
      if (!mCriteria) {
        mCriteria.reset(crit);
      } else {
        FILE_LOG(logERROR) << "Criteria has already been set"; 
	      throw std::invalid_argument("Criteria has already been set");
      }
    };

    double operator() (const vectord &x) 
    { 
      return penalized_acquisition(x);
    };

    virtual void reset() {
      reset_batch();
      mCriteria->reset();
    }

    virtual void update(const vectord &x) {
      if (x.size() == 0) {
        reset_batch();
      } else if (current_point < X_batch.size()) {
        update_batch(x);
      } else {
        mCriteria->update(x);
      }
    }

    /**
     * @brief reset batch
     * 
     */
    void reset_batch() {
      FILE_LOG(logDEBUG) << "Reset batch";
      current_point = 0;
      min = 0.0;
      L = 0.0;
    };

    /**
     * @brief add point x to batch
     * 
     * @param x 
     */
    void update_batch(const vectord &x) {
      FILE_LOG(logDEBUG) << "Adding point (" << (current_point+1) << ") " << x << " to batch";
      if (current_point == 0) {
        L = mLCallback->estimateL(x.size());
        min = mProc->getValueAtMinimum();
        FILE_LOG(logDEBUG) << "LP: " << L;
        FILE_LOG(logDEBUG) << "Min: " << min;
      }

      X_batch[current_point++] = x;

      for (int i = 0; i < current_point; i++) {
        ProbabilityDistribution* prob = mProc->prediction(X_batch[i]);
        double std = prob->std(), mean = prob->mean();

        if (std < 1e-16) {
            std = 1e-16;
        }
        s[i] = sqrt(std) / L;
        r[i] = (mean - min) / L;
        FILE_LOG(logDEBUG) << "(" << (i+1) << ") " << X_batch[i] << " -> R: " << r[i] << " S: " << s[i];
      }
    };

    std::string name() {return "cLP";};

  private:
    /**
     * @brief Defines the exclusion zones from batch to point x
     * 
     * @param x query point
     * @return vector of penalization balls
     */
    vectord hammer(const vectord& x) {
      vectord res(current_point);

      for (int i = 0; i < current_point; i++) {
          vectord aux = x - X_batch[i];
          aux = ublas::element_prod(aux, aux);
          
          double v = ublas::sum(aux);
          v = (sqrt(v) - r[i]) / s[i];
          
          res[i] = std::log(boost::math::cdf(boost::math::normal(), v));          
      }

      return res;
    }

    /**
     * @brief Creates a penalized acquisition function using 'hammer' functions around the points collected in the batch
     * 
     * @param x query point
     * @return double 
     */
    double penalized_acquisition(const vectord &x) {
      double fval = -mCriteria->evaluate(x);

      switch (transformation) // map to log natural space
      {
      case 1: // Softplus
          if (fval >= 40.0) {
            fval = std::log(fval); // g(z) = ln(z)
          } else {
            fval = std::log(std::log1p(std::exp(fval))); // g(z) = ln(ln(1+e^z))
          }
          break;
      
      default: // None
          fval = std::log(fval + 1e-50); // g(z) = ln(z)
          break;
      }

      fval = -fval;
      
      if (current_point > 0) {
        vectord exclusion = hammer(x);
        FILE_LOG(logDEBUG) << x << " -> exclusion: " << exclusion;
        fval += ublas::sum(exclusion) * -1.0;
      }

      return fval;
    }

    std::unique_ptr<Criteria> mCriteria;

    vecOfvec X_batch;
    int current_point = 0;
    double min = 0.0, L = 0.0;

    int transformation;
    vectord r, s;

    boost::scoped_ptr<EstimateLCallback> mLCallback;
  };

  //@}

} //namespace bayesopt


#endif
