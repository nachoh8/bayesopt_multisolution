/**  \file criteria_ei.hpp \brief Expected improvement based criteria */
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

#ifndef  _CRITERIA_EI_HPP_
#define  _CRITERIA_EI_HPP_

#include <boost/numeric/ublas/vector_expression.hpp>
#include <boost/numeric/ublas/vector_proxy.hpp>

#include "criteria/criteria_functors.hpp"
#include "distributions/prob_distribution.hpp"

namespace bayesopt
{

  /**\addtogroup CriteriaFunctions  */
  //@{

  /// Expected improvement criterion by Mockus \cite Mockus78
  class ExpectedImprovement: public Criteria
  {
  public:
    virtual ~ExpectedImprovement(){};
    void init(NonParametricProcess* proc)
    { 
      mProc = proc;
      mExp = 1;
    };

    void setParameters(const vectord &params)
    { mExp = static_cast<size_t>(params(0)); };

    size_t nParameters() {return 1;};

    double operator() (const vectord &x) 
    { 
      const double min = mProc->getValueAtMinimum();
      return mProc->prediction(x)->negativeExpectedImprovement(min,mExp); 
    };

    std::string name() {return "cEI";};

  private:
    size_t mExp;
  };

  class ClusterEI: public Criteria
  {
  public:
    virtual ~ClusterEI(){};
    void init(NonParametricProcess* proc)
    { 
      mProc = proc;
      mExp = 1;
    };

    void setParameters(const vectord &params)
    { mExp = static_cast<size_t>(params(0)); };

    size_t nParameters() {return 1;};

    double operator() (const vectord &x) 
    {
      double min_d = 9999999999999;
      size_t c_idx;
      for (size_t i = 0; i < centroids.size(); i++) {
        double d = norm_2(x - centroids[i]);
        if (d < min_d) {
          min_d = d;
          c_idx = i;
        }
      }

      double d_factor = c_idx != current_cluster ? min_d / norm_2(x - centroids[current_cluster]) : 1.0;

      const double min = cluster_min[current_cluster];
      return mProc->prediction(x)->negativeExpectedImprovement(min,mExp) * d_factor;
    };

    std::string name() {return "cClusterEI";};

    virtual void update(const vectord &x) {
      if (x.size() == 0) {
        FILE_LOG(logDEBUG) << "ClusterEI: from active cluster " << current_cluster+1 << " to " << current_cluster+2;
        current_cluster++;
      } else { // clusters info -> num_clusters, ndim, [centroid(flatten), cluster min]+
        using boost::numeric::ublas::subrange;
        
        const int num_clusters = int(x[0]);
        const int ndim = int(x[1]);
        FILE_LOG(logDEBUG) << "ClusterEI: setting clusters, num clusters: " << num_clusters;
        
        centroids = vecOfvec(num_clusters);
        cluster_min = vectord(num_clusters);

        int s = 2, c = 0;
        while (s < x.size()) {
          int end = s + ndim;
          vectord centroid = subrange(x, s, end);
          double ymin = x[end];

          centroids[c] = centroid;
          cluster_min[c] = ymin;
          c++;

          s = end + 1;
        }

        current_cluster = 0;

        FILE_LOG(logDEBUG) << "ClusterEI: set clusters ok - " << c;
      }
    }

  private:
    size_t mExp;

    size_t current_cluster;
    vecOfvec centroids;
    vectord cluster_min;
  };

  /// Expected improvement criterion modification by Lizotte
  class BiasedExpectedImprovement: public Criteria
  {
  public:
    virtual ~BiasedExpectedImprovement(){};
    void init(NonParametricProcess* proc)
    { 
      mProc = proc;
      mBias = 0.01;
      mExp = 1;
    };

    void setParameters(const vectord &params)
    {
      mExp = static_cast<size_t>(params(0));
      mBias = params(1);
    };

    size_t nParameters() {return 2;};

    double operator() (const vectord &x) 
    { 
      const double sigma = mProc->getSignalVariance();
      const double min = mProc->getValueAtMinimum() - mBias/sigma;
      return mProc->prediction(x)->negativeExpectedImprovement(min,mExp); 
    };
    std::string name() {return "cBEI";};
  private:
    double mBias;
    size_t mExp;
  };


  /// Expected improvement criterion using Schonlau annealing. \cite Schonlau98
  class AnnealedExpectedImprovement: public Criteria
  {
  public:
    virtual ~AnnealedExpectedImprovement(){};
    void init(NonParametricProcess* proc)
    { 
      mProc = proc;
      reset();
    };

    void setParameters(const vectord &params)
    { mExp = static_cast<size_t>(params(0)); };

    size_t nParameters() {return 1;};
    void reset() { nCalls = 1; mExp = 8;};
    double operator() (const vectord &x) 
    {
      ProbabilityDistribution* d_ = mProc->prediction(x);
      const double min = mProc->getValueAtMinimum();
      return d_->negativeExpectedImprovement(min,mExp); 
    };
    void update(const vectord &x) 
    {
      ++nCalls;
      if ((nCalls % 20) &&  (mExp > 1))
	{
	  mExp = mExp/2;
	}
    }
    std::string name() {return "cEIa";};
  private:
    size_t mExp;
    unsigned int nCalls;
  };

  class ExponentialExpectedImprovement: public Criteria
  {
  public:
    virtual ~ExponentialExpectedImprovement(){};
    void init(NonParametricProcess* proc)
    { 
      mProc = proc;
      mExp = 1;
      mTemperature = 0.4;
    };

    void setParameters(const vectord &params)
    { 
      mExp = static_cast<size_t>(params(0)); 
      mTemperature = params(1);
    };

    size_t nParameters() {return 2;};

    double operator() (const vectord &x) 
    { 
      const double min = mProc->getValueAtMinimum();
      double value = mProc->prediction(x)->negativeExpectedImprovement(min,mExp); 

      // Normalize temperature
      double maxY = mProc->getData()->getValueAtMaximum();
      double minY = mProc->getData()->getValueAtMinimum();
      double temp = mTemperature * (maxY - minY);

      // Apply exponential
      double exp_value = exp(-value/temp);
      return -exp_value;
    };

    std::string name() {return "cEIexp";};

  private:
    double mTemperature;
    size_t mExp;
  };


  class ExpectedImprovementUnscIncumbent: public Criteria
  {
  public:
      virtual ~ExpectedImprovementUnscIncumbent(){};
      void init(NonParametricProcess* proc)
      {
        mProc = proc;
        mExp = 1;
        mShouldComputeMin = true;
      };

      void setParameters(const vectord &params)
      { mExp = static_cast<size_t>(params(0)); };

      size_t nParameters() {return 1;};

      void update(const vectord &x)
      {
        size_t num_points = mProc->getData()->getNSamples();

        // Find unscented min
        vectord px = mProc->getData()->getSampleX(0);
        mUnscentedMin = mProc->getIntegratedOutcome(px);
        for(size_t i = 1; i<num_points; i++)
          {
            px = mProc->getData()->getSampleX(i);
            double val = mProc->getIntegratedOutcome(px);
            if(val < mUnscentedMin) mUnscentedMin = val;
          }

        mShouldComputeMin = false;
      }

      double operator() (const vectord &x)
      {
        if(mShouldComputeMin) update(x); //TODO(Javier): Fix update not called before first criteria call

        //TODO(Javier): Dont use mean, as it affects other things of the EI not just the diff
        ProbabilityDistribution* prob = mProc->prediction(x);
        prob->setMeanAndStd(mProc->getIntegratedOutcome(x), prob->std());

        double diffhack = mUnscentedMin - mProc->getIntegratedOutcome(x) + prob->mean();
        return prob->negativeExpectedImprovement(diffhack,mExp);
      };

      std::string name() {return "cEIUnscIncumbent";};

  private:
      size_t mExp;
      double mUnscentedMin;
      bool mShouldComputeMin;
  };
  
  // TODO(Javier): Remove this criteria function
  class RandomCriteria: public Criteria
  {
  public:
    virtual ~RandomCriteria(){};
    void init(NonParametricProcess* proc)
    { 
      mProc = proc;
    };
    
    void setParameters(const vectord &params)
    { 
    };

    size_t nParameters() {return 0;};

    double operator() (const vectord &x) 
    { 
        return 0.5;
    };

    std::string name() {return "cRandom";};
  };

  //@}

} //namespace bayesopt


#endif
