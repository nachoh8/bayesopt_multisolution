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

#include <limits>
#include <boost/numeric/ublas/matrix_proxy.hpp>

#include "lhs.hpp"
#include "dataset.hpp"
#include "randgen.hpp"
#include "log.hpp"
#include "boundingbox.hpp"
#include "inneroptimization.hpp"
#include "mcmc_sampler.hpp"
#include "posteriors/posterior_model.hpp"

#include "bayesopt/bayesopt.hpp"


namespace bayesopt  {

  class CritCallback: public RBOptimizable
  {
  public:
    explicit CritCallback(ContinuousModel* model, bool performCorrection)
      :mBO(model)
    {
      mPerformCorrection = performCorrection;
      mConsecutiveErrors = 0;
    };
    double evaluate(const vectord &query)
    {
      if(mPerformCorrection)
        {
          double raw_value = 0.0;
          double value = 0.0;
          if (mBO->checkBounds(query))
            {
              raw_value = mBO->evaluateCriteria(query);
              value = mBO->criteriaCorrection(raw_value);
            }
          if(value>1e-20)
            {
            mConsecutiveErrors += 1;

            if(mConsecutiveErrors > 8)
              {
                  FILE_LOG(logDEBUG) << "Positive values in criteria during MCMC ("
                                       << mConsecutiveErrors << " consecutive errors), current value:"
                                       << value;
              }
            }
          else{
            mConsecutiveErrors = 0;
          }
          return value;
        }
      else
        {
          return mBO->evaluateCriteria(query);
        }
    }

    void reset_callback() {
      mConsecutiveErrors = 0;
    }

  private:
    ContinuousModel* mBO;
    bool mPerformCorrection;
    int mConsecutiveErrors;
  };

  class OptCallback: public RBOptimizable
  {
  public:
    explicit OptCallback(ContinuousModel* model):mBO(model){};
    double evaluate(const vectord &query)
    {
      return mBO->evaluatePolishSample(query);
    }
    ContinuousModel* mBO;
  };

  ContinuousModel::ContinuousModel(size_t dim, Parameters parameters):
    BayesOptBase(dim,parameters)
  {
    mCallback.reset(new CritCallback(this, false));
    cOptimizer.reset(new NLOPT_Optimization(mCallback.get(),dim));
    cOptimizer->setAlgorithm(COMBINED);
    cOptimizer->setMaxEvals(parameters.n_inner_iterations);

    vectord lowerBound = zvectord(mDims);
    vectord upperBound = svectord(mDims,1.0);
    mBB.reset(new utils::BoundingBox<vectord>(lowerBound,upperBound));
  } // Constructor

  ContinuousModel::~ContinuousModel()
  {
    //    delete cOptimizer;
  } // Default destructor

  void ContinuousModel::setBoundingBox(const vectord &lowerBound,
				       const vectord &upperBound)
  {
    // We don't change the bounds of the inner optimization because,
    // thanks to this bounding box model, everything is mapped to the
    // unit hypercube, thus the default inner optimization are just
    // right.
    mBB.reset(new utils::BoundingBox<vectord>(lowerBound,upperBound));

    FILE_LOG(logINFO) << "Bounds: ";
    FILE_LOG(logINFO) << lowerBound;
    FILE_LOG(logINFO) << upperBound;
  } //setBoundingBox

  bool ContinuousModel::checkBounds(const vectord &query)
  {
    const size_t ll = query.size();
    for(size_t ii = 0; ii < ll; ++ii)
      {
        if(query(ii) < 0 || query(ii) > 1)
          {
            return false;
        }
      }
    return true;
  }



  // vectord ContinuousModel::getLastPoint()
  // {
  //   return mBB->unnormalizeVector(getData()->getLastSampleX());
  // }
  void ContinuousModel::finalizeOptimization()
  {
    if(mParameters.n_final_samples > 0)
      {
        polishOptimization(mParameters.n_final_samples);
      }
  }

  void ContinuousModel::polishOptimization(int numIters,
					   double unitInterval)
  {
    // Configure Optimization Callback
    boost::scoped_ptr<OptCallback> optCallback;
    optCallback.reset(new OptCallback(this));

    boost::scoped_ptr<NLOPT_Optimization> localOptimizer;
    localOptimizer.reset(new NLOPT_Optimization(optCallback.get(), mDims));
    localOptimizer->setAlgorithm(BOBYQA);
    localOptimizer->setMaxEvals(numIters);

    // Configure initial point and bounds
    vectord initialPoint = getData()->getPointAtMinimum();

    vectord vd(initialPoint.size());
    vectord vu(initialPoint.size());
    for (size_t i = 0; i < initialPoint.size(); ++i)
      {
        vd[i] = initialPoint(i) - unitInterval;
        vu[i] = initialPoint(i) + unitInterval;

        // Clamp values
        if(vd[i] < 0) vd[i] = 0;
        if(vu[i] > 1) vu[i] = 1;
      }
    localOptimizer->setLimits(vd,vu);

    // Run optimization
    bool perDimension = false; // do not scale maxEvals by dimensions
    localOptimizer->run( initialPoint, perDimension );

    // Update model in case is needed for more iterations
    mModel->updateHyperParameters();
    mModel->fitSurrogateModel();
  }

  //////////////////////////////////////////////////////////////////////

  vectord ContinuousModel::samplePoint()
  {
    randFloat drawSample(mEngine,realUniformDist(0,1));
    vectord Xnext(mDims);
    for(vectord::iterator x = Xnext.begin(); x != Xnext.end(); ++x)
      {
	    *x = drawSample();
      }
    return Xnext;
  };

  void ContinuousModel::findOptimal(vectord &xOpt)
  {
    xOpt = samplePoint(); // random init point
    double minf = cOptimizer->run(xOpt);
    if (mParameters.num_solutions > 1) {
      return;
    }

    //Let's try some local exploration like spearmint
    randNFloat drawSample(mEngine,normalDist(0,0.001));
    for(size_t ii = 0;ii<5; ++ii)
      {
	vectord pert = getData()->getPointAtMinimum();
	for(size_t j=0; j<xOpt.size(); ++j)
	  {
	    pert(j) += drawSample();
	  }
	try
	  {
	    double minf2 = cOptimizer->localTrialAround(pert);
	    if (minf2<minf)
	      {
		minf = minf2;
		FILE_LOG(logDEBUG) << "Local beats Global";
		xOpt = pert;
	      }
	  }
	catch(std::invalid_argument& e)
	  {
	    FILE_LOG(logDEBUG) << "There is some issue with local optimization. Ignoring";
      FILE_LOG(logDEBUG) << e.what();
	    /*We ignore this one*/
	  }
      }
  };

  void ContinuousModel::sampleCriteria(vectord &xNext)
  {
    if (! mSamplingCallback || ! cSampler) {
      mSamplingCallback.reset(new CritCallback(this, true));

      cSampler.reset(new MCMCSampler(mSamplingCallback.get(), mDims, mEngine));
      cSampler->setAlgorithm(MH_MCMC);
      cSampler->setNParticles(1);
      cSampler->setNBurnOut(mBurnOutSamples);
      cSampler->setLogScale(false);
    } else {
      mSamplingCallback->reset_callback();
    }

    if (cluster_sampling) {
      // vectord centroid = centroids[current_cluster];
      // svectord dv = svectord(centroid.size(), max_dist_centroid[current_cluster]);

      vectord clb = cluster_lb[current_cluster]; // centroid - dv;
      vectord cub = cluster_ub[current_cluster]; // centroid + dv;
      for (size_t i = 0; i < clb.size(); i++) {
        if (clb[i] < 0.0) {
          clb[i] = 0.0;
        }

        if (cub[i] > 1.0) {
          cub[i] = 1.0;
        }
      }

      FILE_LOG(logDEBUG) << "Cluster bounds: " << clb << ", " << cub;

      cOptimizer->setLimits(clb, cub);
      cSampler->setLimits(clb, cub);
    } else {
      cOptimizer->setLimits(0.0, 1.0);
      cSampler->setLimits(0.0, 1.0);
    }

    startAnnealingCycle();

    // Use optimal as start of the chain
    vectord xInit = xNext; // vectord(xNext.size());
    findOptimal(xInit);
    FILE_LOG(logDEBUG) << "Init sampling point: " << xInit << " -> " << evaluateCriteria(xInit);
    
    cSampler->run(xInit);
    xNext = cSampler->getParticle(0);

    FILE_LOG(logDEBUG) << "Final sampling point: " << xNext << " -> " << evaluateCriteria(xNext);
  
    /*mLastMCMCSamples.clear();
    mLastMCMCParticles.clear();
    for(size_t i=0; i<cSampler->mBurnout.size(); i++){
      mLastMCMCSamples.push_back(cSampler->mBurnout[i]);
    }
    for(size_t i=0; i<cSampler->mParticles.size(); i++){
      mLastMCMCParticles.push_back(cSampler->mParticles[i]);
    }*/
  };
  void ContinuousModel::parallelTemperingCriteria(vectord &xNext)
  {
    mCallback.reset(new CritCallback(this, true));

    cSampler.reset(new MCMCSampler(mCallback.get(), mDims, mEngine));
    cSampler->setAlgorithm(MH_MCMC);
    cSampler->setNParticles(1);
    cSampler->setNBurnOut(mBurnOutSamples);
    cSampler->setLogScale(false);
    cSampler->mBurnout.clear();
    cSampler->mParticles.clear();

    vectord temperatures(5);
    size_t j = 0;
    temperatures[j] = 0.01; j++;
    temperatures[j] = 0.1; j++;
    temperatures[j] = 0.4; j++;
    temperatures[j] = 1.0; j++;
    temperatures[j] = 2.5; j++;

    double maxY = getData()->getValueAtMaximum();
    double minY = getData()->getValueAtMinimum();
    for(size_t i=0; i<temperatures.size(); i++){
      temperatures[i] = temperatures[i] * (maxY - minY);
    }

    cSampler->runParallelTempering(temperatures);
    xNext = cSampler->getParticle(0);

    mLastMCMCSamples.clear();
    mLastMCMCParticles.clear();
    for(size_t i=0; i<cSampler->mBurnout.size(); i++){
      mLastMCMCSamples.push_back(cSampler->mBurnout[i]);
    }
    for(size_t i=0; i<cSampler->mParticles.size(); i++){
      mLastMCMCParticles.push_back(cSampler->mParticles[i]);
    }
  }
  void ContinuousModel::rejectionSamplingCriteria(vectord &xNext)
  {
    boost::scoped_ptr<CritCallback> ccb;
    ccb.reset(new CritCallback(this, true));
    
    // Rejection Sampling
    findOptimal(xNext);

    std::cout << "optimizer: " << xNext << " = " << evaluateCriteria(xNext) << std::endl;

    double neg_lim = 1.0 * ccb->evaluate(xNext);

    randFloat sample( mEngine, realUniformDist(0,1) );

    int num_iters = 0;

    // Random search a better value if limit is zero
    if (neg_lim > -1e-15){
      for(size_t i=0; i<100000; i++){
        for(vectord::iterator it = xNext.begin(); it != xNext.end(); ++it)
          {
            *it = sample();
          }
        
        double val = ccb->evaluate(xNext);
        if(val < neg_lim){
          neg_lim = val;
        }
      }

      neg_lim = neg_lim * 1.1;
      std::cout << "RS recalculate limit:" << neg_lim << std::endl;
    }
    else{
      std::cout << "RS limit:" << neg_lim << std::endl;
    }

    if (neg_lim > -1e-15){
      std::cout << "Cannot find non-zero values" << std::endl;

      for(size_t i=0; i<100; i++){
        for(vectord::iterator it = xNext.begin(); it != xNext.end(); ++it)
          {
            *it = sample();
          }
        ProbabilityDistribution* p = getPrediction(xNext);
        std::cout << p->mean() << "+-" <<p->std() << " - pdf:" << p->pdf(0.49) << std::endl;
      }
    }
    else{
      bool found = false;
      while(!found){
        for(vectord::iterator it = xNext.begin(); it != xNext.end(); ++it)
          {
            *it = sample();
          }
        
        double val = ccb->evaluate(xNext) / neg_lim;
        double u = sample();

        num_iters++;

        if(val > u){
          found = true;
        }
        if(num_iters > 150000){
          found = true;
        }
      }
    }

    std::cout << "RS req iters:" << num_iters << std::endl;
    std::cout << "final X:" << xNext << " = " << evaluateCriteria(xNext) << std::endl;
  };

  vectord ContinuousModel::remapPoint(const vectord& x)
  {
    return mBB->unnormalizeVector(x);
  }

  void ContinuousModel::generateInitialPoints(matrixd& xPoints)
  {
    utils::samplePoints(xPoints,mParameters.init_method,mEngine);
  }

  double ContinuousModel::evaluatePolishSample(const vectord &x)
  {
    double y = evaluateSampleInternal(x);
    mModel->addSample(x,y);
    mCurrentIter++;

    if(mParameters.verbose_level >0)
      {
        size_t iteration = mCurrentIter - mParameters.n_iterations;

        FILE_LOG(logINFO) << "Polish Iteration: " << iteration << " of "
                  << mParameters.n_final_samples << " | Total samples: "
                  << mCurrentIter+mParameters.n_init_samples ;
        FILE_LOG(logINFO) << "Query: "         << remapPoint(x); ;
        FILE_LOG(logINFO) << "Query outcome: " << y ;
        FILE_LOG(logINFO) << "Best query: "    << getFinalResult();
        FILE_LOG(logINFO) << "Best outcome: "  << getValueAtMinimum();
      }

    return y;
  }
}  //namespace bayesopt
