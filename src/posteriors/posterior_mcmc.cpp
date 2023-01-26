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
#include "log.hpp"
#include "posteriors/posterior_mcmc.hpp"

namespace bayesopt
{
  MCMCModel::MCMCModel(size_t dim, Parameters parameters, 
		       randEngine& eng):
    PosteriorModel(dim,parameters,eng), nParticles(10), mtRandom(eng)
  {
    //TODO: Take nParticles from parameters
    
    // Configure Surrogate and Criteria Functions
    setSurrogateModel(eng);
    setCriteria(eng);
    mIterAnnealing = 1;
    mTemperature = 1.0;
    mActiveFlag = false;
    mCoefAnnealing = parameters.active_hyperparams_coef;

    // Seting MCMC for kernel hyperparameters...
    // We use the first GP as the "walker" to get the particles. Then,
    // we will use a whole vector of GPs to avoid recomputing the
    // kernel matrices after every data point.
    size_t nhp = mGP[0].nHyperParameters();
    kSampler.reset(new MCMCSampler(&mGP[0],nhp,eng));

    kSampler->setNParticles(nParticles);
    kSampler->setNBurnOut(100);
  }

  MCMCModel::~MCMCModel()
  { } // Default destructor


  void MCMCModel::updateHyperParameters()
  {
    // We take the initial point as the last particle from previous update.
    size_t last = mGP.size()-1;
    vectord lastTheta = mGP[last].getHyperParameters();

    FILE_LOG(logDEBUG) << "Initial kernel parameters: " << lastTheta;
    kSampler->run(lastTheta);
    for(size_t i = 0; i<nParticles; ++i)
      {
	FILE_LOG(logDEBUG) << "MCMC Particle(" << i << "):" << kSampler->getParticle(i);
	mGP[i].setHyperParameters(kSampler->getParticle(i));
      }
    FILE_LOG(logDEBUG) << "Final kernel parameters: " << lastTheta;	
  };


  void MCMCModel::setSurrogateModel(randEngine& eng)
  {
    for(size_t i = 0; i<nParticles; ++i)
      {
	mGP.push_back(NonParametricProcess::create(mDims,mParameters,
						   mData,mMean,eng));
      } 
  } // setSurrogateModel

  void MCMCModel::setCriteria(randEngine& eng)
  {
    CriteriaFactory mCFactory;

    for(size_t i = 0; i<nParticles; ++i)
      {
	mCrit.push_back(mCFactory.create(mParameters.crit_name,&mGP[i]));
	mCrit[i].setRandomEngine(eng);

	if (mCrit[i].nParameters() == mParameters.crit_params.size())
	  {
	    mCrit[i].setParameters(mParameters.crit_params);
	  }
	else // If the number of parameters is different, use default.
	  {
	    if (mParameters.crit_params.size() != 0)
	      {
		FILE_LOG(logERROR) << "Expected " << mCrit[i].nParameters() 
				   << " parameters. Got " 
				   << mParameters.crit_params.size() << " instead.";
	      }
	    FILE_LOG(logINFO) << "Using default parameters for criteria.";
	  }
      }
  } // setCriteria


  double MCMCModel::getParameterInformationGain(const vectord& query)
  { 
    // What we want here is
    // log(sum(w_i (m_i - m)^2 + s_i^2)) - sum(w_i log(s_i^2))

    double mean = 0.0, std = 0.0;
    vectord mi(nParticles);
    vectord s2i(nParticles);
    for(size_t i = 0; i<nParticles; ++i)
      {
	ProbabilityDistribution* d = mGP[i].prediction(query);
	mi(i) = d->mean();
	s2i(i) = d->std()*d->std();
	mean += mi(i);
      }
    mean /= nParticles;
    double sum1 = 0.0, sum2 = 0.0;
    for(size_t i = 0; i<nParticles; ++i)
      {
	sum1 += (mi(i)-mean)*(mi(i)-mean) + s2i(i);
	sum2 += log(s2i(i));
      }
    sum1 /= nParticles; sum2 /= nParticles;   // We asume all w_i = 1/n
    return log(sum1) - sum2;
  };



}// namespace bayesopt
