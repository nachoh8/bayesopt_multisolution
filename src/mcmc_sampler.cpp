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
#include "lhs.hpp"
#include "mcmc_sampler.hpp"
#include <algorithm> //min

namespace bayesopt
{
  MCMCSampler::MCMCSampler(RBOptimizable* rbo, size_t dim, randEngine& eng):
    mtRandom(eng), obj(new RBOptimizableWrapper(rbo))
  {
    mAlg     = SLICE_MCMC;
    mDims    = dim;
    nBurnOut = 100;
    nSamples = 10;
    mStepOut = true;
    mSigma   = svectord(dim,6);
    mLogscale = true;
    mProposalSigma = 0.3;

    mApplyTemperature = false;
    mCurrentTemperature = 0.0;

    lb = zvectord(dim);
    ub = svectord(dim, 1.0);
  };

  MCMCSampler::~MCMCSampler()
  {};

  double MCMCSampler::evaluate(vectord& x){
    if(mApplyTemperature){
      return evaluateTemperature(x, mCurrentTemperature);
    }
    else{
      return -obj->evaluate(x);
    }
  };

  double MCMCSampler::evaluateTemperature(vectord& x, double temperature){
    double y = -obj->evaluate(x);
    return std::exp(y/temperature);
  };

  void MCMCSampler::runParallelTempering(vectord temperatures){
    bool origUseTemp = mApplyTemperature;
    double origTemp = mCurrentTemperature;
    mBurnout.clear();

    // Set initial configurations
    size_t swap_every = 50;

    int numConf = temperatures.size();
    vecOfvec configurations;
    for(size_t i=0; i < numConf; i++){
      configurations.push_back(vectord(mDims, 0.5));
      randomUniformJump(configurations[i]);
      std::cout << configurations[i] << std::endl;
    }

    mDebugSwapAcceptance.clear();
    mDebugSwapAttempts.clear();
    for(size_t i=0; i<numConf; i++){
      mDebugSwapAcceptance.push_back(0.);
      mDebugSwapAttempts.push_back(0);
    }

    for(size_t i=0; i < nBurnOut; i++){
      if(i % swap_every == 0){
        swapParallelTempering(temperatures, configurations, true);
      }
      else if ((i + swap_every/2) % swap_every == 0){
        swapParallelTempering(temperatures, configurations, false);
      }
      stepParallelTempering(temperatures, configurations);
    }

    for(size_t i=0; i<numConf; i++){
      mDebugSwapAcceptance[i] /= (float)mDebugSwapAttempts[i];
      std::cout << "swap acceptance of " << temperatures[i] << ": " << mDebugSwapAcceptance[i] << std::endl;
    }

    mParticles.clear();
    mParticles.push_back(configurations[0]);

    mApplyTemperature = origUseTemp;
    mCurrentTemperature = origTemp;
  };

  void MCMCSampler::stepParallelTempering(vectord& temperatures, vecOfvec& configurations){
    for(size_t i; i<temperatures.size(); i++){
      mCurrentTemperature = temperatures[i];
      metropolisHastingSample(configurations[i]);
      mBurnout.push_back(configurations[i]);
    }
  }

  void MCMCSampler::swapParallelTempering(vectord& temperatures, vecOfvec& confs, bool even_index){
    randNFloat sample( mtRandom, normalDist(0,1) );
    size_t init = 0;
    if (!even_index) init = 1;

    bool changeDirection = sample() < 0.5; // Randomly choose which way we compute swaps

    for(size_t i = init; i+1 < temperatures.size(); i+=2){
      size_t from = i;
      size_t to = i+1;
      if(changeDirection){
        from = i+1;
        to = i;
      }

      mDebugSwapAttempts[from] += 1;
      mDebugSwapAttempts[to] += 1;
      double e1 = evaluateTemperature(confs[from], temperatures[from]);
      double e2 = evaluateTemperature(confs[to], temperatures[to]);
      double b1 = 1./temperatures[from];
      double b2 = 1./temperatures[to];
      double deltaS = (b1-b2)*(e1 - e2);
      double prob = std::exp(deltaS);

      //std::cout << " deltaE: " << e1 << " vs " << e2 << " - prob: " << prob << std::endl;
      if(prob > sample()){
        std::swap(confs[from], confs[to]);
        mDebugSwapAcceptance[from] += 1;
        mDebugSwapAcceptance[to] += 1;
      }
    }
  }

  void MCMCSampler::randomJump(vectord &x)
  {
    randNFloat sample( mtRandom, normalDist(0,1) );
    for(vectord::iterator it = x.begin(); it != x.end(); ++it)
      {
	      *it = sample()*6;
      }
  }

  void MCMCSampler::randomUniformJump(vectord &x)
  {
    randFloat sample( mtRandom, realUniformDist(0,1) );
    for(vectord::iterator it = x.begin(); it != x.end(); ++it)
      {
	      *it = sample();
      }
  }

  void MCMCSampler::normalJump(vectord &x, float stdev)
  {
    randNFloat normal(mtRandom,normalDist(0, stdev));
    for(vectord::iterator it = x.begin(); it != x.end(); ++it)
      {
	      *it = normal();
      }
  }

  void MCMCSampler::sliceSample(vectord &x)
  {
    randFloat sample( mtRandom, realUniformDist(0,1) );
    size_t n = x.size();
    size_t n_steps;
    size_t n_limit = 10;

    std::vector<int> perms = utils::return_index_vector(0,n);

    utils::randomPerms(perms, mtRandom);

    for (size_t i = 0; i<n; ++i)
      {
	const size_t ind = perms[i];
	const double sigma = mSigma(ind);

	const double y_max = evaluate(x);

	const double y = y_max + std::log(sample());

	if (y == 0.0)
	  {
	    throw std::runtime_error("Error in MCMC: Initial point"
				     " out of support region.");
	  }

	// Step out
	const double x_cur = x(ind);
	const double r = sample();
	double xl = x_cur - r * sigma;
	double xr = x_cur + (1-r)*sigma;

	if (mStepOut)
	  {
	    x(ind) = xl; n_steps = 0;
	    while ((evaluate(x) > y) & (n_steps < n_limit)) { x(ind) -= sigma; n_steps++;}
	    xl = x(ind);

	    x(ind) = xr; n_steps = 0;
	    while ((evaluate(x) > y) & (n_steps < n_limit)) { x(ind) += sigma; n_steps++;}
	    xr = x(ind);
	  }

	//Shrink
	bool on_slice = false;
	while (!on_slice)
	  {
	    x(ind) = (xr-xl) * sample() + xl;
	    if (evaluate(x) < y)
	      {
		if      (x(ind) > x_cur)  xr = x(ind);
		else if (x(ind) < x_cur)  xl = x(ind);
		else throw std::runtime_error("Error in MCMC. Slice colapsed.");
	      }
	    else
	      {
		on_slice = true;
	      }
	  }
      }
  }

  void MCMCSampler::metropolisHastingSample(vectord &x)
  {
    randFloat sampleU( mtRandom, realUniformDist(0,1) );

    // Get new candidate xp
    vectord xp = vectord(x.size());
    if( mLogscale )
      {
        randomJump(xp);
        xp = x - xp;
      }
    else if( mApplyTemperature ) // TODO(Javier): quick hack to use a different proposal distribution for parallel tempering
    {
      normalJump(xp, 0.25);
      xp = x - xp;
    }
    else {
      float val = sampleU();
      if(val < 0.2){
        randomUniformJump(xp);
      }
      else if(val < 0.4){
        normalJump(xp, 0.3);
        xp = x - xp;
      }
      else if(val < 0.6){
        normalJump(xp, 0.1);
        xp = x - xp;
      }
      else if(val < 0.8){
        normalJump(xp, 0.01);
        xp = x - xp;
      }
      else{
        normalJump(xp, 0.001);
        xp = x - xp;
      }
    }

    // Patch to prevent the mcmc going out of bounds
    bool within_bounds = true;
    for(size_t i=0; i<xp.size(); i++){
      if (xp(i) < lb(i) || xp(i) > ub(i)){
        within_bounds = false;
      }
    }

    if(within_bounds)
    {
      // Check acceptance probability
      double a;
      if( mLogscale ) a = std::exp( evaluate(xp) - evaluate(x) );
      else
        {
          double x_eval = evaluate(x);
          if(x_eval == 0.0) a = 1.0;
          else              a = evaluate(xp) / x_eval;
        }

      if ( sampleU() < a )
        {
          x = xp;
        }
    }
  }

  //TODO: Include new algorithms when we add them.
  void MCMCSampler::sampleParticle(vectord &x)
  {
    try
    {
      switch(mAlg){
        case SLICE_MCMC:
          sliceSample(x);
          break;

        case MH_MCMC:
          metropolisHastingSample(x);
          break;
      }
    }
    catch(std::runtime_error& e)
    {
      FILE_LOG(logERROR) << e.what();
      FILE_LOG(logERROR) << "Doing random jumps.";

      int loops = 0;

      do{
        if(mLogscale) randomJump(x);
        else          randomUniformJump(x);

        loops++;

      }while ( evaluate(x) == 0.0);

      FILE_LOG(logERROR) << "Likelihood." << x << " | " << -evaluate(x);

      if(loops > 1)
        {
          std::cout << "Number of random retrials: " << loops << std::endl;
        }

    }
  }

  void MCMCSampler::burnOut(vectord &x)
  {
    for(size_t i=0; i<nBurnOut; ++i)
      {
        sampleParticle(x);
        FILE_LOG(logDEBUG) << "Burning particle (" << (i+1) << " / " << nBurnOut << ") -> " << x;
        // mBurnout.push_back(x);
      }
  }

  void MCMCSampler::run(vectord &Xnext)
  {
    // mBurnout.clear();

    if (nBurnOut>0) burnOut(Xnext);

    mParticles.clear();
    for(size_t i=0; i<nSamples; ++i)
      {
        FILE_LOG(logDEBUG) << "MCMC particle:" << i;
	sampleParticle(Xnext);
	mParticles.push_back(Xnext);
      }
    printParticles();
  }

  void MCMCSampler::setLimits(const vectord& down, const vectord& up) {
    std::copy(down.begin(),down.end(),lb.begin());
    std::copy(up.begin(),up.end(),ub.begin());
  }

  void MCMCSampler::setLimits(double down, double up) {
    for(size_t i = 0; i<lb.size();++i)
    {
      lb[i] = down; ub[i] = up;
    }
  }
}// namespace bayesopt
