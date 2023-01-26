/**  \file mcmc_sampler.hpp \brief Markov Chain Monte Carlo algorithms */
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


#ifndef  _MCMC_SAMPLER_HPP_
#define  _MCMC_SAMPLER_HPP_

#include <boost/scoped_ptr.hpp>
#include "randgen.hpp"
#include "optimizable.hpp"
#include "log.hpp"

namespace bayesopt {

  // We plan to add more in the future
  typedef enum {
    SLICE_MCMC,           ///< Slice sampling
    MH_MCMC,              ///< Metropolis-Hasting sampling
    REJSAMPLE_MCMC        ///< Rejection Sampling
  } McmcAlgorithms;


  /**
   * \brief Markov Chain Monte Carlo sampler
   *
   * It generates a set of particles that are distributed according to
   * an arbitrary pdf. IMPORTANT: As it should be a replacement for
   * the optimization (ML or MAP) estimation, it also assumes a
   * NEGATIVE LOG PDF.
   *
   * @see NLOPT_Optimization
   */
  class MCMCSampler
  {
  public:
    /**
     * \brief Constructor (Note: default constructor is private)
     *
     * @param rbo point to RBOptimizable type of object with the PDF
     *            to sample from. IMPORTANT: We assume that the
     *            evaluation of rbo is the NEGATIVE LOG PDF.
     * @param dim number of input dimensions
     * @param eng random number generation engine (boost)
     */
    MCMCSampler(RBOptimizable* rbo, size_t dim, randEngine& eng);
    virtual ~MCMCSampler();

    /** Sets the sampling algorithm (slice, MH, etc.) */
    void setAlgorithm(McmcAlgorithms newAlg);

    /** Sets the number of particles that are stored */
    void setNParticles(size_t nParticles);

    /**Usually, the initial samples of any MCMC method are biased and
     *	they are discarded. This phase is called the burnout. This
     *	method sets the number of particles to be discarded
     */
    void setNBurnOut(size_t nParticles);

    /** Limits of the hypercube. */
    void setLimits(const vectord& down, const vectord& up);

    /** Limits of the hypercube assuming that all dimensions have the same limits. */
    void setLimits(double down, double up);

    /**Changes the behaviour of the mcmc to take into account Log-scale
     *  values (true) or Linear-scale values (false).
     */
    void setLogScale(bool activate);

    /** Compute the set of particles according to the target PDF.
     * @param Xnext input: initial point of the Markov Chain,
     *              output: last point of the Markov Chain
     */
    void run(vectord &Xnext);
    void runParallelTempering(vectord temperatures);
    void stepParallelTempering(vectord& temperatures, vecOfvec& configurations);
    void swapParallelTempering(vectord& temperatures, vecOfvec& configurations, bool even_index);
    double evaluate(vectord& x);
    double evaluateTemperature(vectord& x, double temperature);

    vectord getParticle(size_t i);

    void printParticles();

  private:
    void randomJump(vectord &x);
    void randomUniformJump(vectord &x);
    void normalJump(vectord &x, float stdev);

    void sampleParticle(vectord &x);
    void burnOut(vectord &x);

    void sliceSample(vectord &x);
    void metropolisHastingSample(vectord &x);

    boost::scoped_ptr<RBOptimizableWrapper> obj;

    McmcAlgorithms mAlg;
    size_t mDims;
    size_t nBurnOut;
    size_t nSamples;
    bool mStepOut;

    bool mLogscale;

    bool mApplyTemperature;
    double mCurrentTemperature;

    vectord mSigma;
    randEngine& mtRandom;
  public: // TODO (Javier): public getter/setter for testing purposes
    vecOfvec mParticles;
    vecOfvec mBurnout;
    float mProposalSigma;

    std::vector<double> mDebugSwapAcceptance;
    std::vector<int> mDebugSwapAttempts;

  private: //Forbidden
    MCMCSampler();
    MCMCSampler(MCMCSampler& copy);
  
  private:
    vectord lb, ub;
  };

  inline void MCMCSampler::setAlgorithm(McmcAlgorithms newAlg)
  { mAlg = newAlg; };

  inline void MCMCSampler::setNParticles(size_t nParticles)
  { nSamples = nParticles; };

  inline void MCMCSampler::setNBurnOut(size_t nParticles)
  { nBurnOut = nParticles; };

  inline void MCMCSampler::setLogScale(bool activate)
  { mLogscale = activate; };

  inline vectord MCMCSampler::getParticle(size_t i)
  { return mParticles[i]; };

  inline void MCMCSampler::printParticles()
  {
    for(size_t i=0; i<mParticles.size(); ++i)
      {
	FILE_LOG(logDEBUG) << i << "->" << mParticles[i]
			   << " | Log-lik " << -obj->evaluate(mParticles[i]);
      }
  }

} //namespace bayesopt


#endif
