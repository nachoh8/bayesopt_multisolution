/*
-------------------------------------------------------------------------
   This file is part of BayesOpt, an efficient C++ library for
   Bayesian optimization.

   Copyright (C) 2011-2014 Ruben Martinez-Cantin <rmcantin@unizar.es>

   BayesOpt is free software: you can redistribute it and/or modify it
   under the terms of the GNU General Public License as published by
   the Free Software Foundation, either version 3 of the License, or
   (at your option) any later version.

   BayesOpt is distributed in the hope that it will be useful, but
   WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU General Public License for more details.

   You should have received a copy of the GNU General Public License
   along with BayesOpt.  If not, see <http://www.gnu.org/licenses/>.
------------------------------------------------------------------------
*/
#include "log.hpp"
#include "ublas_extra.hpp"
#include "posteriors/posterior_mcmc.hpp"

namespace bayesopt
{

  // We plan to add more in the future
  typedef enum {
    GRID_SAMPLING,           ///< n x n x n... samples
    SOBOL_SAMPLING,          ///< Sampling based on sobol sequences
    LHS_SAMPLING             ///< Sampling based on latin hypercubes
  } GridAlgorithms;


  /**
   * \brief Grid sampler
   *
   * It generates a set of particles that are distributed according to
   * an arbitrary pdf. IMPORTANT: As it should be a replacement for
   * the optimization (ML or MAP) estimation, it also assumes a
   * NEGATIVE LOG PDF.
   *
   * @see NLOPT_Optimization
   */
  class GridSampler
  {
  public:
    /**
     * \brief Constructor (Note: default constructor is private)
     *
     * @param rbo point to RBOptimizable type of object with the PDF
     *            to sample from. IMPORTANT: We assume that the
     *            evaluation of rbo is the NEGATIVE LOG PDF.
     * @param dim number of input dimensions
     */
    GridSampler(RBOptimizable* rbo, size_t dim);
    virtual ~GridSampler();

    /** Sets the sampling algorithm (slice, MH, etc.) */
    void setAlgorithm(GridAlgorithms newAlg)
    { mAlg = newAlg; };

    /** Sets the number of particles that are stored */
    void setNParticles(size_t nParticles)
    { nSamples = nParticles; };

    /** Generate grid of samples     */
    void sampleGrid()
    {
      // We take the n-th root of the number of particles to compute the number of samples
      // per dimension. But instead of computing the n-th root directly, we compute the
      // power of 1/n.
      unsigned int density = static_cast<int>(
			      std::pow(static_cast<double>(nParticles),
				       1.0/static_cast<double>(mDims))
					      );

      if (density >= 2) throw std::runtime_error("Insuficient samples");

      double step = float(ub-lb)/(density-1)
        return [lb+i*step for i in range(density)]

    };

    /** Compute the set of particles according to the target PDF.
     * @param Xnext input: initial weights,
     *              output: final weights
     */
    void update(vectord &w);
    {
      for (size_t i = 0; i < mParticles.size(); ++i)
	{
	  // TODO: Check if it should be incremental or replace.
	  mWeights(i) *= -obj->evaluate(mParticles[i]);
	}
      utils::normalizeDist(mWeights);
    }

    vectord getParticle(size_t i);
    { return mParticles[i]; };

    void printParticles();
    {
      for(size_t i=0; i<mParticles.size(); ++i)
	{
	  FILE_LOG(logDEBUG) << i << "->" << mParticles[i]
			     << " | Log-lik " << -obj->evaluate(mParticles[i]);
	}
    }

  private:
    boost::scoped_ptr<RBOptimizableWrapper> obj;

    GridAlgorithms mAlg;
    size_t mDims;
    size_t nSamples;

    vecOfvec mParticles;
    vectord mWeights;         ///< Weight of each model sample
    randEngine& mtRandom;

  private: //Forbidden
    GridSampler();
    GridSampler(GridSampler& copy);
  };


  GridModel::GridModel(size_t dim, Parameters parameters,
		       randEngine& eng):
    PosteriorModel(dim,parameters,eng), nParticles(10)
  {
    //TODO: Take nParticles from parameters

    // Configure Surrogate and Criteria Functions
    setSurrogateModel(eng);
    setCriteria(eng);
    mIterAnnealing = 1;
    mCoefAnnealing = parameters.active_hyperparams_coef;

    // Seting MCMC for kernel hyperparameters...
    // We use the first GP as the "walker" to get the particles. Then,
    // we will use a whole vector of GPs to avoid recomputing the
    // kernel matrices after every data point.
    size_t nhp = mGP[0].nHyperParameters();
    kSampler.reset(new GridSampler(&mGP[0],nhp,eng));

    kSampler->setNParticles(nParticles);
    kSampler->setNBurnOut(100);
  }

  GridModel::~GridModel()
  { } // Default destructor


  void GridModel::updateHyperParameters()
  {
    // We take the initial point as the last particle from previous update.
    size_t last = mGP.size()-1;
    vectord lastTheta = mGP[last].getHyperParameters();

    FILE_LOG(logDEBUG) << "Initial kernel parameters: " << lastTheta;
    kSampler->run(lastTheta);
    for(size_t i = 0; i<nParticles; ++i)
      {
	mGP[i].setHyperParameters(kSampler->getParticle(i));
      }
    FILE_LOG(logDEBUG) << "Final kernel parameters: " << lastTheta;
  };


  void GridModel::setSurrogateModel(randEngine& eng)
  {
    for(size_t i = 0; i<nParticles; ++i)
      {
	mGP.push_back(NonParametricProcess::create(mDims,mParameters,
						   mData,mMean,eng));
      }
  } // setSurrogateModel

  void GridModel::setCriteria(randEngine& eng)
  {
    CriteriaFactory mCFactory;

    for(size_t i = 0; i<nParticles; ++i)
      {
	mCrit.push_back(mCFactory.create(mParameters.crit_name,&mGP[i]));
	mCrit[i].setRandomEngine(eng);

	if (mCrit[i].nParameters() == mParameters.n_crit_params)
	  {
	    mCrit[i].setParameters(utils::array2vector(mParameters.crit_params,
						       mParameters.n_crit_params));
	  }
	else // If the number of paramerters is different, use default.
	  {
	    if (mParameters.n_crit_params != 0)
	      {
		FILE_LOG(logERROR) << "Expected " << mCrit[i].nParameters()
				   << " parameters. Got "
				   << mParameters.n_crit_params << " instead.";
	      }
	    FILE_LOG(logINFO) << "Usign default parameters for criteria.";
	  }
      }
  } // setCriteria




}// namespace bayesopt
