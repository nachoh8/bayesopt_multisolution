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
#include "posteriors/posterior_model.hpp"
#include "posteriors/posterior_fixed.hpp"
#include "posteriors/posterior_empirical.hpp"
#include "posteriors/posterior_mcmc.hpp"
#include "fixed_dataset.hpp"

namespace bayesopt
{

  PosteriorModel* PosteriorModel::create(size_t dim, Parameters params, randEngine& eng)
  {
    switch (params.l_type)
      {
      case L_FIXED: return new PosteriorFixed(dim,params,eng);
      case L_EMPIRICAL: return new EmpiricalBayes(dim,params,eng);
      case L_DISCRETE: // TODO:return new
      case L_MCMC: return new MCMCModel(dim,params,eng);
      case L_ERROR:
      default:
	throw std::invalid_argument("Learning type not supported");
      }
  };

  PosteriorModel::PosteriorModel(size_t dim, Parameters parameters,
				 randEngine& eng):
    mParameters(parameters), mDims(dim), mMean(dim, parameters)
  {
    // std::cout << "Fixed data size: " << parameters.fixed_dataset_size << std::endl;
    const size_t fd_size = parameters.fixed_dataset_size;
    if (fd_size > 0) {
      if (parameters.n_init_samples > fd_size) {
        throw std::invalid_argument("The number of init samples must be less than or equal to fixed dataset size");
      }
      mData = new FixedDataset(parameters.fixed_dataset_size);
    } else {
      mData = new Dataset();
    }
  }

  PosteriorModel::~PosteriorModel()
  { } // Default destructor


  void PosteriorModel::setSamples(const matrixd &x, const vectord &y)
  {
    mData->setSamples(x,y);
    mMean.setPoints(mData->mX);  //Because it expects a vecOfvec instead of a matrixd
  }

  void PosteriorModel::setSamples(const matrixd &x)
  {
    mData->setSamples(x);
    mMean.setPoints(mData->mX);  //Because it expects a vecOfvec instead of a matrixd
  }

  void PosteriorModel::setSamples(const vectord &y)
  {
    mData->setSamples(y);
  }


  void PosteriorModel::setSample(const vectord &x, double y)
  {
    matrixd xx(1,x.size());  vectord yy(1);
    row(xx,0) = x;           yy(0) = y;
    mData->setSamples(xx,yy);
    mMean.setPoints(mData->mX);  //Because it expects a vecOfvec instead of a matrixd
  }

  void PosteriorModel::addSample(const vectord &x, double y)
  {  mData->addSample(x,y); mMean.addNewPoint(x);  };

  void PosteriorModel::addSample(double y)
  {  mData->addSample(y); }


  vectord PosteriorModel::getPointAtMinimum(bool useMean)
  {

    if(!useMean)
      {
        return mData->getPointAtMinimum();
      }

    vectord minPoint = getData()->mX[0];
    double minMean = getPredictionMean( minPoint );
    for(size_t i=1; i < getData()->getNSamples(); i++)
      {
        vectord point = getData()->mX[i];
        double mean = getPredictionMean( point );

        if(mean < minMean)
          {
            minMean = mean;
            minPoint = point;
          }
      }

    return minPoint;
  }

} //namespace bayesopt

