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

#include "ublas_trace.hpp"
#include "surrogates/gaussian_process.hpp"

namespace bayesopt
{

  namespace ublas = boost::numeric::ublas;

  GaussianProcess::GaussianProcess(size_t dim, Parameters params, 
				   const Dataset* data, MeanModel& mean,
				   randEngine& eng):
    ConditionalBayesProcess(dim, params, data, mean, eng)
  {
    mSigma = params.sigma_s;
    d_ = new GaussianDistribution(eng);
    jointd_ = new JointDistribution();
  }  // Constructor


  GaussianProcess::~GaussianProcess()
  {
    delete d_;
    delete jointd_;
  } // Default destructor


  double GaussianProcess::negativeTotalLogLikelihood()
  {
    // In this case it is equivalent.
    return negativeLogLikelihood();
  }


  double GaussianProcess::negativeLogLikelihood()
  {
    const matrixd K = computeCorrMatrix();
    const size_t n = K.size1();
    matrixd L(n,n);
    utils::cholesky_decompose(K,L);

    vectord alpha(mData->mY-mMean.muTimesFeat());
    inplace_solve(L,alpha,ublas::lower_tag());
    double loglik = ublas::inner_prod(alpha,alpha)/(2*mSigma);
    loglik += utils::log_trace(L);
    return loglik;
  }

  ProbabilityDistribution* GaussianProcess::prediction(const vectord &query)
  {
    const double kq = computeSelfCorrelation(query);
    const vectord kn = computeCrossCorrelation(query);


    vectord vd(kn);
    inplace_solve(mL,vd,ublas::lower_tag());
    double basisPred = mMean.muTimesFeat(query);
    double yPred = basisPred + ublas::inner_prod(vd,mAlphaV);
    double sPred = sqrt(mSigma*(kq - ublas::inner_prod(vd,vd)));

    // TODO(Javier): Reusing a member pointer could break, for example parallel evals
    d_->setMeanAndStd(yPred,sPred);
    return d_;
  }
  
  JointDistribution* GaussianProcess::pointsPrediction(const vecOfvec &points)
  {
    size_t numSamples = mData->getNSamples();
    size_t numPoints = numSamples + points.size(); 
        
    vecOfvec allPoints;
    for(size_t ii=0; ii<mData->getNSamples(); ii++)
      {
        allPoints.push_back(mData->mX[ii]);
      }
    
    for(size_t jj=0; jj<points.size(); jj++)
      {
        allPoints.push_back(points[jj]);
      }
    
    matrixd corrMatrix(numPoints,numPoints);
    mKernel.computeCorrMatrix(allPoints,corrMatrix,getRegularizer());

    // self correlation and cross correlation
    matrixd kq = ublas::subrange<matrixd>(corrMatrix, numSamples, numPoints, numSamples, numPoints);
    matrixd kn = ublas::subrange<matrixd>(corrMatrix, 0, numSamples, numSamples, numPoints);
    
    matrixd md(kn);
    inplace_solve(mL, md, ublas::lower_tag());
    matrixd mdt = ublas::trans(md);
    
    jointd_->mMean = ublas::prod(mdt, mAlphaV);
    for(size_t jj=0; jj<points.size(); jj++)
      {
        jointd_->mMean[jj] += mMean.muTimesFeat(points[jj]);
      }
    jointd_->mCov = kq - ublas::prod(mdt, md);

    return jointd_;
  }

  void GaussianProcess::precomputePrediction()
  {
    const size_t n = mData->getNSamples();
  
    mAlphaV.resize(n,false);
    mAlphaV = mData->mY-mMean.muTimesFeat();
    inplace_solve(mL,mAlphaV,ublas::lower_tag());
  }
	
} //namespace bayesopt
