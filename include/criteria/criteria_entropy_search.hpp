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

#ifndef  _CRITERIA_ENTROPY_SEARCH_HPP_
#define  _CRITERIA_ENTROPY_SEARCH_HPP_

#include "criteria/criteria_functors.hpp"
#include "distributions/prob_distribution.hpp"
#include "surrogates/gaussian_process.hpp"
#include "distributions/gauss_distribution.hpp"
#include "ublas_cholesky.hpp"

namespace bayesopt
{

  /**\addtogroup CriteriaFunctions  */
  //@{

  /// Entropy Search criterion
  class EntropySearch: public Criteria
  {
  public:    
    virtual ~EntropySearch(){};
    void init(NonParametricProcess* proc)
    { 
      mProc = proc;
      mNumCandidates = 20;
      mNumGPSamples = 100;
      mNumSamplesY = 10;
      mNumTrialPoints = 100;
    };

    void setParameters(const vectord &params)
    {
      std::cout << "setting parameters" << std::endl;
      mNumCandidates = static_cast<size_t>(params(0));
      mNumGPSamples = static_cast<size_t>(params(1));
      mNumSamplesY = static_cast<size_t>(params(2));
      mNumTrialPoints = static_cast<size_t>(params(3));
      
      mPercentPoints = vectord(mNumSamplesY);
      GaussianDistribution gaussian (*mtRandom);
      
      double step = 1.0/(2*mNumSamplesY);
      size_t count = 0;
      for(size_t i=1; i<2*mNumSamplesY+1; i+=2){
        double val = step*i;
        mPercentPoints[count] = gaussian.quantile(val);
        count++;
      }
    };

    size_t nParameters() { return 4; };
    
    matrixd sampleMultivariateNormal(size_t n, vectord& mean, matrixd& cov)
    { 
      namespace ublas = boost::numeric::ublas;
      //ublas::triangular_matrix<double> L (cov.size1(), cov.size2());
      matrixd L (cov.size1(), cov.size2());
      for(size_t i=0; i<L.size1(); i++){
          for(size_t j=0; j<L.size2(); j++){
            L(i,j) = 0.0;
          }
      }

      int error = utils::cholesky_decompose(cov, L);
      
      GaussianDistribution gaussian (*mtRandom);
      matrixd meanRep(mean.size(), n);
      matrixd sampleMatrix(mean.size(), n);
      for(size_t i=0; i<mean.size(); i++){
          for(size_t j=0; j<n; j++){
              meanRep(i,j) = mean(i);
              sampleMatrix(i, j) = gaussian.sample_query();
          }
      }
      
      //      if(error > 0) std::cout << "ERROR: " << error
      //		              << "C:" << cov << std::endl;
      return meanRep + ublas::prod(L, sampleMatrix);
    }
    
    double entropy(vectord prob){
      double result = 0.0;
      for(size_t i=0; i<prob.size(); i++)
        {
          double val = prob[i] * log(prob[i]);
          // Check for 0 probability (0*log(0) == 0*Inf == NaN by definition instead of 0) 
          if(val == val){
              result += val;
          }
        }
      return -result;
    }
    
    vectord bincount(matrixd f_samples){
        vectord result(mNumCandidates, 0);
        for(size_t jsample=0; jsample<f_samples.size2(); jsample++) // sample
          {
            size_t min_index = 0;
            for(size_t icand=1; icand<f_samples.size1(); icand++) // candidate
              {
                if(f_samples(icand, jsample) < f_samples(min_index, jsample))
                  {
                    min_index = icand;
                  }
              }
            result[min_index] += 1.0/f_samples.size2();
          }    
        return result;
    }
    
    void update(const vectord &x) 
    {
        std::cout << "updating Entropy Search criteria" << std::endl;
	      mProc->fitSurrogateModel();
	
        namespace ublas = boost::numeric::ublas;
        // TODO (Javier): Check that mX[0] exists
        size_t dims = mProc->getData()->mX[0].size();
        
        // Sample mNunCandidates data points which are checked for being
        // selected as representer points using (discretized) Thompson sampling
        mCandidates.clear();
        randFloat sample( *mtRandom, realUniformDist(0,1) );
        for(size_t i=0; i<mNumCandidates; i++)
        {
            // Select mNumTrialsPoints data points randomly [0,1]
            vectord candidate;
            
            double min_value = 0;            
            for(size_t j=0; j<mNumTrialPoints; j++)
            {
                vectord possibleCandidate(dims);
                for(size_t k=0; k<dims; k++)
                {
                    possibleCandidate(k) = sample();
                }
                
                // Sample function from GP posterior and select the trial points
                // which maximizes the posterior sample as representer points
                ProbabilityDistribution* pDist = mProc->prediction(possibleCandidate);
                double y = pDist->sample_query();
                
                if(j == 0 || y < min_value)
                {
                    candidate = possibleCandidate;
                    min_value = y;
                }                
            }
            
            mCandidates.push_back(candidate);
        }
        
        // Calculate base entropy
        // draw mNumGPSamples functions from GP posterior
        JointDistribution* jointd = mProc->pointsPrediction (mCandidates);
    

	
        matrixd nu_cov = jointd->mCov; //+ mProc->getNoise() * ublas::identity_matrix<double>(jointd->mCov.size1());
        matrixd f_samples = sampleMultivariateNormal(mNumGPSamples, jointd->mMean, jointd->mCov);
        // Count frequency of the candidates being the optima in the samples
        vectord p_max = bincount(f_samples);
        // Determine base entropy
        mBaseEntropy = entropy(p_max);
        
        std::cout << "baseentropy:" << mBaseEntropy << std::endl;
    }
    
    double operator() (const vectord &x){

      namespace ublas = boost::numeric::ublas;
      mProc->fitSurrogateModel();
      
      // Force update on first call
      if(mCandidates.size() == 0){
          std::cout << "NumCandidates: " << mNumCandidates << std::endl;
        std::cout << "Updating from operator()" << std::endl;
        update(x);
      }
      
      
      // mean_all, cov_all = gp.predict()
      vecOfvec points;      
      for(size_t i=0; i<mNumCandidates; i++)
        {
          points.push_back(mCandidates[i]);          
        }
      points.push_back(x);
      
      JointDistribution* jointd = mProc->pointsPrediction (points);
      
      vectord f_mean = ublas::subrange( jointd->mMean, 0, mNumCandidates );
      matrixd f_cov = ublas::subrange( jointd->mCov, 0, mNumCandidates,
                                                     0, mNumCandidates);
      
      // f_cov_query
      // f_cov_cross
      // f_cov_query_inv
      // f_cov_delta   
      double f_cov_query = jointd->mCov(mNumCandidates, mNumCandidates);
      
      vectord f_cov_cross = ublas::column(ublas::subrange( jointd->mCov,
                                            0,mNumCandidates,
                                            mNumCandidates, mNumCandidates+1), 0);

      matrixd f_cov_delta = -ublas::outer_prod(f_cov_cross, f_cov_cross);
      f_cov_delta /= f_cov_query;
      
      // f_samples
      matrixd nu_cov = f_cov + f_cov_delta; // + mProc->getNoise() * ublas::identity_matrix<double>(f_cov.size1());
      matrixd f_samples = sampleMultivariateNormal(mNumGPSamples, f_mean, nu_cov);
      
      double cumulativeDiffEntropy = 0.0;
      
      for(size_t j=0; j<mNumSamplesY; j++)
        {                    
          // y_delta
          double y_delta = sqrt(f_cov_query + mProc->getNoise()) * mPercentPoints(j);
          
          // f_mean_delta
          vectord f_mean_delta = f_cov_cross * y_delta / f_cov_query;
          
          // f_samples_j
          matrixd f_samples_j = ublas::zero_matrix<double>(f_samples.size1(), f_samples.size2());
          for(size_t ii=0; ii<f_samples.size1(); ii++)
            {
              for(size_t jj=0; jj<f_samples.size2(); jj++)
                {
                  f_samples_j(ii,jj) = f_samples(ii,jj) + f_mean_delta(ii);
                }
            }
          
          // p_max
          vectord p_max = bincount(f_samples_j);

          // a_ES
          cumulativeDiffEntropy += (mBaseEntropy - entropy(p_max)) / (double)mNumSamplesY;
        }
      
      // return negative entropy
      return -cumulativeDiffEntropy;
    };

    std::string name() {return "cEntropySearch";};

  private:
    size_t mNumCandidates;
    size_t mNumGPSamples;
    size_t mNumSamplesY;
    size_t mNumTrialPoints;
    
    vectord mPercentPoints;
    vecOfvec mCandidates;
    double mBaseEntropy;
  };
  //@}

} //namespace bayesopt


#endif
