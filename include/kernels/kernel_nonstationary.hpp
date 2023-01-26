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

#ifndef  _KERNEL_NONSTATIONARY_HPP_
#define  _KERNEL_NONSTATIONARY_HPP_

//#include "ublas_elementwise.hpp"
#include <valarray>
#include "kernels/kernel_combined.hpp"

namespace bayesopt
{
  
  /**\addtogroup KernelFunctions */
  //@{

  class WeihtingFunction
  {
  public:
    WeihtingFunction(const vectord& center, const vectord& stds):
      center_(center), std_(stds)
    {}

    void updateCenter(const vectord& center)
    {
      center_ = center;
    }

    double unnormalizedWeight(const vectord& x) 
    {
      vectord bias = x - center_;
      vectord nx = utils::ublas_elementwise_div(bias, std_);
      double w = 1.0;
      for (vectord::iterator it = nx.begin(); it != nx.end(); ++it) 
	{
	  w *= boost::math::pdf(d_,*it); 
	}

      return w;
    };

  private:
    boost::math::normal d_;
    vectord center_;
    vectord std_;
  };

  /** \brief Weghted sum of two kernels */
  class KernelNonstationary: public CombinedKernel
  {
  public:
    typedef boost::ptr_vector<WeihtingFunction> weihtVector;

    double operator()(const vectord &x1, const vectord &x2)
    { 
      double sum = 0.0;
      vectord w1 = computeNormalizedWeights(x1);
      vectord w2 = computeNormalizedWeights(x2);

      for(size_t i = 0; i<mKernelList.size(); ++i)
	{ 
	  sum += mKernelList[i](x1,x2)*w1(i)*w2(i); 
	}
      return sum;
    };

    double gradient(const vectord &x1, const vectord &x2,
		    size_t component)
    { 
      double sum = 0.0;
      vectord w1 = computeNormalizedWeights(x1);
      vectord w2 = computeNormalizedWeights(x2);
      for(size_t i = 0; i<mKernelList.size(); ++i)
	{ 
	  sum += mKernelList[i].gradient(x1,x2,component)*w1(i)*w2(i); 
	}
      return sum;
    };

    virtual void setDomain(const vecOfvec &theta, const vecOfvec &th_std) 
    {
      for (size_t i = 0; i < theta.size(); ++i) 
	{
	  mW.push_back(new WeihtingFunction(theta[i],th_std[i]));
	}
      if (mW.size() != mKernelList.size()) 
	{
	  std::stringstream ss;
	  ss << "Incorrect number of domain inputs"
	     << mW.size() << ","
	     << mKernelList.size();
	  std::string error_msg = ss.str();
	  throw std::runtime_error(error_msg); 
	}
    };

    vectord computeNormalizedWeights(const vectord& x)
    {
      vectord lambda(mW.size());
      for (size_t i = 0; i < mW.size(); ++i) 
	{
	  lambda(i) = mW[i].unnormalizedWeight(x);
	}

      lambda /= norm_1(lambda);
      lambda = utils::ublas_elementwise_sqrt(lambda);

      return lambda;
    }


  protected:
    weihtVector mW;
  };

  /** \brief Weghted sum of two kernels */
  class KernelNonstationaryMoving: public KernelNonstationary
  {
  public:
    virtual void setHyperParameters(const vectord &theta) 
    {
      using boost::numeric::ublas::subrange;
      centers_ = subrange(theta,0,n_inputs);
      mW[0].updateCenter(centers_);
      KernelNonstationary::setHyperParameters(subrange(theta,n_inputs,theta.size()));
    };

    virtual vectord getHyperParameters() 
    {
      using boost::numeric::ublas::subrange;
      vectord theta(nHyperParameters());

      subrange(theta,0,n_inputs) = centers_;      
      subrange(theta,n_inputs,theta.size()) = KernelNonstationary::getHyperParameters();
      return theta;
    }

    virtual size_t nHyperParameters() 
    {
      return n_inputs + KernelNonstationary::nHyperParameters();
    };

  private:
    vectord centers_;
  };

  /** \brief Weghted sum of two kernels */
  class SpartanKernel: public KernelNonstationaryMoving
  {
  public:
    virtual void init(size_t input_dim)
    { 
      Kernel* loc = new MaternARD5();
      loc->init(input_dim);
      pushKernel(loc);

      vectord thl = svectord(input_dim,0.5);
      vectord thl_std = svectord(input_dim,0.05);
      mW.push_back(new WeihtingFunction(thl,thl_std));


      Kernel* glob = new MaternARD5();
      glob->init(input_dim);
      pushKernel(glob);

      vectord thg = svectord(input_dim,0.5);
      vectord thg_std = svectord(input_dim,10);
      mW.push_back(new WeihtingFunction(thg,thg_std));

      n_inputs = input_dim;
    };

    virtual void setDomain(const vecOfvec &theta, const vecOfvec &th_std) 
    {
      // We already set it up at init
    };

  };



  //@}

} //namespace bayesopt

#endif
