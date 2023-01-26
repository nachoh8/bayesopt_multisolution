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

#ifndef  _KERNEL_WARP_HPP_
#define  _KERNEL_WARP_HPP_

#include <boost/math/distributions/beta.hpp>
#include "kernels/kernel_functors.hpp"

namespace bayesopt
{
  
  /**\addtogroup KernelFunctions */
  //@{
  /** \brief Applying warping over a kernel */
  class KernelWarp: public Kernel
  {
  public:
    virtual void init(size_t input_dim)
    { 
      n_inputs = input_dim;
      n_params = n_inputs*2;
      for (size_t i = 0; i < n_inputs; i++) 
	{
	  boost::math::beta_distribution<>* bd = new boost::math::beta_distribution<>(1,1);
	  mBetaDist.push_back(bd);
	}
    };

    double operator()(const vectord &x1, const vectord &x2)
    {       
      vectord w_x1(n_inputs);
      vectord w_x2(n_inputs);

      for (size_t i = 0; i < n_inputs; i++) 
	{
	  if (x1(i) < 0.0) std::cout << x1  << std::endl;
	  if (x2(i) < 0.0) std::cout << x2  << std::endl;
	  w_x1(i) = cdf(mBetaDist[i],x1(i));
	  w_x2(i) = cdf(mBetaDist[i],x2(i));
	}

      return (*mInnerKernel)(w_x1,w_x2);
    };

    //TODO: Not implemented
    double gradient(const vectord &x1, const vectord &x2,
		    size_t component)
    { return 0.0; };


    void pushKernel(Kernel* crit)
    {
      mInnerKernel.reset(crit);
    };

    void setHyperParameters(const vectord &theta) 
    {
      using boost::numeric::ublas::subrange;
   
      if (theta.size() != nHyperParameters())
	{
	  FILE_LOG(logERROR) << "Wrong number of kernel parameters: " 
			     << nHyperParameters() << " expected and "
			     << theta.size() << " received."; 
	  throw std::invalid_argument("Wrong number of kernel parameters");
	}

      for (size_t i = 0; i < n_inputs; i++) 
	{
	  boost::math::beta_distribution<> *bd = 
	    new boost::math::beta_distribution<>(std::exp(theta(2*i)),std::exp(theta(2*i+1)));
	  mBetaDist.replace(i,bd);
	}

      mInnerKernel->setHyperParameters(subrange(theta,n_params,theta.size()));
    };

    vectord getHyperParameters() 
    {
      using boost::numeric::ublas::subrange;

      vectord par(nHyperParameters());

      for (size_t i = 0; i < n_inputs; ++i)
	{
	  par(2*i) = std::log(mBetaDist[i].alpha());
	  par(2*i+1) = std::log(mBetaDist[i].beta());
	}
      subrange(par,n_params,par.size()) = mInnerKernel->getHyperParameters();
      return par;
    };

    size_t nHyperParameters() 
    {
      return mInnerKernel->nHyperParameters() + n_params;
    };

    virtual ~KernelWarp(){};



  private:
    size_t n_params;
    boost::scoped_ptr<Kernel> mInnerKernel;
    boost::ptr_vector<boost::math::beta_distribution<> > mBetaDist;
  };

  //@}

} //namespace bayesopt

#endif
