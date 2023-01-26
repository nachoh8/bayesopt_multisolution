/** \file kernel_combined.hpp 
    \brief Kernel functions that combine other kernels */
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

#ifndef  _KERNEL_COMBINED_HPP_
#define  _KERNEL_COMBINED_HPP_

#include <boost/ptr_container/ptr_vector.hpp>
#include <boost/numeric/ublas/vector_proxy.hpp>
#include "kernels/kernel_functors.hpp"

namespace bayesopt
{
  
  /**\addtogroup KernelFunctions */
  //@{

  /** \brief Abstract class for combined kernel.
   *  It allows combinations of other kernels (addition, product, etc.)
   */
  class CombinedKernel : public Kernel
  {
  public:
    virtual void init(size_t input_dim)
    { 
      n_inputs = input_dim;
    };

    void pushKernel(Kernel* crit)
    {
      mKernelList.push_back(crit);
    };


    virtual void setHyperParameters(const vectord &theta) 
    {
      using boost::numeric::ublas::subrange;

      const size_t np = mKernelList.size();
      vectori sizes(np);

      for (size_t i = 0; i < np; ++i)
	{
	  sizes(i) = mKernelList[i].nHyperParameters();
	}

      if (theta.size() != norm_1(sizes))
	{
	  FILE_LOG(logERROR) << "Wrong number of kernel parameters"; 
	  throw std::invalid_argument("Wrong number of kernel parameters");
	}

      size_t start = 0;
      for (size_t i = 0; i < np; ++i)
	{
	  mKernelList[i].setHyperParameters(subrange(theta,start,start+sizes(i)));
	  start += sizes(i);
	}
    };

    virtual vectord getHyperParameters() 
    {
      using boost::numeric::ublas::subrange;
      const size_t np = mKernelList.size();
      vectori sizes(np);

      for (size_t i = 0; i < np; ++i)
	{
	  sizes(i) = mKernelList[i].nHyperParameters();
	}
      vectord par(norm_1(sizes));

      size_t start = 0;
      for (size_t i = 0; i < np; ++i)
	{
	  size_t end = start+sizes(i);
	  subrange(par,start,end) = mKernelList[i].getHyperParameters();
	  start = end;
	}
      return par;
    };

    virtual size_t nHyperParameters() 
    {
      size_t sum = 0;
      for (size_t i = 0; i < mKernelList.size(); ++i)
	{
	  sum += mKernelList[i].nHyperParameters();
	}
      return sum;
    };

    virtual ~CombinedKernel(){};

  protected:
    boost::ptr_vector<Kernel> mKernelList;
  };

  //@}

} //namespace bayesopt

#endif
