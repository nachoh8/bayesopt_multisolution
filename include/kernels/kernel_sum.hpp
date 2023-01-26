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

#ifndef  _KERNEL_SUM_HPP_
#define  _KERNEL_SUM_HPP_

#include "kernels/kernel_combined.hpp"

namespace bayesopt
{
  
  /**\addtogroup KernelFunctions */
  //@{

  /** \brief Sum of two kernels */
  class KernelSum: public CombinedKernel
  {
  public:
    double operator()(const vectord &x1, const vectord &x2)
    { 
      double sum = 0.0;
      for(size_t i = 0; i<mKernelList.size(); ++i)
	{ 
	  sum += mKernelList[i](x1,x2); 
	}
      return sum;
    };

    double gradient(const vectord &x1, const vectord &x2,
		    size_t component)
    { 
      double sum = 0.0;
      for(size_t i = 0; i<mKernelList.size(); ++i)
	{ 
	  sum += mKernelList[i].gradient(x1,x2,component); 
	}
      return sum;
    };

    void setDomain(const vecOfvec &theta, const vecOfvec &th_std) 
    {
      for(size_t i = 0; i<mKernelList.size(); ++i)
	{ 
	  mKernelList[i].setDomain(theta,th_std); 
	}
    };

    double getNoise()
    {
      double sum = 0.0;
      for(size_t i = 0; i<mKernelList.size(); ++i)
	{ 
	  sum += mKernelList[i].getNoise(); 
	}
      return sum;      
    }


  };

  //@}

} //namespace bayesopt

#endif
