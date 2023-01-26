/**  \file spsa.hpp \brief Point perturbations based on SPSA algorithm. */
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

#ifndef _SPSA_HPP_
#define _SPSA_HPP_

#include "randgen.hpp"

namespace bayesopt
{
  class SPSA
  {
  public:
    SPSA(){};
    virtual ~SPSA(){};

    void perturbPoint(vectord& x, size_t step)
    {
      vectord pert(x.size());
      xDiffSPSA(pert, step);
      x = x + pert; 
      project01(x);
    }

    void setRandomEngine(randEngine& eng)
    { mEngine = &eng; };

    void setNoiseLevel(double noise)
    { mNoise = noise; };

  protected:

    /** 
     * \brief Computes the difference/perturbation vector for the
     * simulatenous perturbation algorithm (SPSA) to approximate the
     * gradient.
     */
    void xDiffSPSA(vectord& xDiff, size_t step)
    {
      // Bernoulli +-1 distribution
      randInt sample( *mEngine, intUniformDist(0,1) );
      double gamma = 0.101;
      double c = 10*std::sqrt(mNoise);
      double ck = c / std::pow(step,gamma);
      for (vectord::iterator it = xDiff.begin(); it != xDiff.end(); ++it)
	{
	  *it = (2*sample()-1)*ck;
	}
      return;
    }


    /** 
     * \brief Projects the vector inside the 0-1 hypercube.
     */
    void project01(vectord& x)
    {
      for (vectord::iterator it = x.begin(); it != x.end(); ++it)
	{
	  if (*it < 0.0)	  *it = 0.0;
	  if (*it > 1.0)	  *it = 1.0;
	}
      return;
    }


  protected:
    double mNoise;
    randEngine *mEngine;
  };

} //namespace bayesopt

#endif
