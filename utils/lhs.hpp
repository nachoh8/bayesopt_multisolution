/**  \file lhs.hpp \brief Latin Hypercube Sampling. */
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

#ifndef _LHS_HPP_
#define _LHS_HPP_

#include "randgen.hpp"
#include "log.hpp"
#include "indexvector.hpp"
#if defined (USE_SOBOL)
#  include "sobol.hpp"
#endif

namespace bayesopt
{
  namespace utils
  {      
    

    /** \brief Selects the sampling method.  
     *
     * @param method: 1-> LHS | 2-> Sobol (if available) | 3-> Uniform
     */
    template<class M>
    void samplePoints(M& xPoints, int method, randEngine& mtRandom);


    /** \brief Latin hypercube sampling. It is used to generate a
     * uniform Latin hypercube.
     * 
     * Latin hypercube sampling generate nA samples for nB dimensions.
     * Samples are generate by random permutations of a linearly
     * spaced vector with a random perturbation.
     *
     * Example:
     *
     *   1) Initial linspace => [1 2 3; 
     *                           1 2 3]
     *
     *   2) Random permutations => [3 1 2; 
     *                              2 3 1]
     *
     *   3) Random perturbation => [2.3  0.6  1.7; 
     *                              1.2  2.3  0.5]
     *
     *   4) Normalization => [0.767  0.200  0.567; 
     *                        0.400  0.767  0.167]
     *
     */
    template<class M>
    void lhs(M& Result,randEngine& mtRandom);

#if defined (USE_SOBOL)
    /** \brief Hypercube sampling based on Sobol sequences
     * It uses the external Sobol library. Thus it do not depend on
     * boost random.
     */
    template<class M>
    void sobol(M& result, long long int seed);
#endif

    /** \brief Uniform hypercube sampling
     * It is used to generate a set of uniformly distributed
     * samples in hypercube
     */
    template<class M>
    void uniformSampling(M& Result, randEngine& mtRandom);



    /** \brief Linearly spaced vector in [0,1] interval */
    template<class V>
    void linspace(V& Result);

    /** \brief Modify an array using ramdom permutations.
     *
     * It is used to generate a uniform Latin hypercube.
     * Equivalent to std::random_shuffle but using boost::random
     */
    template<class D>
    void randomPerms(D& arr, randEngine& mtRandom);

    //////////////////////////////////////////////////////////////////////
    // Implementations
    //////////////////////////////////////////////////////////////////////

    template<class D>
    void randomPerms(D& arr, randEngine& mtRandom)
    {
      typedef typename D::iterator iter;

      randInt sample(mtRandom, intUniformDist(0,arr.size()-1));
      for (iter it=arr.begin(); it!=arr.end(); ++it)
	iter_swap(arr.begin()+sample(),it);
    } // randomPerms 


    template<class M>
    void lhs(M& Result, randEngine& mtRandom)
    {
      randFloat sample( mtRandom, realUniformDist(0,1) );
      size_t nA = Result.size1();
      size_t nB = Result.size2();
      double ndA = static_cast<double>(nA);
      //  std::vector<int> perms(nA);
  
      for (size_t i = 0; i < nB; i++)
	{
	  // TODO: perms starts at 1. Check this
	  std::vector<int> perms = return_index_vector(nA);
	  randomPerms(perms, mtRandom);

	  for (size_t j = 0; j < nA; j++)
	    {		
	      Result(j,i) = ( static_cast<double>(perms[j]) - sample() ) / ndA;
	    }
	}
    }

#if defined (USE_SOBOL)
    template<class M>
    void sobol(M& result, long long int seed)
    {
      size_t nSamples = result.size1() + 1;  // First element of the Sobol sequence will be skipped (zeroes point)
      size_t nDims = result.size2();

      double *sobol_seq = i8_sobol_generate(nDims,nSamples,seed);

      std::copy(sobol_seq+nDims,sobol_seq+(nSamples*nDims),result.data().begin());  // Skip first element of sequence
    }
#endif

    template<class M>
    void uniformSampling(M& Result,
			randEngine& mtRandom)
    {
      randFloat sample( mtRandom, realUniformDist(0,1) );
      size_t nA = Result.size1();
      size_t nB = Result.size2();

      // TODO: Improve with iterators
      // std::generate(Result.begin2(),Result.end2(),sample);
      for (size_t i = 0; i < nA; i++)
      	for (size_t j = 0; j < nB; j++)
      	  Result(i,j) = sample();
    }

    template<class V>
    void linspace(V& Result)
    {
      size_t nA = Result.size();
      double ndA = static_cast<double>(nA);
      
      std::vector<int> perms = return_index_vector(nA);
      for (size_t j = 0; j < nA; j++)
	{		
	  Result(j) = static_cast<double>(perms[j])/ ndA;
	}
    }


    template<class M>
    void samplePoints(M& xPoints, int method,
		     randEngine& mtRandom)
    {
      if (method == 1) 
	{
	  FILE_LOG(logINFO) << "Latin hypercube sampling";
	  lhs(xPoints, mtRandom);
	}
      else if (method == 2)
	{
#if defined (USE_SOBOL)
	  FILE_LOG(logINFO) << "Sobol sampling";
	  sobol(xPoints, 0);
#else
	  FILE_LOG(logINFO) << "Latin hypercube sampling";
	  lhs(xPoints, mtRandom);
#endif
	}
      else
	{
	  FILE_LOG(logINFO) << "Uniform sampling";
	  uniformSampling(xPoints, mtRandom);
	}
    }


  } //namespace utils

} //namespace bayesopt

#endif
