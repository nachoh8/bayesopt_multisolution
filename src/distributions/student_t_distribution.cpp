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

#include "distributions/student_t_distribution.hpp"


namespace bayesopt
{

  StudentTDistribution::StudentTDistribution(randEngine& eng): 
    ProbabilityDistribution(eng), d_(2)
  {
    mean_ = 0.0;  std_ = 1.0; dof_ = 2;
  }

  StudentTDistribution::~StudentTDistribution(){}


  double StudentTDistribution::negativeExpectedImprovement(double min,
							   size_t g)
  {
    const double diff = min - mean_;
    const double z = diff / std_;
  
    assert((g == 1) && "Students t EI with exponent not yet supported.");
    return -(diff * boost::math::cdf(d_,z) 
	     + (dof_*std_+z*diff)/(dof_-1) * boost::math::pdf(d_,z) ); 
  }  // negativeExpectedImprovement


  double StudentTDistribution::lowerConfidenceBound(double beta)
  {    
    return mean_ - beta*std_/sqrt(static_cast<double>(dof_));
  }  // lowerConfidenceBound

  double StudentTDistribution::negativeProbabilityOfImprovement(double min,
								double epsilon)
  {  
    return -cdf(d_,(min - mean_ + epsilon)/std_);
  }  // negativeProbabilityOfImprovement


  double StudentTDistribution::sample_query()
  { 
    double n = static_cast<double>(dof_);
    randNFloat normal(mtRandom,normalDist(mean_,std_));
    randGFloat gamma(mtRandom,gammaDist(n/2.0));
    return normal() / sqrt(2*gamma()/n);
  }  // sample_query

} //namespace bayesopt
