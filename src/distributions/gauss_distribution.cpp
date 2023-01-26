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

#include "distributions/gauss_distribution.hpp"
#include <boost/math/special_functions/factorials.hpp>
#include <boost/math/distributions/normal.hpp>
#include<boost/math/distributions.hpp>

namespace bayesopt
{

  GaussianDistribution::GaussianDistribution(randEngine& eng): 
    ProbabilityDistribution(eng)
  {
    mean_ = 0.0;  std_ = 1.0;
  }


  GaussianDistribution::~GaussianDistribution(){}

  double GaussianDistribution::negativeExpectedImprovement(double min,
							   size_t g)
  {
  
    using boost::math::factorial;

    const double diff = min - mean_;
    const double z = diff / std_;
    const double pdf_z = boost::math::pdf(d_,z);
    const double cdf_z = boost::math::cdf(d_,z);
  
    if (g == 1)
      return -1.0 * ( diff * cdf_z + std_ * pdf_z );
    else
      {
	const double fg = factorial<double>(g);

	double Tm2 = cdf_z;
	double Tm1 = pdf_z;
	double sumEI = pow(z,static_cast<double>(g))*Tm2 - g*pow(z,static_cast<double>(g-1))*Tm1;

	for (size_t ii = 2; ii < g; ++ii) 
	  {
	    double Tact = (ii-1)*Tm2 - pdf_z*pow(z,static_cast<double>(ii-1));
	    sumEI += pow(-1.0,static_cast<double>(ii))* 
	      (fg / ( factorial<double>(ii)*factorial<double>(g-ii) ) )*
	      pow(z,static_cast<double>(g-ii))*Tact;
	  
	    //roll-up
	    Tm2 = Tm1;   Tm1 = Tact;
	  }
	return -1.0 * pow(std_,static_cast<double>(g)) * sumEI;
      }
  
  }  // negativeExpectedImprovement

  double GaussianDistribution::lowerConfidenceBound(double beta)
  {    
    return mean_ - beta*std_;
  }  // lowerConfidenceBound


  double GaussianDistribution::negativeProbabilityOfImprovement(double min,
								double epsilon)
  {
    return -cdf(d_,(min - mean_ + epsilon)/std_);
  }  // negativeProbabilityOfImprovement


  double GaussianDistribution::sample_query()
  { 
    randNFloat sample(mtRandom,normalDist(mean_,std_));
    return sample();
  } // sample_query
  
  double GaussianDistribution::quantile(double x)
  {
    return boost::math::quantile(d_, x);
  }

}
