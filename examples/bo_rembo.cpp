/*
-------------------------------------------------------------------------
   This file is part of BayesOpt, an efficient C++ library for 
   Bayesian optimization.

   Copyright (C) 2011-2013 Ruben Martinez-Cantin <rmcantin@unizar.es>
 
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

#define _USE_MATH_DEFINES
#include <cmath>
#include "testfunctions.hpp"
#include "randgen.hpp"

class BraninRembo: public bayesopt::ContinuousModel
{
public:
  BraninRembo(bopt_params par, size_t xind, size_t yind):
    ContinuousModel(2,par),mA(1000,2) 
  {
    mXind = xind;
    mYind = yind;

    randNFloat sample(mEngine,normalDist(0,1));
    for (size_t ii = 0; ii < 1000; ++ii)
      {
	for (size_t jj = 0; jj< 2; ++jj)
	  {
	    mA(ii,jj) = sample();
	  } 
     }
  }

  double evaluateSample(const vectord& xin)
  {
    if (xin.size() != 2)
      {
	std::cout << "WARNING: This only works for 2D inputs." << std::endl
		  << "WARNING: Using only first two components." << std::endl;
      }
    
    vectord zin = prod(mA,xin);

    for (size_t ii = 0; ii < zin.size(); ++ii)
      {
	if (zin(ii) < 0.0) zin(ii) = 0.0;
	if (zin(ii) > 1.0) zin(ii) = 1.0;	
      }
    return branin_norm(zin(mXind),zin(mYind));
  }

  double branin_norm(double x, double y)
  {
    x = x * 15.0 - 5.0;
    y = y * 15.0;
    
    return branin(x,y);
  }

  double branin(double x, double y)
  {
    return sqr(y-(5.1/(4*sqr(M_PI)))*sqr(x)
	       +5*x/M_PI-6)+10*(1-1/(8*M_PI))*cos(x)+10;
  };

  bool checkReachability(const vectord &query)
  {return true;};

  inline double sqr( double x ){ return x*x; };

  void printOptimal()
  {
    vectord sv(2);  
    sv(0) = 0.1238938; sv(1) = 0.818333;
    std::cout << "Solutions: " << sv << "->" 
	      << branin_norm(sv(0),sv(1)) << std::endl;
    sv(0) = 0.5427728; sv(1) = 0.151667;
    std::cout << "Solutions: " << sv << "->" 
	      << branin_norm(sv(0),sv(1)) << std::endl;
    sv(0) = 0.961652; sv(1) = 0.1650;
    std::cout << "Solutions: " << sv << "->" 
	      << branin_norm(sv(0),sv(1)) << std::endl;
  }
  
  //TODO: Make it priv
public:
  matrixd mA;
  size_t mXind, mYind;
};


int main(int nargs, char *args[])
{
  size_t nrembo = 10;
  double best_y_res = 1e10;
  vectord best_res(2);

  bopt_params par = initialize_parameters_to_default();
  par.n_iterations = 190;
  //  par.surr_name = "sStudentTProcessNIG";
  par.verbose_level = 1;
  par.noise = 1e-10;
  
  vectord lb = svectord(2,-std::sqrt(2.0));
  vectord ub = svectord(2,std::sqrt(2.0));

  for (size_t ii = 0; ii < nrembo; ++ii)
    {
      par.random_seed = ii*10;
      BraninRembo branin(par,150,237);
      vectord result(2);
      branin.setBoundingBox(lb,ub);
      branin.optimize(result);
      vectord hd_res = prod(branin.mA,result);
      vectord ld_res(2);
      ld_res(0) = hd_res(150); ld_res(1) = hd_res(237);
      double y_res = branin.branin_norm(ld_res(0),ld_res(1));
      if (y_res < best_y_res)
	{
	  best_y_res = y_res;
	  best_res = ld_res;
	}
      branin.printOptimal();
    }
  std::cout << "Result: " << best_res << "->" 
	    << best_y_res << std::endl;

  return 0;
}
