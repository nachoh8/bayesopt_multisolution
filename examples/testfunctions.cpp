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
#include <algorithm>
#include <boost/math/constants/constants.hpp>
#include <boost/numeric/ublas/assignment.hpp>
#include "testfunctions.hpp"

double sqr( double x ){ return x*x; };

//////////////////////////////////////////////////////////////////////

ExampleOneD::ExampleOneD(bayesopt::Parameters par, double noise):
  ContinuousModel(1,par), mNoiseStd(noise) {}

double ExampleOneD::evaluateSample(const vectord& xin)
{

  if (xin.size() != 1)
    {
      std::cout << "WARNING: This only works for 1D inputs." << std::endl
		<< "WARNING: Using only first component." << std::endl;
    }
  
  double x = xin(0);
  double noise = 0.0;

  if (mNoiseStd > 0.0)
    {
       normalDist normal(0,0.02);
       randNFloat sample(mEng, normal);
       noise = sample();
    }

  return (x-0.3)*(x-0.3) + sin(20*x)*0.2 + noise;
};

void ExampleOneD::printOptimal()
{
  std::cout << "Optimal:" << 0.23719 << std::endl;
}

//////////////////////////////////////////////////////////////////////

Exponential2D::Exponential2D(bayesopt::Parameters par):
  ContinuousModel(2,par) {}

double Exponential2D::evaluateSample(const vectord& xin)
{
  if (xin.size() != 2)
    {
      std::cout << "WARNING: This only works for 2D inputs." << std::endl
		<< "WARNING: Using only first component." << std::endl;
    }
  
  double x1 = xin(0)*20 - 2;
  double x2 = xin(1)*20 - 2;
  return x1*std::exp(-x1*x1-x2*x2);
};

void Exponential2D::printOptimal()
{
  std::cout << "Optimal:" << 0 << std::endl;
}

//////////////////////////////////////////////////////////////////////

Easom::Easom(bayesopt::Parameters par):
  ContinuousModel(2,par) {}

double Easom::evaluateSample(const vectord& xin)
{
  if (xin.size() != 2)
    {
      std::cout << "WARNING: This only works for 2D inputs." << std::endl
		<< "WARNING: Using only first component." << std::endl;
    }
  
  double x1 = xin(0)*200 - 100;
  double x2 = xin(1)*200 - 100;
  double x1p = x1 - boost::math::constants::pi<double>();;
  double x2p = x2 - boost::math::constants::pi<double>();;

  return -cos(x1)*cos(x2)*std::exp(-x1p*x1p-x2p*x2p);
};

void Easom::printOptimal()
{
  std::cout << "Optimal:" << -1 << " at [" << boost::math::constants::pi<double>() 
	    << "," << boost::math::constants::pi<double>() << "]" << std::endl;
}


//////////////////////////////////////////////////////////////////////

HardOneD::HardOneD(bayesopt::Parameters par):
  ContinuousModel(1,par) {}

double HardOneD::gaussian(double x, double mu, double sigma)
{
  double diff = x-mu;
  return exp(-(diff*diff)/(sigma*sigma));
};

double HardOneD::evaluateSample(const vectord& xin)
{
  normalDist normal(0,1e-2);
  randNFloat sample(mEng, normal);
  if (xin.size() != 1)
    {
      std::cout << "WARNING: This only works for 1D inputs." << std::endl
		<< "WARNING: Using only first component." << std::endl;
    }
  
  double x = xin(0);
  return -(2*gaussian(x,0.1,0.1) + 4*gaussian(x,0.9,0.01) + sample());
};

void HardOneD::printOptimal()
{
  std::cout << "Optimal:" << 0.9 << std::endl;
}

//////////////////////////////////////////////////////////////////////

BraninPlusEasomNormalized::BraninPlusEasomNormalized(bayesopt::Parameters par):
  ContinuousModel(2,par) {}

double BraninPlusEasomNormalized::evaluateSample( const vectord& xin)
{
  if (xin.size() != 2)
    {
      std::cout << "WARNING: This only works for 2D inputs." << std::endl
		<< "WARNING: Using only first two components." << std::endl;
    }
  
  double x = xin(0) * 15 - 5;
  double y = xin(1) * 15;
    
  return branineasom(x,y);
}

double BraninPlusEasomNormalized::branineasom(double x, double y)
{
  const double pi = boost::math::constants::pi<double>();
  double branin = sqr(y-(5.1/(4*pi*pi))*sqr(x)
		       +5*x/pi-6)+10*(1-1/(8*pi))*cos(x)+10;

  double x1p = x-2*pi;
  double x2p = y-2*pi;
  double easom = -x*y*std::exp(-x1p*x1p-x2p*x2p);

  return branin+5*easom;
};

void BraninPlusEasomNormalized::printOptimal()
{
  vectord sv(2);  
  sv(0) = 0.1238938; sv(1) = 0.818333;
  std::cout << "Local Solutions: " << sv << "->" 
	    << evaluateSample(sv) << std::endl;
  sv(0) = 0.5427728; sv(1) = 0.151667;
  std::cout << "Local Solutions: " << sv << "->" 
	    << evaluateSample(sv) << std::endl;
  sv(0) = 0.961652; sv(1) = 0.1650;
  std::cout << "Local Solutions: " << sv << "->" 
	    << evaluateSample(sv) << std::endl;
  sv(0) = 0.7522; sv(1) = 0.4188;
  std::cout << "Global Solutions: " << sv << "->" 
	    << evaluateSample(sv) << std::endl;


}


//////////////////////////////////////////////////////////////////////

BraninNormalizedNoise::BraninNormalizedNoise(bayesopt::Parameters par, double variance):
  BraninNormalized(par), variance(variance) {}

double BraninNormalizedNoise::evaluateSample( const vectord& xin)
{
  if (xin.size() != 2)
    {
      std::cout << "WARNING: This only works for 2D inputs." << std::endl
		<< "WARNING: Using only first two components." << std::endl;
    }
  
  double x = xin(0) * 15 - 5;
  double y = xin(1) * 15;
    
  return sampleGaussian(branin(x,y), sqrt(variance));
}


double BraninNormalizedNoise::sampleGaussian(double mu, double sigma)
{
  // Box-Muller Transform
  const double epsilon = std::numeric_limits<double>::min();
  const double pi = boost::math::constants::pi<double>();
  double u1,u2;
  do
    {
      u1 = rand() * (1.0 / RAND_MAX);
      u2 = rand() * (1.0 / RAND_MAX);
    }
  while (u1 <= epsilon);

  double z0 = sqrt(-2.0 * log(u1)) * cos(2*pi*u2);
  return z0 * sigma + mu;
}

void BraninNormalizedNoise::testGaussian(int num_samples, double delta, int numBars)
{
  numBars = (numBars / 2) * 2;
  if(numBars < 2) numBars = 2;

  std::vector<int> reps(numBars, 0);

  for(int i=0; i<num_samples; i++)
    {
      double v = sampleGaussian();
    
      int index = (int)(v/delta) + (numBars/2);
      if(v < 0)
        {
          index -= 1;
        }
      if(index >= 0 && index < reps.size())
        {
          reps[index]++;
        }
    }

  double maxRep = 0;
  for(int i=0; i<reps.size(); i++)
    {
      if(reps[i] > maxRep)
        {
          maxRep = reps[i];
        } 
    }

  
  std::cout << "Box-Muller Transform Histogram" << std::endl;
  for(int r=0; r<reps.size(); r++){
    int n = ((double)reps[r] / maxRep)*10;
      for(int i=0; i<n; i++){
        std::cout << "*";
      }
      std::cout << " = " << reps[r] << std::endl;
  }
}


//////////////////////////////////////////////////////////////////////

BraninNormalized::BraninNormalized(bayesopt::Parameters par):
  ContinuousModel(2,par) {}

double BraninNormalized::evaluateSample( const vectord& xin)
{
  if (xin.size() != 2)
    {
      std::cout << "WARNING: This only works for 2D inputs." << std::endl
		<< "WARNING: Using only first two components." << std::endl;
    }
  
  double x = xin(0) * 15 - 5;
  double y = xin(1) * 15;
  
  return branin(x,y);
}

double BraninNormalized::branin(double x, double y)
{
  const double pi = boost::math::constants::pi<double>();
  return sqr(y-(5.1/(4*pi*pi))*sqr(x)
	     +5*x/pi-6)+10*(1-1/(8*pi))*cos(x)+10;
};

void BraninNormalized::printOptimal()
{
  vectord sv(2);  
  sv(0) = 0.1238938; sv(1) = 0.818333;
  std::cout << "Solutions: " << sv << "->" 
	    << evaluateSample(sv) << std::endl;
  sv(0) = 0.5427728; sv(1) = 0.151667;
  std::cout << "Solutions: " << sv << "->" 
	    << evaluateSample(sv) << std::endl;
  sv(0) = 0.961652; sv(1) = 0.1650;
  std::cout << "Solutions: " << sv << "->" 
	    << evaluateSample(sv) << std::endl;
}

//////////////////////////////////////////////////////////////////////
AckleyTest::AckleyTest(bayesopt::Parameters par, size_t dim):
  ContinuousModel(dim,par), mDim(dim) {}

vectord AckleyTest::getUpperBound()
{
  svectord v(mDim,32.768);
  return v;
}

double AckleyTest::evaluateSample( const vectord& xin)
{
  double n = static_cast<double>(xin.size());
  const double a = 20;
  const double b = 0.2;
  const double c = boost::math::constants::two_pi<double>();;
  double ccx = 0.0;
  double sx2 = 0.0;

  for(vectord::const_iterator it = xin.begin(); it != xin.end(); ++it)
    {
      ccx += std::cos(*it * c);
      sx2 += *it * *it;      
    }

  return -a*std::exp(-b*std::sqrt(sx2/n)) - std::exp(ccx/n) + a + std::exp(1);
}


void AckleyTest::printOptimal()
{
  zvectord sv(mDim);  
  std::cout << "Solutions: " << sv << "->" 
	    << evaluateSample(sv) << std::endl;
}

///////////////////////////////////////////////////////////////////////////////
Eggholder::Eggholder(bayesopt::Parameters par):
  ContinuousModel(2,par) {}

vectord Eggholder::getUpperBound()
{
  svectord v(2,512);
  return v;
}

double Eggholder::evaluateSample( const vectord& xin)
{
  if (xin.size() != 2)
    {
      std::cout << "WARNING: This only works for 2D inputs." << std::endl
		<< "WARNING: Using only first two components." << std::endl;
    }
  double x1 = xin(0);
  double x2 = xin(1);

  double term1 = -(x2+47) * sin(sqrt(abs(x2+x1/2+47)));
  double term2 = -x1 * sin(sqrt(abs(x1-(x2+47))));

 return term1 + term2;
}


void Eggholder::printOptimal()
{
  vectord sv(2);
  sv(0) = 512; sv(1) = 404.2319;
  std::cout << "Solutions: " << sv << "->" 
	    << evaluateSample(sv) << std::endl;
}


///////////////////////////////////////////////////////////////////////////////

ExampleCamelback::ExampleCamelback(bayesopt::Parameters par):
  ContinuousModel(2,par) {}

double ExampleCamelback::evaluateSample( const vectord& x)
{
  if (x.size() != 2)
    {
      std::cout << "WARNING: This only works for 2D inputs." << std::endl
		<< "WARNING: Using only first two components." << std::endl;
    }
  double x1_2 = x(0)*x(0);
  double x2_2 = x(1)*x(1);
  
  double tmp1 = (4 - 2.1 * x1_2 + (x1_2*x1_2)/3) * x1_2;
  double tmp2 = x(0)*x(1);
  double tmp3 = (-4 + 4 * x2_2) * x2_2;
  return tmp1 + tmp2 + tmp3;
}
void ExampleCamelback::printOptimal()
{
  vectord sv(2);  
  sv(0) = 0.0898; sv(1) = -0.7126;
  std::cout << "Solutions: " << sv << "->" 
	    << evaluateSample(sv) << std::endl;
  sv(0) = -0.0898; sv(1) = 0.7126;
  std::cout << "Solutions: " << sv << "->" 
	    << evaluateSample(sv) << std::endl;
}

///////////////////////////////////////////////////////////////////////////////

ExampleMichalewicz::ExampleMichalewicz(size_t dim, bayesopt::Parameters par):
  ContinuousModel(dim,par) 
{
  mExp = 10;
}

double ExampleMichalewicz::evaluateSample(const vectord& x)
{
  size_t dim = x.size();
  double sum = 0.0;
  
  for(size_t i = 0; i<dim; ++i)
    {
      double frac = x(i)*x(i)*(i+1);
      frac /= boost::math::constants::pi<double>();;
      sum += std::sin(x(i)) * std::pow(std::sin(frac),2*mExp);
    }
  return -sum;
}

void ExampleMichalewicz::printOptimal()
{
  std::cout << "Solutions: " << std::endl;
  std::cout << "f(x)=-1.8013 (n=2)"<< std::endl;
  std::cout << "f(x)=-4.687658 (n=5)"<< std::endl;
  std::cout << "f(x)=-9.66015 (n=10);" << std::endl;
}


///////////////////////////////////////////////////////////////////////////////

Rosenbrock::Rosenbrock(size_t dim, bayesopt::Parameters par):
  ContinuousModel(dim,par){}

double Rosenbrock::evaluateSample(const vectord& x)
{
  size_t dim = x.size();
  double sum = 0.0;
  
  for(size_t i = 0; i<dim-1; ++i)
    {
      double xi = x(i)*15-5;
      double xn = x(i+1)*15-5;
      sum += 100*sqr(xn+sqr(xi)) + sqr(xi-1);
    }
  return sum;
}

void Rosenbrock::printOptimal()
{
  std::cout << "Solution: " << std::endl;
  std::cout << "f(x)=0 at x=(1,..,1)"<< std::endl;
}
///////////////////////////////////////////////////////////////////////////////

Schubert::Schubert(bayesopt::Parameters par):
  ContinuousModel(2,par){}

double Schubert::evaluateSample(const vectord& x)
{
  double x1 = x(0)*10.24-5.12;
  double x2 = x(1)*10.24-5.12;

  double sum1 = 0.0;
  double sum2 = 0.0;
  
  for(size_t i = 1; i<=5; ++i)
    {
      sum1 += i * std::cos((i+1)*x1+i);
      sum2 += i * std::cos((i+1)*x2+i);
    }
  return sum1*sum2;
}

void Schubert::printOptimal()
{
  std::cout << "Solutions [many global minima]: " << std::endl;
  std::cout << "f(x)=-186.7309"<< std::endl;
}


///////////////////////////////////////////////////////////////////////////////

ExampleHartmann6::ExampleHartmann6(bayesopt::Parameters par):
  ContinuousModel(6,par), mA(4,6), mC(4), mP(4,6)
{
  mA <<= 10.0,   3.0, 17.0,   3.5,  1.7,  8.0,
    0.05, 10.0, 17.0,   0.1,  8.0, 14.0,
    3.0,   3.5,  1.7,  10.0, 17.0,  8.0,
    17.0,   8.0,  0.05, 10.0,  0.1, 14.0;
  
  mC <<= 1.0, 1.2, 3.0, 3.2;
  
  mP <<= 0.1312, 0.1696, 0.5569, 0.0124, 0.8283, 0.5886,
    0.2329, 0.4135, 0.8307, 0.3736, 0.1004, 0.9991,
    0.2348, 0.1451, 0.3522, 0.2883, 0.3047, 0.6650,
    0.4047, 0.8828, 0.8732, 0.5743, 0.1091, 0.0381;
}

double ExampleHartmann6::evaluateSample( const vectord& xin)
{
  double y = 0.0;
  for(size_t i=0; i<4; ++i)
    {
      double sum = 0.0;
      for(size_t j=0;j<6; ++j)
	{
	  double val = xin(j)-mP(i,j);
	  sum -= mA(i,j)*val*val;
	}
      y -= mC(i)*std::exp(sum);
    }
  return y;
};

void ExampleHartmann6::printOptimal()
{
  vectord sv(6);
  sv <<= 0.20169, 0.150011, 0.476874, 0.275332, 0.311652, 0.6573;
  
  std::cout << "Solution: " << sv << "->" 
	    << evaluateSample(sv) << std::endl;
}
