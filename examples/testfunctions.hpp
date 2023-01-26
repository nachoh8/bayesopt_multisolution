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

#ifndef __TEST_FUNCTIONS_HPP__
#define __TEST_FUNCTIONS_HPP__

#include "bayesopt/bayesopt.hpp"
#include "specialtypes.hpp"
#include "randgen.hpp"



class ExampleOneD: public bayesopt::ContinuousModel
{
public:
  ExampleOneD(bayesopt::Parameters par, double noise = 0.0);

  double evaluateSample(const vectord& xin);
  bool checkReachability(const vectord &query)  {return true;};

  void printOptimal();
private:
  randEngine mEng;
  double mNoiseStd;
};

class Exponential2D: public bayesopt::ContinuousModel
{
public:
  Exponential2D(bayesopt::Parameters par);

  double evaluateSample(const vectord& xin);
  bool checkReachability(const vectord &query)  {return true;};

  void printOptimal();
};

class Easom: public bayesopt::ContinuousModel
{
public:
  Easom(bayesopt::Parameters par);

  double evaluateSample(const vectord& xin);
  bool checkReachability(const vectord &query)  {return true;};

  void printOptimal();
};


class HardOneD: public bayesopt::ContinuousModel
{
public:
  HardOneD(bayesopt::Parameters par);

  double gaussian(double x, double mu, double sigma);
  double evaluateSample(const vectord& xin);
  bool checkReachability(const vectord &query)  {return true;};

  void printOptimal();
private:
  randEngine mEng;
};



class BraninNormalized: public bayesopt::ContinuousModel
{
public:
  BraninNormalized(bayesopt::Parameters par);

  double evaluateSample( const vectord& xin);
  bool checkReachability(const vectord &query) {return true;};

  double branin(double x, double y);
  void printOptimal();
};

class BraninNormalizedNoise: public BraninNormalized
{
private:
  double variance;
  double sampleGaussian(double mu = 0, double sigma = 1);

public:
  BraninNormalizedNoise(bayesopt::Parameters par, double variance = 1.0);

  double evaluateSample( const vectord& xin);
  void testGaussian(int num_samples, double delta, int numBars);
};



class BraninPlusEasomNormalized: public bayesopt::ContinuousModel
{
public:
  BraninPlusEasomNormalized(bayesopt::Parameters par);

  double evaluateSample( const vectord& xin);
  bool checkReachability(const vectord &query) {return true;};

  double branineasom(double x, double y);
  void printOptimal();

};


class AckleyTest: public bayesopt::ContinuousModel
{
public:
  AckleyTest(bayesopt::Parameters par, size_t dim = 2);

  vectord getUpperBound();
  double evaluateSample( const vectord& xin);
  bool checkReachability(const vectord &query) {return true;};

  void printOptimal();
private:
  size_t mDim;
};

class Eggholder: public bayesopt::ContinuousModel
{
public:
  Eggholder(bayesopt::Parameters par);

  vectord getUpperBound();
  double evaluateSample( const vectord& xin);
  bool checkReachability(const vectord &query) {return true;};

  void printOptimal();
};

class ExampleCamelback: public bayesopt::ContinuousModel
{
public:
  ExampleCamelback(bayesopt::Parameters par);

  double evaluateSample( const vectord& x);
  bool checkReachability(const vectord &query) {return true;};

  void printOptimal();

};

class ExampleMichalewicz: public bayesopt::ContinuousModel
{
public:
  ExampleMichalewicz(size_t dim, bayesopt::Parameters par);

  double evaluateSample(const vectord& x);
  bool checkReachability(const vectord &query) {return true;};

  void printOptimal();

private:
  double mExp;
};


class Rosenbrock: public bayesopt::ContinuousModel
{
public:
  Rosenbrock(size_t dim, bayesopt::Parameters par);

  double evaluateSample(const vectord& x);
  bool checkReachability(const vectord &query) {return true;};

  void printOptimal();
};


class Schubert: public bayesopt::ContinuousModel
{
public:
  Schubert(bayesopt::Parameters par);

  double evaluateSample(const vectord& x);
  bool checkReachability(const vectord &query) {return true;};

  void printOptimal();
};



class ExampleHartmann6: public bayesopt::ContinuousModel
{
public:
  ExampleHartmann6(bayesopt::Parameters par);

  double evaluateSample( const vectord& xin);
  bool checkReachability(const vectord &query) {return true;};

  void printOptimal();

private:
  matrixd mA;
  vectord mC;
  matrixd mP;
};

#endif
