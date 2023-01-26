/**  \file unscented_transformation.hpp \brief Unscented transformation */
/*
-------------------------------------------------------------------------
   This file is part of BayesOpt, an efficient C++ library for 
   Bayesian optimization.

   Copyright (C) 2011-2018 Ruben Martinez-Cantin <rmcantin@unizar.es>
 
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

#ifndef __UNSCENTED_TRANSFORM_HPP__
#define __UNSCENTED_TRANSFORM_HPP__

#include "bayesopt/parameters.hpp"
#include "surrogates/nonparametricprocess.hpp"
#include "criteria/criteria_functors.hpp"
#include "posteriors/posterior_model.hpp"

namespace bayesopt
{

  // Multivariate GaussHermite https://gist.github.com/markvdw/f9ca12c99484cf2a881e84cb515b86c8
  
  // Scaled unscented transformation with isotropic uncertainty from \cite Julier04
  // Formulation based on \cite vanderMerwe
  class IntegratedResponse
  {
  public:
    IntegratedResponse():_usingModel(true){};
    virtual ~IntegratedResponse(){};

    void setPosteriorModel(PosteriorModel* model)
    {
      _model = model;
      _usingModel = true;
    }

    void setProcess(NonParametricProcess* process)
    {
      _process = process;
      _usingModel = false;
    }

    double getPredictionMean(const vectord &query)
    {
      if(_usingModel)
        {
          return _model->getPredictionMean(query);
        }
      else
        {
          return _process->prediction(query)->mean();
        }
    }


    virtual double integratedOutcome(const vectord x) = 0;
    virtual double integratedCriteria(const vectord x) = 0;

  protected:
    PosteriorModel* _model;
    NonParametricProcess* _process;
    bool _usingModel;
  };

  class SigmaPointIntegration: public IntegratedResponse
  {
  public:
    SigmaPointIntegration(size_t dim, Parameters par)
      {
        _dim = dim;
      };
    
    virtual ~SigmaPointIntegration(){};

    double integratedOutcome(const vectord x)
    {
      double y = getPredictionMean(x);
      double sum = y * _w0;
      
      for(size_t i = 0; i < _dim; ++i)
        {
          vectord xip = x;
          xip[i] += _sigma_point;
          y = getPredictionMean(xip);
          sum += y * _wi;

          vectord xim = x;
          xim[i] -= _sigma_point;
          y = getPredictionMean(xim);
          sum += y * _wi;
        }

      return sum;
    }

    double integratedCriteria(const vectord x)
    {
      double y = _model->evaluateCriteria(x);
      double sum = y * _w0;
      
      for(size_t i = 0; i < _dim; ++i)
        {
          vectord xip = x;
          xip[i] += _sigma_point;
          y = _model->evaluateCriteria(xip);
          sum += y * _wi;

          vectord xim = x;
          xim[i] -= _sigma_point;
          y = _model->evaluateCriteria(xim);
          sum += y * _wi;
        }

      return sum;
    }

  protected:
    double _dim;
    double _w0, _wi;
    double _sigma_point;
  };


  // coef = lambda + L
  class IsotropicUnscentedTransform: public SigmaPointIntegration
  {
  public:
    IsotropicUnscentedTransform(size_t dim, Parameters par):
      SigmaPointIntegration(dim,par)
    {
      _alpha = par.unscented_alpha;
      _beta = par.unscented_beta;
      _kappa = par.unscented_kappa;
      _sigma2 = par.input_noise_variance;

      double coef = _alpha * _alpha * (_dim + _kappa);

      _w0 = (coef - dim) / coef;
      _wi = 1/(2 * coef);
      _sigma_point = sqrt(coef * _sigma2);
      _model = NULL;
      _process = NULL;
    };
    virtual ~IsotropicUnscentedTransform(){};
    
  private:
    IsotropicUnscentedTransform();
    
    double _alpha, _beta, _kappa;
    double _sigma2;
  };

  // coef = h * h
  class CentralDifferenceTransform: public SigmaPointIntegration
  {
  public:
    CentralDifferenceTransform(size_t dim, Parameters par):
      SigmaPointIntegration(dim,par)
    {
      double coef = 3;

      _w0 = (coef - dim) / coef;
      _wi = 1/(2 * coef);
      _sigma_point = sqrt(coef * _sigma2);
      _model = NULL;
    };
    virtual ~CentralDifferenceTransform(){};
    
  private:
    CentralDifferenceTransform();
    
    double _sigma2;
  };

}

#endif
