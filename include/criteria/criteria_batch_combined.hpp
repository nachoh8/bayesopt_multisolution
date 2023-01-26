/**  \file criteria_combined.hpp \brief Abstract module for combined criteria */
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

#ifndef  _BATCH_CRITERIA_COMBINED_HPP_
#define  _BATCH_CRITERIA_COMBINED_HPP_

#include <boost/ptr_container/ptr_vector.hpp>
#include <boost/numeric/ublas/vector_proxy.hpp>
#include "criteria_combined.hpp"


namespace bayesopt
{
  /**\addtogroup CriteriaFunctions */
  //@{

  /// Abstract class for batch combined criteria functions
  /// The user must specify the number of points to be selected with each criterion.
  /// The sum of all points must be equal to the batch size
  /// Example:
  /// "crit_name": "cBComb(cEI, cLCBnorm)",
  /// "n_crit_params": 4,
  /// "crit_params": [1, 2, 1.0, 2.0] # first the num points for each criteria, then the parameters for each criteria
  class BatchCombinedCriteria: public CombinedCriteria
  {
  public:
    virtual void init(NonParametricProcess *proc) 
    {
      CombinedCriteria::init(proc);
      current_point = 0;
      n_points = 0;
      current_criteria = 0;
    };

    virtual ~BatchCombinedCriteria() {};
    
    double operator() (const vectord &x) {
      return mCriteriaList[current_criteria](x);
    }

    size_t nParameters() {
      return CombinedCriteria::nParameters() + mCriteriaList.size();
    }

    void setParameters(const vectord &theta) {
      using boost::numeric::ublas::subrange;
      const size_t nc = mCriteriaList.size();

      crit_npoints = vectori(nc);
      for (size_t i = 0; i < nc; ++i)
      {
        crit_npoints(i) = theta[i];
        FILE_LOG(logDEBUG) << "Criteria " << mCriteriaList[i].name() << " -> " << theta[i] << " points";
      }

      n_points = norm_1(crit_npoints);
      FILE_LOG(logDEBUG) << "Total points: " << n_points;
      FILE_LOG(logDEBUG) << "Params: " << subrange(theta, nc, theta.size());
      CombinedCriteria::setParameters(subrange(theta, nc, theta.size()));
    }

    void reset_batch() {
      FILE_LOG(logDEBUG) << "Reset batch";
      current_point = 0;
      current_criteria = 0;
    };

    virtual void update(const vectord &x) {
      if (x.size() == 0) {
        reset_batch();
      } else if (current_point < n_points) {
        FILE_LOG(logDEBUG) << "Last batch point (" << current_point+1 << ") -> criteria " << mCriteriaList[current_criteria].name();
        current_point++;
        for (size_t i = 0; i < mCriteriaList.size(); ++i) {
          if (current_point < crit_npoints[i]) {
            current_criteria = i;
            break;
          }
        }
        if (current_point < n_points) {
          FILE_LOG(logDEBUG) << "Next batch point (" << current_point+1 << ") -> criteria " << mCriteriaList[current_criteria].name();
        }
      } else {
        for (auto& crit : mCriteriaList) {
          crit.update(x);
        }
      }
    }

    std::string name() {return "cBComb";};
  private:
    int current_point;
    int n_points;
    vectori crit_npoints;
    size_t current_criteria;
  };

  //@}

} //namespace bayesopt


#endif
