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

#ifndef  _MULTISOLUTION_CRITERIA_HPP_
#define  _MULTISOLUTION_CRITERIA_HPP_

#include "criteria_combined.hpp"

namespace bayesopt
{
  /**\addtogroup CriteriaFunctions */
  //@{

  /// Abstract class for multisolution method
  /// cMS(cluster_criteria, cBComb(...))
  /// Example:
  /// "crit_name": "cMS(cClusterEI, cBComb(cEI, cLCBnorm))",
  /// "n_crit_params": 5,
  /// "crit_params": [1, 1, 2, 1.0, 2.0]
  class MultisolutionCriteria: public CombinedCriteria
  {
  public:
    virtual void init(NonParametricProcess *proc) 
    {
      CombinedCriteria::init(proc);
      current_criteria = 1;
    };

    virtual ~MultisolutionCriteria() {};
    
    double operator() (const vectord &x) {
      return mCriteriaList[current_criteria](x);
    }

    virtual void update(const vectord &x) {
      if (x.size() == 1 && x[0] < 0) { // command to change criteria
        current_criteria = x[0] * -1 - 1;
        FILE_LOG(logDEBUG) << "cMS: set active criteria to " << current_criteria;
      } else {
        mCriteriaList[current_criteria].update(x);
      }
    }

    std::string name() {return "cMS";};

    Criteria* getCurrentCriteria() {
        return &mCriteriaList[current_criteria];
    }

  private:
    size_t current_criteria;
  };

  //@}

} //namespace bayesopt


#endif
