/** \file dataset.hpp
    \brief Dataset model */
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


#ifndef __LIMITED_DATASET_HPP__
#define __LIMITED_DATASET_HPP__

#include "log.hpp"
#include "specialtypes.hpp"
#include "ublas_extra.hpp"
#include "dataset.hpp"
#include <boost/numeric/ublas/vector_proxy.hpp>

namespace ublas = boost::numeric::ublas;

namespace bayesopt
{

  /** \addtogroup NonParametricProcesses */
  /**@{*/

  /** \brief Dataset model with a limit of N points to deal with the vector (real) based datasets */
  class FixedDataset : public Dataset
  {
  public:
    FixedDataset(const size_t n_points);
    FixedDataset(const size_t n_points, const matrixd& x, const vectord& y);

    void addSample(const vectord &x, double y); // to override
    void addSample(double y); // to override
    
    vectord getPointAtGlobalMinimum() const;
    double getValueAtGlobalMinimum() const;
    double getValueAtGlobalMaximum() const;

    size_t getTotalNSamples() const;

  private:
    void deleteOldestPointX(const vectord newX);
    void deleteOldestPointY(const double newY);

    size_t n_points;

    vecOfvec deletedXs;
    vectord deletedYs;
    size_t gMinIndex, gMaxIndex;
  };

  inline vectord FixedDataset::getPointAtGlobalMinimum() const { 
    double sY = mY[mMinIndex];
    if (deletedYs.size() > 0) {
      double gY = deletedYs[gMinIndex];
      if (sY < gY) {
        return mX[mMinIndex];
      } else {
        return deletedXs[gMinIndex];
      }
    } else {
      return mX[mMinIndex];
    }
  };
  inline double FixedDataset::getValueAtGlobalMinimum() const {
    double sY = mY[mMinIndex];
    if (deletedYs.size() > 0) {
      double gY = deletedYs[gMinIndex];
      if (sY < gY) {
        return sY;
      } else {
        return gY;
      }
    } else {
      return sY;
    }
  };
  inline double FixedDataset::getValueAtGlobalMaximum() const {
    double sY = mY[mMaxIndex];
    if (deletedYs.size() > 0) {
      double gY = deletedYs[gMaxIndex];
      if (sY > gY) {
        return sY;
      } else {
        return gY;
      }
    } else {
      return sY;
    }
  };
  
  inline void FixedDataset::deleteOldestPointX(const vectord newX) {    
    deletedXs.push_back(mX[0]);
    mX.erase(mX.begin());

    mX.push_back(newX);
  }

  inline void FixedDataset::deleteOldestPointY(const double newY) {    
    utils::append(deletedYs, mY[0]);
    mY = ublas::subrange(mY, 1, mY.size());

    utils::append(mY, newY);
    size_t idx = deletedYs.size() - 1;

    if ( deletedYs(gMinIndex) > deletedYs(idx) )       gMinIndex = idx;
    else if ( deletedYs(gMaxIndex) < deletedYs(idx) )  gMaxIndex = idx;
  }

  inline size_t FixedDataset::getTotalNSamples() const {
    return getNSamples() + deletedYs.size();
  }

  /**@}*/

} //namespace bayesopt

#endif
