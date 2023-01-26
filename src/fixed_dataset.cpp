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

#include "fixed_dataset.hpp"

#include <boost/numeric/ublas/matrix_proxy.hpp>


namespace bayesopt
{

  FixedDataset::FixedDataset(const size_t n_points) : n_points(n_points), gMinIndex(0), gMaxIndex(0), Dataset() {};

  FixedDataset::FixedDataset(const size_t n_points, const matrixd& x, const vectord& y):
    n_points(n_points), gMinIndex(0), gMaxIndex(0), Dataset(x, y)
  {
  };

  void FixedDataset::addSample(const vectord &x, double y)
  {
    if (mY.size() == n_points) {
      FILE_LOG(logDEBUG) << "MAX SIZE REACHED: removing " << mX[0] << " -> " << mY[0];
      FILE_LOG(logDEBUG) << "MAX SIZE REACHED: adding " << x << " -> " << y;
      deleteOldestPointX(x);
      deleteOldestPointY(y);

      FILE_LOG(logDEBUG) << "MAX SIZE REACHED: mMinIdx: " << mMinIndex << ", mMaxIdx: " << mMaxIndex;
      if (mMinIndex == 0 || mMaxIndex == 0) { // the erased point was the min/mas
        for (size_t i=0; i<mY.size(); ++i) {
          Dataset::updateMinMax(i);
        }
      } else {
        mMinIndex--; mMaxIndex--; // update position
        Dataset::updateMinMax(mY.size()-1);
      }
    } else {
      Dataset::addSample(x, y);
    }
  }

  void FixedDataset::addSample(double y)
  {

    if (mY.size() == n_points) {
      deleteOldestPointY(y);
      
      if (mMinIndex == 0 || mMaxIndex == 0) { // the erased point was the min/mas
        for (size_t i=0; i<mY.size(); ++i) {
          Dataset::updateMinMax(i);
        }
      } else {
        mMinIndex--; mMaxIndex--; // update position
        Dataset::updateMinMax(mY.size()-1);
      }
    } else {
      Dataset::addSample(y);
    }
  }
} //namespace bayesopt
