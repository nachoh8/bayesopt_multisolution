/**  \file displaycrit.hpp \brief Plots the evolution (nonparametric
     process, criteria or contour plots) of 1D and 2D problems. */
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

#ifndef DISPLAYCRIT_H
#define DISPLAYCRIT_H

#include "bayesopt/bayesopt.hpp"
#include "matplotpp.h"  

namespace bayesopt
{
  namespace utils
  {      
    enum RunningStatus
      {
	RUN, STEP, STOP, NOT_READY
      };

    class DisplayCrit1D :public MatPlot
    { 
    private:
      RunningStatus status;
      size_t state_ii;
      ContinuousModel* bopt_model;
      std::vector<double> _x,_z,_bx;
      std::vector<double> lx,ly, hist;

    public:
      DisplayCrit1D();
      void init(ContinuousModel* bopt, size_t dim);
      void setSTEP();
      void toogleRUN();
      void DISPLAY();
      
      int mNumSamples;
    };

  } //namespace utils

} //namespace bayesopt

#endif /* DISPLAYCRIT_H */

