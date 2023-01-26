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


#include "displaycrit.hpp"
#include "dataset.hpp"
#include "distributions/prob_distribution.hpp"

namespace bayesopt
{
  namespace utils
  {      
    const double bucket_size = 0.01;

    DisplayCrit1D::DisplayCrit1D(): MatPlot()
    {
      status = NOT_READY;
      int number_of_buckets = static_cast<int>(ceil(1.0 / bucket_size)); 
      for (size_t i= 0; i < number_of_buckets; i++) 
	{
	  hist.push_back(0.0);
	}
      
      mNumSamples = 250;
    }

    void DisplayCrit1D::init(ContinuousModel* bopt, size_t dim)
    {
      if (dim != 1) 
	{ 
	  throw std::invalid_argument("Display only works for 1D problems"); 
	}

      bopt_model = bopt;

      // Query points (_x) to plot values and query points (_bx) to plot histogram
      int n=mNumSamples;
      _x = linspace(0,1,n);
      int number_of_buckets = static_cast<int>(ceil(1.0 / bucket_size));
      _bx = linspace(0,1,number_of_buckets);
      
      // Function true value
      _z = _x;
      vectord q(1);
      for(size_t i=0; i<n; ++i)
	{
	  q(0) = _x[i];                                                 // Query
	  _z[i] = bopt_model->evaluateSample(q);                //Target function true value
	}
      
      status = STOP;
    };

    void DisplayCrit1D::setSTEP()
    {
      if (status != NOT_READY)
	{
	  status = STEP;
	}
    };

    void DisplayCrit1D::toogleRUN()
    {
      if (status != NOT_READY)
	{
	  if(status != RUN)
	    {
	      status = RUN;
	    }
	  else
	    {
	      status = STOP;
	    }
	}
    }

    void DisplayCrit1D::DISPLAY()
    {
      if (status != NOT_READY)
        {
          size_t nruns = bopt_model->getParameters ()->n_iterations;
          if ((status != STOP) && (state_ii < nruns))
            {
              // We are moving. Next iteration
	      ++state_ii;
	      bopt_model->stepOptimization(); 
	      const double res = bopt_model->getData()->getLastSampleY();
	      const vectord last = bopt_model->getData()->getLastSampleX();
	      ly.push_back(res);
	      lx.push_back(last(0));
	      
	      if (status == STEP) { status = STOP; }

	      int bucket = static_cast<int>(floor(last(0) / bucket_size));
	      hist[bucket] += 1.0; 
            }
          
	  // We compute the prediction, true value and criteria at 1000 points
	  int n=mNumSamples;
          
          // Data to display GP posterior
          std::vector<double> y,su,sl;
	  y = _x;  su = _x; sl = _x;
          
          // Data to display criteria
	  std::vector<double> c,cTarget,cNormalTemp,cLowTemp,cHighTemp;
	  c = _x;
          cTarget = _x;
          cNormalTemp = _x;
          cLowTemp = _x;
          cHighTemp = _x;
          std::vector<double> h;
          // histogram of criteria samples
          h = std::vector<double>(_bx.size(), 0);

	  // Query functions at the n points
	  vectord q(1);
          for (size_t i = 0; i < n; ++i)
            {
              q (0) = _x[i];
              // Display posterior GP
              ProbabilityDistribution* pd = bopt_model->getPrediction(q);
	      y[i] = pd->mean();                                //Expected value
	      su[i] = y[i] + 2*pd->std();                       //Upper bound (95 %)
	      sl[i] = y[i] - 2*pd->std();                       //Lower bound (95 %)
              
              // Evaluate criteria
              double value = bopt_model->evaluateCriteria (q);

              if (i % 100 == 0) std::cout << "iter:" << i << std::endl;

              // Raw criteria value and different modifications
              c[i] = value;

              cTarget[i] = bopt_model->criteriaCorrection (value); //modifyCriteria(value, true);
              
              ///*
              cNormalTemp[i] = bopt_model->modifyCriteria (value, false, false, true, 0.5);
              cLowTemp[i] = bopt_model->modifyCriteria (value, false, false, true, 0.1);
              cHighTemp[i] = bopt_model->modifyCriteria (value, false, false, true, 1.0);
              // */
            }

          size_t init = bopt_model->cSampler->mBurnout.size () * 0.1;
          for (size_t i = init; i < bopt_model->cSampler->mBurnout.size (); i++)
            {
              double particle = bopt_model->cSampler->mBurnout[i][0];
              int bucket = static_cast<int> (floor (particle / bucket_size));
              h[bucket] += 1.0;
            }

          // Normalize histogram and target criteria
          
          float maxh = 1;
          float mintarget = 0;
          for (size_t i = 0; i < h.size (); i++)
            {
              if (h[i] > maxh) maxh = h[i];
            }
          for (size_t i = 0; i < cTarget.size (); i++)
            {
              if (cTarget[i] < mintarget) mintarget = cTarget[i];
            }
          for (size_t i = 0; i < h.size (); i++)
            {
              h[i] = h[i] / maxh*mintarget;
            }
          
          
          double minNormal = cNormalTemp[0];
          double minLow = cLowTemp[0];
          double minHigh = cHighTemp[0];
          for (size_t i = 1; i<cNormalTemp.size(); i++)
            {
              if (cNormalTemp[i] < minNormal) minNormal = cNormalTemp[i];
              if (cLowTemp[i] < minLow) minLow = cLowTemp[i];
              if (cHighTemp[i] < minHigh) minHigh = cHighTemp[i];
              //minNormal += cNormalTemp[i];
              //minLow += cLowTemp[i];
              //minHigh += cHighTemp[i];
            }
          for (size_t i = 0; i<cNormalTemp.size(); i++)
            {
              cNormalTemp[i] /= -minNormal;
              cLowTemp[i] /= -minLow;
              cHighTemp[i] /= -minHigh;
            }


          

          subplot (3, 1, 1);
          
          plot(_x,y); set(3);                            // Expected value in default color (blue)
	  plot(lx,ly);set("k");set("o");set(4);         // Data points as black star
	  plot(_x,su);set("g"); set(2);                  // Uncertainty as green lines
	  plot(_x,sl);set("g"); set(2);
	  plot(_x,_z);set("r"); set(3);                   // True function as red line
          
          subplot (3, 1, 2);
                    
	  plot(_x,c); set("g"); set(2);          // green: raw criteria values
          
          plot(_x,cTarget); set("m"); set(2);     // magenta: normalized criteria values
          plot(_x,cNormalTemp); set("y"); set(2); // yellow: normalized and clamped criteria values
          plot(_x,cLowTemp); set("b"); set(2);   // blue: low temp boltzmann criteria values
          plot(_x,cHighTemp); set("r"); set(2);  // red: high temp boltzmann criteria values
            
          //MCMC particles histogram
          subplot(3,1,3);
          plot(_bx,h); set("k"); set(3);           // black: histogram of particles
          plot(_x,cTarget); set("m"); set(2);       // magenta: target distribution 
	}
    };
    
  } //namespace utils

} //namespace bayesopt