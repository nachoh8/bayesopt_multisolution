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

#include <limits>
#include <sstream>
#include "testfunctions.hpp"
#include "param_loader.hpp"
#include "bopt_state.hpp"
#include "fileparser.hpp"
#include "dataset.hpp"

int main(int nargs, char *args[])
{
  bayesopt::Parameters par;
  if(nargs > 1){
    if(!bayesopt::utils::ParamLoader::load(args[1], par)){
        std::cout << "ERROR: provided file \"" << args[1] << "\" does not exist" << std::endl;
        return -1;
    }
  }
  else{
    par = initialize_parameters_to_default();
    par.n_iterations = 190;
    par.random_seed = -1;
    par.verbose_level = 1;
    par.noise = 1e-3;
    par.load_save_flag = 2;
    //bayesopt::utils::ParamLoader::save("bo_branin.txt", par);
  }

  srand (time(NULL));

  // Test Gaussian noise
  BraninNormalizedNoise bnn(par);
  bnn.testGaussian(2000, 0.2, 20);
  bnn.printOptimal();

  int num_executions = 25;
  int init_value = 100;
  for(int i=init_value; i<num_executions+init_value; i++)
    {
      std::ostringstream valss;
      valss << "Dist_Value_" << i << ".dat";
      std::ostringstream meanss;
      meanss << "Dist_Mean_" << i << ".dat";

      par.random_seed = i;
      par.save_filename = valss.str();

      vecOfvec meanPoints;

      BraninNormalizedNoise braninNoise(par, par.noise);
      vectord result(2);


      braninNoise.initializeOptimization();

      // Fill first mean values with the initial values
      for(size_t i=0; i<par.n_init_samples; i++)
        {
          meanPoints.push_back(braninNoise.getData()->getSampleX(i));
        }

      for(size_t i=0; i<par.n_iterations; i++)
        {
          braninNoise.stepOptimization();

          // Calculate and store mean point
          vectord point = braninNoise.getFinalResult(true);
          meanPoints.push_back(point);

          bayesopt::utils::FileParser fp(meanss.str());
          fp.openOutput();
          fp.readOrWrite("mX", meanPoints);
          fp.close();
        }
      braninNoise.finalizeOptimization();

      vectord meanPoint = braninNoise.getFinalResult(true);
      std::cout << "Mean: " << meanPoint << std::endl;
      braninNoise.printOptimal();
    }

  return 0;
}
