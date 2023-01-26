#include <ctime>
#include <fstream>
#include <sstream>
#include "testfunctions.hpp"
#include "bopt_state.hpp"

int main(int nargs, char *args[])
{
  // BayesOpt Only Samples: 20 init + 80 bopt            = 100 samples
  // BayesOpt+BOBYQA:       20 init + 60 bopt + 20 final = 100 samples
  size_t rep = 5;
  size_t num_init_samples = 20;
  size_t num_iters = 80;
  size_t num_polish_iters = 20;
  int verbose = 1;

  learning_type l_type = L_EMPIRICAL;

  long dims = 5;

  std::string filename = "test_polish_opt.dat";

  std::vector<double> completeBO, branchPoint, withBOBYQA;

  for (size_t i=0; i<rep; i++)
    {
      std::cout << "Repetition: " << i+1 <<"/"<< rep << std::endl;

      int randomSeed = std::time(0);

      // Optimization with BayesOpt only
      {
        bayesopt::Parameters par;
        par.n_iterations = num_iters;
        par.n_init_samples = num_init_samples;
        par.n_final_samples = 0;

        par.noise = 1e-7;
        par.random_seed = randomSeed;
        par.verbose_level = verbose;
	par.surr_name = "sStudentTProcessJef";
        par.l_type = l_type;
        par.n_iter_relearn = 1;

        Rosenbrock continuous(dims,par);

        continuous.initializeOptimization();

        for (size_t ii = 0; ii < par.n_iterations; ++ii)
          {
            continuous.stepOptimization();

            if( ii == num_iters - num_polish_iters - 1)
              {
                // Store the optimization at the branch point between BOptOnly and BOBYQA
                bayesopt::BOptState state;
                continuous.saveOptimization(state);
                // hack the state to appear as completed at branch point and with n_final_samples
                state.mParameters.n_iterations = num_iters - num_polish_iters;
                state.mParameters.n_final_samples = num_polish_iters;
                state.saveToFile(filename);

                // Store the value at branch point
                branchPoint.push_back(continuous.getValueAtMinimum());
              }
          }

        // Store values at end of BOpt Only
        completeBO.push_back(continuous.getValueAtMinimum());

        // For dat plotting comparison
        bayesopt::BOptState state;
        continuous.saveOptimization(state);
        std::stringstream sstm;
        sstm << "test_polish_BOptOnly_" << i << ".dat";
        state.saveToFile(sstm.str());
      }

      // Load optimiztoin from branch point and perform final refinement by BOBYQA
      {
        bayesopt::Parameters par;
        par.n_iterations = num_iters-num_polish_iters;
        par.n_final_samples = num_polish_iters;

        par.load_filename = filename;
        par.load_save_flag = 1;

        Rosenbrock continuous(dims,par);

        vectord bestPoint;
        continuous.optimize(bestPoint);

        // Store values at end of BOBYQA
        withBOBYQA.push_back(continuous.getValueAtMinimum());

        // For dat plotting comparison
        bayesopt::BOptState state;
        continuous.saveOptimization(state);
        std::stringstream sstm;
        sstm << "test_polish_BOBYQAPolish_" << i << ".dat";
        state.saveToFile(sstm.str());
      }
    }


  // Results
  std::cout.precision(17);

  int countSame = 0;
  int countBO = 0;
  int countPolish = 0;
  int countPolishDoesNothing = 0;

  double meanBO = 0;
  double meanPolish = 0;

  std::cout << "Comparison of each repetition:" << std::endl;

  for(size_t i=0; i<rep; i++)
    {
      if(completeBO[i] == withBOBYQA[i])
        {
          std::cout << "("<<i<<")" << completeBO[i] << "==" << withBOBYQA[i] << std::endl;
          countSame++;
        }
      else if(completeBO[i] < withBOBYQA[i])
        {
          std::cout << "("<<i<<")" << completeBO[i] << "<" << withBOBYQA[i] << std::endl;
          countBO++;
          meanBO += withBOBYQA[i]-completeBO[i];
        }
      else
        {
          std::cout << "("<<i<<")" << completeBO[i] << ">" << withBOBYQA[i] << std::endl;
          countPolish++;
          meanPolish += completeBO[i]-withBOBYQA[i];
        }

      if(branchPoint[i] == withBOBYQA[i])
        {
          countPolishDoesNothing++;
        }
    }

  if(countBO > 0)
    {
      meanBO /= countBO;
    }
  if(countPolish > 0)
    {
      meanPolish /= countPolish;
    }

  std::cout << "Results:" << std::endl;

  std::cout << countSame << "/" << rep << " Both are equal" << std::endl;

  std::cout << countBO << "/" << rep << " BayesOpt only is better (" << meanBO <<")" <<std::endl;

  std::cout << countPolish << "/" << rep << " with BOBYQA is better (" << meanPolish <<")" <<std::endl;

  std::cout << countPolishDoesNothing << "/" << rep << " BOBYQA not improving" << std::endl;

  // Try to remove used .dat file
  if( remove( filename.c_str() ) == 0 )
    {
      std::cout << "File \"" << filename << "\" successfully removed" << std::endl;
    }
  else
    {
      std::cout << "Error: cannot remove \"" << filename << "\" file" << std::endl;
    }

  return 0;
}
