/*
-------------------------------------------------------------------------
   This file is part of BayesOpt, an efficient C++ library for 
   Bayesian optimization.

   Copyright (C) 2011-2013 Ruben Martinez-Cantin <rmcantin@unizar.es>
 
   BayesOpt is free software: you can redistribute it and/or modify it 
   under the terms of the GNU General Public License as published by
   the Free Software Foundation, either version 3 of the License, or
   (at your option) any later version.

   BayesOpt is distributed in the hope that it will be useful, but 
   WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU General Public License for more details.

   You should have received a copy of the GNU General Public License
   along with BayesOpt.  If not, see <http://www.gnu.org/licenses/>.
------------------------------------------------------------------------
*/

//#include "testfunctions.hpp"
#include "randgen.hpp"
#include "bayesopt/bayesopt.hpp"
#include "displaygp.hpp"

#include "specialtypes.hpp"

//////////////////////////////////////////////////////////////////////
class NonStationaryOneD: public bayesopt::ContinuousModel
{
public:
  NonStationaryOneD(bopt_params par):
    ContinuousModel(1,par) {};

  double evaluateSample(const vectord& xin)
  {
    normalDist normal(0,0.02);
    randNFloat sample(mEng, normal);
    if (xin.size() != 1)
      {
	std::cout << "WARNING: This only works for 1D inputs." << std::endl
		  << "WARNING: Using only first component." << std::endl;
      }
  
    double x = xin(0);
    double coef = (1-x)+0.2;
    return (x-0.2)*(x-0.2) + sin(10*x/coef)*0.2 + sample();
  };

  bool checkReachability(const vectord &query)  
  {return true;};

  void printOptimal()
  {
    std::cout << "Optimal:" << 0.760 << std::endl;
  }
private:
  randEngine mEng;

};
//////////////////////////////////////////////////////////////////////


// Unfortunately OpenGL functions require no parameters, so the object has to be global.
bayesopt::utils::DisplayProblem1D GLOBAL_MATPLOT;

void display( void ){ GLOBAL_MATPLOT.display(); }
void reshape( int w,int h ){ GLOBAL_MATPLOT.reshape(w,h); }
void idle( void ) { glutPostRedisplay(); } 

void mouse(int button, int state, int x, int y ){ GLOBAL_MATPLOT.mouse(button,state,x,y); }
void motion(int x, int y ){ GLOBAL_MATPLOT.motion(x,y); }
void passive(int x, int y ){ GLOBAL_MATPLOT.passivemotion(x,y); }

void keyboard(unsigned char key, int x, int y)
{
    GLOBAL_MATPLOT.keyboard(key, x, y); 
    if(key=='r')   //Toogle run/stop
      { 
	GLOBAL_MATPLOT.toogleRUN();
      }
    if(key=='s')   //Activate one step
      { 
	GLOBAL_MATPLOT.setSTEP();
      }
}

bool yesnoquestion(std::string question)
{
  unsigned char input;
  while (1)
    {
      std::cout << question << " [y/n]>";
      std::cin >> input;
      switch (input)
	{
	case 'y': return true;
	case 'n': return false;
	}
    }
}

int main(int nargs, char *args[])
{
  bopt_params parameters = initialize_parameters_to_default();
  parameters.n_init_samples = 7;
  parameters.n_iterations = 300;
  parameters.n_iter_relearn = 1;
  parameters.l_type = L_MCMC;
  parameters.verbose_level = 2;
  //parameters.noise = 0.001;

  if (!yesnoquestion("Do you want to optimize the function?"))
    {
      std::cout << "Setting for pure exploration" << std::endl;
      set_criteria(&parameters,"cAopt");
      parameters.n_crit_params = 0;
    }

  if (yesnoquestion("Do you want to actively learn the hyperparameters?"))
    {
      parameters.active_hyperparams_coef = 1;
      std::cout << "Setting active coef to "<< parameters.active_hyperparams_coef << std::endl;
    }

  if (yesnoquestion("Do you want to use nonstationary kernels?"))
    {
      std::cout << "Using 4 Matern kernels " << std::endl;
      // set_kernel(&parameters,"kNonSt(kSum(kMaternISO5,kNoise),kSum(kMaternISO5,kNoise),kSum(kMaternISO5,kNoise))");
      
      // double mean[128] = {1, 0.01, 1, 0.01, 1, 0.01};
      // double std[128] = {5, 0.1, 5, 0.1, 5, 0.1};
      // size_t nhp = 6;

      //      set_kernel(&parameters,"kSum(kNonSt(kMaternISO5,kMaternISO5,kMaternISO5),kNoise)");
      set_kernel(&parameters,
		 "kSum(kNoise,kNonSt(kMaternISO5,kMaternISO5,kMaternISO5,kMaternISO5))");
      
      double mean[128] = {1, 1, 1, 1, 1};
      double std[128] = {5, 5, 5, 5, 5};
      size_t nhp = 5;


      memcpy(parameters.kernel.hp_mean, mean, nhp * sizeof(double));
      memcpy(parameters.kernel.hp_std,std, nhp * sizeof(double));
      parameters.kernel.n_hp = nhp;

      double domain_std;

      if (yesnoquestion("Do you want the kernels to have overlap?"))
	{
	  domain_std = 0.5;
	}
      else
	{
	  domain_std = 0.12;
	}

      parameters.kernel.domain_center[0][0] = 0.1;
      parameters.kernel.domain_std[0][0]    = domain_std;
      parameters.kernel.domain_center[1][0] = 0.35;
      parameters.kernel.domain_std[1][0]    = domain_std;
      parameters.kernel.domain_center[2][0] = 0.65;
      parameters.kernel.domain_std[2][0]    = domain_std;
      parameters.kernel.domain_center[3][0] = 0.9;
      parameters.kernel.domain_std[3][0]    = domain_std;
      parameters.kernel.n_basis_kernels     = 4;
      parameters.kernel.n_dim_domain        = 1;
    }


  boost::scoped_ptr<NonStationaryOneD> opt(new NonStationaryOneD(parameters));
  GLOBAL_MATPLOT.init(opt.get(),1);

  glutInit(&nargs, args);
  glutCreateWindow(50,50,800,650);
  glutDisplayFunc( display );
  glutReshapeFunc( reshape );
  glutIdleFunc( idle );
  glutMotionFunc( motion );
  glutMouseFunc( mouse );
  glutPassiveMotionFunc(passive);    
  glutKeyboardFunc( keyboard );        
  glutMainLoop();    

  return 0;
}
