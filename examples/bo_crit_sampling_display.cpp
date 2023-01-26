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

#include "testfunctions.hpp"
#include "displaycrit.hpp"
#include "param_loader.hpp"
#include "ublas_cholesky.hpp"
#include "ublas_trace.hpp"
#include "bayesopt/bayesopt.hpp"
#include <boost/numeric/ublas/assignment.hpp>

namespace ublas = boost::numeric::ublas;

// Unfortunately OpenGL functions require no parameters, so the object
// has to be global.
bayesopt::utils::DisplayCrit1D GLOBAL_MATPLOT;

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
    //par.init_method = 2;
    par.n_iterations = 50;
    par.n_init_samples = 5;
    par.n_iter_relearn = 1;
    //par.random_seed = 10;
    
    // par.crit_name = "cLCBexp";
    // par.crit_params.resize(2, false);
    // par.crit_params[0] = 1;
    // par.crit_params[1] = 1;

    par.crit_name = "cEI";
    
//    par.crit_name = "cEIexp";
//    par.crit_params.resize(2, false);
//    par.crit_params[0] = 1.0;
//    par.crit_params[1] = 1.0;
    
    
  //  par.crit_name = "cEntropySearch";
  //  par.crit_params.resize(4, false);
  //  par.crit_params[0] = 50; // ES mNumCandidates
  //  par.crit_params[1] = 1000; // ES mNumGPSamples
  //  par.crit_params[2] = 100; // ES mNumSamplesY
  //  par.crit_params[3] = 500; // ES mNumTrialPoints
    
//    par.crit_name = "cProd(cEIexp, cEntropySearch)";
//    int i = 2;
    
  //  par.crit_params.resize(i+4, false);
  //  par.crit_params[0] = 1.0; // (ignore, shared parameter with any EI)
  //  par.crit_params[1] = 0.1; // EI temperature
  //  par.crit_params[i+0] = 20; // ES mNumCandidates
  //  par.crit_params[i+1] = 500; // ES mNumGPSamples
  //  par.crit_params[i+2] = 100; // ES mNumSamplesY
  //  par.crit_params[i+3] = 500; // ES mNumTrialPoints

            
    par.force_jump = 0;

    par.l_type = L_EMPIRICAL;
    par.sc_type = SC_MAP;
    par.verbose_level = 1;
    par.par_type = PAR_MCMC;
    //bayesopt::utils::ParamLoader::save("bo_branin_display.txt", par);
  }

  boost::scoped_ptr<HardOneD> bopt(new HardOneD(par));
  bopt->setNormalizeCriteria(false);
  bopt->setBurnOutSamples(1000); // low amount to speed up pre-display optimization steps
  
  //bopt->setBoltzmannPolicy(true, 1.0, 1.0, 1.0);
  
  
  bopt->initializeOptimization ();
  bopt->stepOptimization ();
  GLOBAL_MATPLOT.init(bopt.get(),1);

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
