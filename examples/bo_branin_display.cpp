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
#include "displaygp.hpp"
#include "param_loader.hpp"
#include "ublas_cholesky.hpp"
#include "ublas_trace.hpp"
#include "bayesopt/bayesopt.hpp"
#include <boost/numeric/ublas/assignment.hpp>

namespace ublas = boost::numeric::ublas;

class MixtureGaussian: public bayesopt::ContinuousModel
{
public:
  MixtureGaussian(bayesopt::Parameters par):
    ContinuousModel(2,par){};

  double mgpdf(const vectord& x, const vectord& mu, const matrixd& S, double p)
  {
    size_t n = mu.size();
    matrixd L(n,n);
    bayesopt::utils::cholesky_decompose(S,L);
    vectord u = mu-x;
    inplace_solve(L,u,ublas::lower_tag());
    double quad = ublas::inner_prod(u,u)/2.0;
    double det = bayesopt::utils::log_trace(L);
    double cst = n*std::log(M_PI)/2.0;

    return p*std::exp(-quad-det-cst);
  }

  double evaluateSample(const vectord& xin)
  {
    if (xin.size() != 2)
      {
	std::cout << "WARNING: This only works for 2D inputs." << std::endl
		  << "WARNING: Using only first component." << std::endl;
      }
  
    vectord x(xin);
    x(0) = x(0) * 7 - 1;
    x(1) = x(1) * 8 - 4;

    double solution = 0.0;

    vectord mu(2); mu <<= 1.5, 0.0;
    matrixd sigma(2,2); sigma <<= 1.0, 0.0, 0.0, 1.0;
    double p = 0.2;

    solution += mgpdf(x,mu,sigma,p);

    mu <<= 3.0, 3.0;
    sigma <<= 5.0, 0.0, 0.0, 1.0;
    p = 0.5;

    solution += mgpdf(x,mu,sigma,p);

    mu <<= 4.0, -1.0;
    sigma <<= 2.0, -0.5, 0.5, 4.0;
    p = 0.5;

    solution += mgpdf(x,mu,sigma,p);

    mu <<= 0.0, -2.0;
    sigma <<= 0.3, 0, 0, 0.3;
    p = 0.1;

    solution += mgpdf(x,mu,sigma,p);

    mu <<= 2.0, 2.0;
    sigma <<= 10.0, 0, 0, 10.0;
    p = 0.5;

    solution += mgpdf(x,mu,sigma,p);

    mu <<= 2.0, -1.0;
    sigma <<= 5.0, 0, 0, 5.0;
    p = 0.5;

    solution += mgpdf(x,mu,sigma,p);

    return -solution;
  };

bool checkReachability(const vectord &query)  {return true;};

void printOptimal() {};
};

// Unfortunately OpenGL functions require no parameters, so the object
// has to be global.
bayesopt::utils::DisplayProblem2D GLOBAL_MATPLOT;

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
    //    par.init_method = 2;
    par.n_iterations = 100;
  par.n_init_samples = 50;
    par.n_iter_relearn = 1;
  //par.random_seed = 10;
  //set_criteria(&par,"cLCB");
    //set_surrogate(&par,"sStudentTProcessNIG");

    par.l_type = L_MCMC;
    par.sc_type = SC_MAP;
    par.verbose_level = 1;
    //bayesopt::utils::ParamLoader::save("bo_branin_display.txt", par);
  }
  
  //  boost::scoped_ptr<BraninPlusEasomNormalized> branin(new BraninPlusEasomNormalized(par));
  boost::scoped_ptr<MixtureGaussian> branin(new MixtureGaussian(par));
  GLOBAL_MATPLOT.init(branin.get(),2);

  vectord sv(2);  
  sv(0) = 0.1239; sv(1) = 0.8183;
  GLOBAL_MATPLOT.setSolution(sv);
  
  sv(0) = 0.5428; sv(1) = 0.1517;
  GLOBAL_MATPLOT.setSolution(sv);

  sv(0) = 0.9617; sv(1) = 0.1650;
  GLOBAL_MATPLOT.setSolution(sv);

  sv(0) = 0.7522; sv(1) = 0.4188;
  GLOBAL_MATPLOT.setSolution(sv);

  branin->printOptimal();

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
