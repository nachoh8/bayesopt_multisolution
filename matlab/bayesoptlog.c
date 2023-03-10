/*
-------------------------------------------------------------------------
   This file is part of BayesOpt, an efficient C++ library for 
   Bayesian optimization.

   Copyright (C) 2011-2014 Ruben Martinez-Cantin <rmcantin@unizar.es>
 
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

#include "bayesoptextras.h"

void mexFunction(int nlhs, mxArray *plhs[],
                 int nrhs, const mxArray *prhs[])
{
  double *xptr;
  char *filename_c;
  mxArray *xopt;
  const mxArray *func_name, *params, *filename;
  user_function_data udata;
  size_t nDim;
  unsigned int ii;
  bopt_params parameters;
  double *ub, *lb;    /* Upper and lower bound */
  double fmin;
  int error_code;
     
  /* Check correct number of parameters */
  CHECK0(nlhs != 4 || nrhs != 3 || nrhs != 6, 
	 "wrong number of arguments");
    
  /* TODO: Change This */
  udata.neval = 0;
  udata.verbose = 0;
   
  /* First term is the function handle or name */
  func_name = prhs[0];

  if (mxIsChar(func_name))
    {
      CHECK0(mxGetString(func_name, udata.f, FLEN) == 0,
	     "error reading function name string (too long?)");
      udata.nrhs = 1;
      udata.xrhs = 0;
    }
#ifndef HAVE_OCTAVE
  else if (mxIsFunctionHandle(func_name))
    {
      udata.prhs[0] = (mxArray *)func_name;
      strcpy(udata.f, "feval");
      udata.nrhs = 2;
      udata.xrhs = 1;
      }
#endif
  else
    {
      mexErrMsgTxt("First term should be a function name or function handle");
    }

  /* Second parameter. nDim */
  CHECK0(mxIsNumeric(prhs[1]) && !mxIsComplex(prhs[1]) 
	 && mxGetM(prhs[1]) * mxGetN(prhs[1]) == 1,
	 "nDim must be a scalar");
  nDim = (unsigned int) mxGetScalar(prhs[1]);

  udata.prhs[udata.xrhs] = mxCreateDoubleMatrix(1, nDim, mxREAL);

  xopt = mxCreateDoubleMatrix(1, nDim, mxREAL);
  xptr = mxGetPr(xopt);
  
  /* Third term. Logfile  */
  filename = prhs[2];
  CHECK0(mxIsChar(filename),"3rd parameter should be a string");

  filename_c = mxArrayToString(filename);

  /* Forth term. Parameters  */
  if (nrhs != 3)
    {
      CHECK0(mxIsStruct(prhs[3]), "4rd element must be a struct");
      params = prhs[3];
    }
  else
    {
      params = mxCreateStructMatrix(1,1,0,NULL);
    }
  parameters = load_parameters(params);

  if(nrhs == 6)
    {
      /* Load bounds */
      mexPrintf("Loading bounds...");
      CHECK0(mxIsDouble(prhs[4]) && !mxIsComplex(prhs[4])
	     && (mxGetM(prhs[4]) == 1    || mxGetN(prhs[4]) == 1)
	     && (mxGetM(prhs[4]) == nDim || mxGetN(prhs[4]) == nDim),
	     "lowerBound must be real row or column vector");

      lb = mxGetPr(prhs[4]);


      CHECK0(mxIsDouble(prhs[5]) && !mxIsComplex(prhs[5])
	     && (mxGetM(prhs[5]) == 1    || mxGetN(prhs[5]) == 1)
	     && (mxGetM(prhs[5]) == nDim || mxGetN(prhs[5]) == nDim),
	     "upperBound must be real row or column vector");

      ub = mxGetPr(prhs[5]);
      mexPrintf("done. \n");
    }
  else
    {
      lb = (double*)(mxCalloc(nDim,sizeof(double)));
      ub = (double*)(mxCalloc(nDim,sizeof(double)));
	 

      
      for (ii = 0; ii < nDim; ++ii) 
	{
	  lb[ii] = 0.;    
	  ub[ii] = 1.;
	}
    }

  error_code = bayes_optimization_log(nDim,user_function,&udata,lb,ub,xptr,
				      &fmin,filename_c,parameters);

  if(nrhs != 6)
    {
      mxFree(lb); 
      mxFree(ub);
    }

  mxDestroyArray(udata.prhs[udata.xrhs]);
  plhs[0] = xopt;
  if (nlhs > 1) 
    {
      plhs[1] = mxCreateDoubleMatrix(1, 1, mxREAL);
      *(mxGetPr(plhs[1])) = fmin;
    }
  if (nlhs > 2)
    {
      plhs[2] = mxCreateDoubleMatrix(1, 1, mxREAL);
      *(mxGetPr(plhs[2])) = (double)(error_code);
    }

}
