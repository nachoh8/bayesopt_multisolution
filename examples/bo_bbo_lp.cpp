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

#include <ctime>
#include <string>
#include <cmath>
#include <iostream>
#include <boost/numeric/ublas/assignment.hpp>
#include <boost/numeric/ublas/vector_proxy.hpp>
#include <boost/math/distributions/normal.hpp>

#include "bayesopt/bayesopt.hpp"               // For the C++ API
#include "specialtypes.hpp"

namespace ublas = boost::numeric::ublas;

/*double test_function(const vectord& query) {
  return 1.0f + ublas::sum(query);
}

double forrester(const vectord& query) {
  assert(query.size() == 1);
  double x = query[0];

  double fval = 6 * x - 2;
  fval = fval * fval;

  fval = fval * std::sin(12 * x - 4);

  return fval;
}

class TestBBOLP : public bayesopt::ContinuousModel {

public:
    TestBBOLP(size_t dim, const bayesopt::Parameters& bo_params)
    : bayesopt::ContinuousModel(dim, bo_params)
    {
    }

    /**
     * @brief Defines the actual function to be optimized
     * 
     * @param query point to be evaluated
     * @return value of the function at the point evaluated
     *
    double evaluateSample(const vectord& query) {
        double res = forrester(query);
        // std::cout << "query " << ": " << query << " -> " << res << std::endl;
        return res;
    }

    vectord evaluateSamples( const vecOfvec &queries ) {
      vectord res(queries.size());

      for (size_t i = 0; i < queries.size(); i++) {
        res[i] = forrester(queries[i]);
        // std::cout << "query " << i << ": " << queries[i] << " -> " << res[i] << std::endl;
      }

      return res;
    }
};

int main(int argc, char *argv[]) {
    const size_t ndim = 1;

    bopt_params params = initialize_parameters_to_default();
    params.n_init_samples = 10;
    params.n_iterations = 50;
    params.verbose_level = 1;
    params.n_parallel_samples = 3;
    params.par_type = PAR_BBO;
    std::string criteria_name = "cLP(cLCB)";
    set_criteria(&params, criteria_name.c_str());
    params.n_crit_params = 4;
    params.crit_params[0] = params.n_parallel_samples;
    params.crit_params[1] = 1;
    params.crit_params[2] = ndim;
    params.crit_params[3] = 1.0;

    TestBBOLP opt(ndim, params);
    
    vectord result(ndim);
    opt.optimize(result);
    const double value = opt.getFinalValue();

    std::cout << "Final result: " << result << std::endl;
    std::cout << "Final value: " << value << std::endl;

    return 0;
}*/

const int BATCH_SIZE = 3;
const int N = 10;

inline double evaluate(const vectord& x) {
  return ublas::sum(x);
}

void prediction(const vectord& x, double& mean, double& std) {
  mean = evaluate(x);
  std = sqrt(abs(mean));
}

class TestLP
  {
  public:
    TestLP() {
      X_batch = vecOfvec(BATCH_SIZE - 1);
        
      r = vectord(X_batch.size());
      s = vectord(X_batch.size());

      transformation = 1;

      min = -1.0;

      L = 10.0;
    }

    double compute (const vectord &x) 
    { 
      return penalized_acquisition(x);
    }

    void reset_batch() {
      current_point = 0;
      min = 0.0;
      L = 0.0;
    };

    void update_batch(const vectord &x) {
      if (current_point == 0) {
        // do nothing
      }

      X_batch[current_point++] = x;

      for (int i = 0; i < current_point; i++) {
        double mean, std;
        prediction(X_batch[i], mean, std);

        if (std < 1e-16) {
            std = 1e-16;
        }
        s[i] = sqrt(std) / L;
        r[i] = (mean - min) / L;
      }
    };

    vectord r, s;
    vecOfvec X_batch;
    int current_point = 0;

  private:
    vectord hammer(const vectord& x) {
      vectord res(current_point);

      for (int i = 0; i < current_point; i++) {
          vectord aux = x - X_batch[i];
          aux = ublas::element_prod(aux, aux);
          
          double v = ublas::sum(aux);
          v = (sqrt(v) - r[i]) / s[i];
          
          res[i] = std::log(boost::math::cdf(boost::math::normal(), v));          
      }

      return res;
    }

    double penalized_acquisition(const vectord &x) {
      double fval = -evaluate(x);

      switch (transformation) // map to log natural space
      {
      case 1: // Softplus
          if (fval >= 40.0) {
            fval = std::log(fval); // g(z) = ln(z)
          } else {
            fval = std::log(std::log1p(std::exp(fval))); // g(z) = ln(ln(1+e^z))
          }
          break;
      
      default: // None
          fval = std::log(fval + 1e-50); // g(z) = ln(z)
          break;
      }

      fval = -fval;
      
      if (current_point > 0) {
        vectord exlcusion = hammer(x);
        fval += ublas::sum(exlcusion) * -1.0;
      }

      return fval;
    }

    double min = 0.0, L = 0.0;

    int transformation;
  };


int main(int argc, char *argv[]) {
  std::cout << "INIT\n";

  TestLP lp = TestLP();
  vecOfvec X;
  double min_v = 100000;
  int min_idx = -1;
  for (int i = 0; i < N; i++) {
    vectord x(1);
    x[0] = double(i)/double(N);
    double v = lp.compute(x);

    std::cout << i << " -> " << x << " -> " << v << std::endl;
    if (v < min_v) {
      min_idx = i;
      min_v = v;
    }
    X.push_back(x);
  }

  std::cout << "Min " << X[min_idx] << " -> " << min_v << std::endl;
  for (int k = 1; k < BATCH_SIZE; k++) {
    lp.update_batch(X[min_idx]);
    std::cout << "B" << k << " -> " << lp.r << " , " << lp.s << std::endl;
    min_v = 100000;
    for (int i = 0; i < N; i++) {
      vectord x = X[i];
      bool found = false;
      for (int j = 0; j < lp.current_point; j++) {
        if (x[0] == lp.X_batch[j][0]) {
          found = true;
          break;
        }
      }
      if (found) {
        continue;
      }

      double v = lp.compute(x);
      std::cout << i << " -> " << x << " -> " << v << std::endl;
      if (v < min_v) {
        min_idx = i;
        min_v = v;
      }
    }

    std::cout << "Min " << X[min_idx] << " -> " << min_v << std::endl;
  }

}



