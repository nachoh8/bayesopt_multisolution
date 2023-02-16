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
#include <math.h>  // exp
#include <utility>
#include <functional>
#include <bits/stdc++.h>
#include <boost/numeric/ublas/vector_expression.hpp>
#include <boost/sort/sort.hpp>
#include <boost/numeric/ublas/vector_proxy.hpp>

#include <pyclustering/cluster/xmeans.hpp>
#include <pyclustering/cluster/kmeans_plus_plus.hpp>
#include <pyclustering/definitions.hpp>

#include "bayesopt/bayesoptbase.hpp"

#include "log.hpp"
#include "posteriors/posterior_model.hpp"
#include "specialtypes.hpp"
#include "spsa.hpp"
#include "bopt_state.hpp"
#include "criteria/criteria_functors.hpp"
#include "unscented_transform.hpp"
#include "fixed_dataset.hpp"

namespace pycc = pyclustering;
namespace pyclst = pyclustering::clst;

namespace bayesopt
{
  BayesOptBase::BayesOptBase(size_t dim, Parameters parameters):
    mParameters(parameters), mDims(dim)
  {
    // Setting verbose stuff (files, levels, etc.)
    int verbose = mParameters.verbose_level;
    if (verbose>=3)
      {
        FILE* log_fd = fopen( mParameters.log_filename.c_str() , "w" );
        Output2FILE::Stream() = log_fd;
        verbose -= 3;
      }

    switch(verbose)
      {
        case 0: FILELog::ReportingLevel() = logWARNING; break;
        case 1: FILELog::ReportingLevel() = logINFO; break;
        case 2: FILELog::ReportingLevel() = logDEBUG4; break;
        default:
        FILELog::ReportingLevel() = logERROR; break;
      }

    // Show bayesopt version
    FILE_LOG(logINFO) << "Current Version: BayesOptPro with Nonstationarity";
    FILE_LOG(logINFO) << "Num Solutions to search: " << mParameters.num_solutions;
    if (mParameters.num_solutions > 1) {
      FILE_LOG(logINFO) << "Min diversity between solutions: " << mParameters.diversity_level;
    }

    // Random seed
    if (mParameters.random_seed < 0) mParameters.random_seed = std::time(0);
    mEngine.seed(mParameters.random_seed);

    // Configure iteration parameters
    if (mParameters.n_init_samples <= 0)
      {
	mParameters.n_init_samples =
	  static_cast<size_t>(ceil(0.1*mParameters.n_iterations));
      }


    // Posterior surrogate model
    mModel.reset(PosteriorModel::create(dim,mParameters,mEngine));

    mSPSA_sampler.reset(new SPSA());
    mSPSA_sampler->setRandomEngine(mEngine);
    mSPSA_sampler->setNoiseLevel(mParameters.noise);

    // TODO(Javier): mcmc criteria sampling
    mNormalizeCriteria = false;
    mUseBoltzmannPolicy = false;
    mBurnOutSamples = mParameters.burnout_mcmc_parallel * dim;

    mUT.reset(new IsotropicUnscentedTransform(dim, mParameters));
    mUT->setPosteriorModel(mModel.get());
  }

  BayesOptBase::~BayesOptBase() {} // Default destructor

  // OPTIMIZATION INTERFACE
  void BayesOptBase::optimize(vectord &bestPoint)
  {
    assert(mDims == bestPoint.size());

    // Restore state from file
    if(mParameters.load_save_flag == 1 || mParameters.load_save_flag == 3)
      {
        BOptState state;
        bool load_succeed = state.loadFromFile(mParameters.load_filename,
					       mParameters);
        if(load_succeed)
          {
            restoreOptimization(state);
            FILE_LOG(logINFO) << "State succesfully restored from file \""
			      << mParameters.load_filename << "\"";
          }
        else
          {
	    // If load not succeed, print info message and then
	    // initialize a new optimization
            FILE_LOG(logINFO) << "File \"" << mParameters.load_filename
			      << "\" does not exist,"
			      << " starting a new optimization";
            initializeOptimization();
          }
      }
    else
      {
        // Initialize a new state
        initializeOptimization();
      }

    for (size_t ii = mCurrentIter; ii < mParameters.n_iterations; ++ii)
      {
        stepOptimization();
      }

    finalizeOptimization();

    bestPoint = getFinalResult();
  } // optimize

  // MULTISOLUTION FUNCTIONS

  void batch_combinations(const int offset, const int k, const vecOfvec& elements, std::vector<size_t>& current, std::vector< std::pair<std::vector<size_t>, double> >& results) {
    // Si hemos recorrido toda la lista de elementos, añadimos la combinación actual a los resultados
    if (offset == elements.size()) {
      if (current.size() == k) {
        std::pair<std::vector<size_t>, double> res;
        res.first = current;

        double dist = 0.0;
        double n = 0.0;
        for (int i = 0; i < current.size(); i++) {
          size_t idx1 = current[i];
          for (int j = i + 1; j < current.size(); j++) {
            size_t idx2 = current[j];
            dist += norm_2(elements[idx1] - elements[idx2]);
            n += 1.0;
          }
        }
        res.second = dist / n;
        
        results.push_back(res);
      }
      return;
    }

    // No incluimos el elemento en la combinación actual
    batch_combinations(offset + 1, k, elements, current, results);

    // Incluimos el elemento en la combinación actual
    current.push_back(offset);
    batch_combinations(offset + 1, k, elements, current, results);

    // Eliminamos el elemento de la combinación actual para no afectar a las siguientes iteraciones
    current.pop_back();
  }
  
  void select_batch_candidates(const vecOfvec& candidates, const size_t batch_size, std::vector<size_t>& selection) {
    std::vector<size_t> current;
    std::vector< std::pair<std::vector<size_t>, double> > results;
    batch_combinations(0, batch_size, candidates, current, results);

    double best = 0.0;
    for (auto& res : results) {
      double dist = res.second;
      // std::cout << "Combination: " << res.first[0] << ", " << res.first[1] << ", " << res.first[2] << " | " << dist << std::endl;
      if (dist > best) {
        selection = res.first;
        best = dist;
      }
    }
    // std::cout << "Len Candidates: " << results.size() << std::endl;
  }

  void cluster_correction(const pycc::dataset& py_centroids, const double divf, std::function<vectord(const vectord&)> remap_pt_func,
                          vecOfvec& centroids, std::set< size_t >& valid_cls, std::set< std::set<size_t> >& to_join_cls
                        ) {
    size_t num_clusters = py_centroids.size();
    size_t ndim = py_centroids[0].size();

    // cast to vectord
    for (size_t cl = 0; cl < num_clusters; cl++) {
      vectord cx(ndim);
      std::copy(py_centroids[cl].begin(), py_centroids[cl].end(), cx.begin());

      centroids.push_back(cx);
    }

    // get valid and no valid sets of clusters
    std::set< std::set<size_t> > novalid_cls;

    for (size_t cl = 0; cl < num_clusters; cl++) {
      vectord cx = remap_pt_func(centroids[cl]);

      bool added = false;
      std::set<size_t> neighbors;
      for (size_t cl2 = 0; cl2 < num_clusters; cl2++) {
            if (cl == cl2) continue;

            if (norm_2(cx - remap_pt_func(centroids[cl2])) < divf) { // both cluster are considered the same
              neighbors.insert(cl2);
            }
      }
      
      if (neighbors.size() > 0) {
        neighbors.insert(cl);
        novalid_cls.insert(neighbors);
      } else {
        valid_cls.insert(cl);
      }
    }

    // Get max differents no valid sets
    for (size_t cl = 0; cl < num_clusters; cl++) {
      if (valid_cls.count(cl) == 1) continue;

      std::set<size_t> nv_max;
      for (auto& nv : novalid_cls) {
        if (nv.count(cl) == 0) continue;
        
        if (nv.size() > nv_max.size()) {
          nv_max = nv;
        }
      }

      to_join_cls.insert(nv_max);
    }

  }

  inline float quantile(const vectord& sorted_values, const float q) {
    double index = q * (sorted_values.size()-1);
    int lowerIndex = floor(index);
    int upperIndex = ceil(index);
      
    if (index > lowerIndex && index < upperIndex) {
        return (sorted_values[lowerIndex] + sorted_values[upperIndex]) / 2;
      
    }
    else {
      return sorted_values[int(index)];
    }
  }
  
  void compute_clusters(const Dataset* data, const size_t num_solutions, const double divf, std::function<vectord(const vectord&)> remap_pt_func,
                        vecOfvec& res_Qs, std::vector<double>& res_Yv,
                        std::vector< std::vector<size_t> >& res_clusters, vecOfvec& res_centroids
                      ) {
    FILE_LOG(logINFO) << "------Computing clusters------";

    const size_t ndata = data->mY.size();

    vectord vsorted(ndata);
    std::copy(data->mY.begin(), data->mY.end(), vsorted.begin());

    boost::sort::spinsort(vsorted.begin(), vsorted.end());

    size_t max_sols = 0;
    size_t q_final = 0;
    std::vector< std::vector<size_t> > clusters;
    vecOfvec centroids;
    std::set< size_t > valid_cls;
    std::set< std::set<size_t> > to_join_cls;

    size_t prev_n_pts = 0;
    for (size_t q = 1; q < 10; q++) { // decils -> 0.1*q, q in [1,9]
      /// 1. Get min value from q-th quantile
      double qd = 0.1 * q;
      double lmin = quantile(vsorted, qd);

      FILE_LOG(logINFO) << "\n===========\nQuantil: " << qd << " -> " << lmin;

      /// 2. Filter data
      pycc::dataset _Qs;
      std::vector<double> _Yv;
      for (size_t i = 0; i < ndata; i++) {
        double v = data->mY[i];
        if (v < lmin) {
          pycc::point pt(data->mX[i].begin(), data->mX[i].end());
          _Qs.push_back(pt);
          _Yv.push_back(v);
        }
      }

      size_t n_pts = _Yv.size();
      FILE_LOG(logINFO) << "Number of points to cluster: " << n_pts;

      if (n_pts < num_solutions || n_pts == prev_n_pts) {
        continue;
      }
      prev_n_pts = n_pts;

      /// 3. Clustering
      pycc::dataset init_centers;
      pyclst::kmeans_plus_plus kmeans_init = pyclst::kmeans_plus_plus(num_solutions);
      kmeans_init.initialize(_Qs, init_centers);

      pyclst::xmeans xmeans_cl = pyclst::xmeans(
        init_centers, num_solutions,
        pyclst::xmeans::DEFAULT_TOLERANCE, pyclst::xmeans::DEFAULT_SPLITTING_TYPE,
        20
        );
      
      pyclst::xmeans_data cl_result;
      xmeans_cl.process(_Qs, cl_result);

      const pyclst::cluster_sequence py_clusters = cl_result.clusters();
      const pycc::dataset py_centroids = cl_result.centers();

      /// 4. Check clusters
      vecOfvec _centroids;
      std::set< size_t > _valid_cls;
      std::set< std::set<size_t> > _to_join_cls;
      cluster_correction(py_centroids, divf, remap_pt_func, _centroids, _valid_cls, _to_join_cls);

      size_t l = _valid_cls.size() + _to_join_cls.size();

      FILE_LOG(logINFO) << "Num. Valid: " << _valid_cls.size() << " | Num. No Valid: " << _to_join_cls.size();

      if (l > max_sols) {
        max_sols = l;
        q_final = q;
        
        // res_Qs = vecOfvec(_Qs.begin(), _Qs.end());
        res_Qs.clear();
        for (auto q : _Qs) {
          vectord qd(q.size());
          std::copy(q.begin(), q.end(), qd.begin());
          res_Qs.push_back(qd);
        }
        res_Yv = std::vector<double>(_Yv.begin(), _Yv.end());
        clusters = std::vector< std::vector<size_t> >(py_clusters.begin(), py_clusters.end());
        centroids = vecOfvec(_centroids.begin(), _centroids.end());
        valid_cls = std::set< size_t >(_valid_cls.begin(), _valid_cls.end());
        to_join_cls = std::set< std::set<size_t> >(_to_join_cls.begin(), _to_join_cls.end());

        if (max_sols == num_solutions) {
          break;
        }
      }

    }

    /// 5. Fuse bad clusters
    FILE_LOG(logINFO) << "---->Final selected quantil: " << q_final << " with " << max_sols << " diverse solutions";

    for (auto& cl : valid_cls) {
      res_clusters.push_back(clusters[cl]);
      res_centroids.push_back(centroids[cl]);
    }

    for (auto& jcls : to_join_cls) {
      std::string s = "Joined clusters:";
      
      std::vector<size_t> final_cluster;
      vectord final_centroid = zvectord(centroids[0].size());
      for (auto& cl : jcls) {
        s += " " + std::to_string(cl);
        
        final_cluster.insert(final_cluster.end(), clusters[cl].begin(), clusters[cl].end());
        final_centroid += centroids[cl];
      }

      final_centroid /= double(jcls.size());

      FILE_LOG(logDEBUG) << s << " to centroid: " << final_centroid;

      res_clusters.push_back(final_cluster);
      res_centroids.push_back(final_centroid);
    }

  }

  void cluster_score(const vectord& cluster_size, const vectord& cluster_min, const double global_max,
                    vectord& cluster_score
                    ) {
    size_t n_cls = cluster_min.size();
    
    /// 1. Transform to maximization problem
    vectord Yv = cluster_min * -1;
    double gmin = -global_max;
    if (gmin < 0.0) {
      Yv = Yv + svectord(n_cls, abs(gmin));
    }

    /// 2. Get constants
    double Ymean = sum(Yv) / double(n_cls);
    double Ymax = Yv[0], Ymin = Yv[0];
    double Smean = sum(cluster_size) / double(n_cls);
    double Smax = cluster_size[0], Smin = cluster_size[0];
    for (size_t cl = 1; cl < n_cls; cl++) {
      double v = Yv[cl];
      if (v > Ymax) {
        Ymax = v;
      }
      if (v < Ymin) {
        Ymin = v;
      }

      double s = cluster_size[cl];
      if (s > Smax) {
        Smax = s;
      }
      if (s < Smin) {
        Smin = s;
      }
    }
    double Vstd = Ymax - Ymin;
    double Sstd = Smax - Smin;

    /// 3. Compute score
    vectord size_score = ( cluster_size - svectord(n_cls, Smean) ) / Sstd;
    vectord value_score = ( Yv - svectord(n_cls, Ymean) ) / Vstd;
    cluster_score = size_score + value_score;

    FILE_LOG(logINFO) << "Cluster size score: " << size_score;
    FILE_LOG(logINFO) << "Cluster value score: " << value_score;
    FILE_LOG(logINFO) << "Cluster final score: " << cluster_score;

    double min_score = cluster_score[0], max_score = cluster_score[0];
    for (size_t cl = 1; cl < n_cls; cl++) {
      double sc = cluster_score[cl];
      if (sc < min_score) {
        min_score = sc;
      }

      if (sc > max_score) {
        max_score = sc;
      }
    }
    
    if (min_score < 0.0) {
      cluster_score += svectord(n_cls, abs(min_score));
      max_score += abs(min_score);
    }

    cluster_score = svectord(n_cls, max_score) - cluster_score;

  }

  // OPTIMIZATION INTERFACE

  void BayesOptBase::stepOptimization()
  {
    // Find what is the next point.
    // Update surrogate model
    
    bool retrain = ((mParameters.n_iter_relearn > 0) &&
                   ((mCurrentIter + 1) % mParameters.n_iter_relearn == 0));

    const FixedDataset* data = dynamic_cast<const FixedDataset*>(mModel->getData());
    if (data != NULL) {
      const size_t cuurent_size = data->getNSamples();
      retrain = retrain || cuurent_size == mParameters.fixed_dataset_size || cuurent_size + mParameters.n_parallel_samples >= mParameters.fixed_dataset_size;
    }

    vectord xNext;
    double yNext;
    
    // Allows simple sample batchparallelization
    if(mParameters.par_type != PAR_NONE)
      {
        vecOfvec xNextSamples(mParameters.n_parallel_samples);
        vectord yNextSamples(mParameters.n_parallel_samples);

        const bool isBBO = mParameters.par_type == PAR_MCMC; // TODO: set this param from parameters
        const bool divfEnabled = mParameters.diversity_level > 0.0;
        const bool multiSolsEnabled = mParameters.num_solutions > 1;
        const bool e_mebo = mParameters.diversity_level == 0.0; // TODO cambiar

        bool explorative_sampling, default_sampling;
        if (multiSolsEnabled) {
          const double c1 = mParameters.n_init_samples + (mCurrentIter+1) * mParameters.n_parallel_samples; // number of functions evals
          const double c2 = mParameters.n_init_samples + mParameters.n_iterations * mParameters.n_parallel_samples; // total number of functions evals
          
          if (divfEnabled) { // clustering enabled
            if (c1 < c2*0.4) {
              explorative_sampling = true;
              cluster_sampling = false;
              default_sampling = true;
            } else if (c1 < c2*0.5) {
              explorative_sampling = false;
              cluster_sampling = false;
              default_sampling = true;
            } else if (c1 < c2*0.7) {
              explorative_sampling = false;
              cluster_sampling = (mCurrentIter+1) % 2 == 0;
              default_sampling = !cluster_sampling;
            } else if (c1 < c2*0.9) {
              explorative_sampling = false;
              cluster_sampling = (mCurrentIter+1) % 3 != 0;
              default_sampling = !cluster_sampling;
            } else {
              explorative_sampling = false;
              cluster_sampling = true;
              default_sampling = false;
            }
            
          } else {
            explorative_sampling = e_mebo && (c1 <= c2*0.4 || ((mCurrentIter+1) % 3 == 0 && c1 >= c2*0.5));
            cluster_sampling = false;
            default_sampling = true;
          }

        } else {
          cluster_sampling = false;
          explorative_sampling = false;
          default_sampling = true;
        }

        if (cluster_sampling) {
          using boost::numeric::ublas::subrange;
          FILE_LOG(logINFO) << "=====Sampling batch with cluster strategy=====";
          
          /// 1. Get best clusters
          centroids.clear();

          vecOfvec Qs;
          std::vector<double> Yv;
          std::vector< std::vector<size_t> > clusters;

          compute_clusters(
            mModel->getData(), mParameters.num_solutions, mParameters.diversity_level, [this](const vectord& x) -> vectord{ return remapPoint(x) ;},
            Qs, Yv, clusters, centroids
          );

          size_t num_clusters = clusters.size();
          if (num_clusters < mParameters.num_solutions * 0.5) {
            FILE_LOG(logINFO) << "WARNING: there are not enough diverse zones " << num_clusters << "/" << mParameters.num_solutions << "!!!";
            cluster_sampling = false;
            default_sampling = true;
            explorative_sampling = true;
          } else {
            /// 2. Update Cluster criteria

            mModel->updateCriteria(vectord(1, -1)); // set criteria to batch combined

            size_t ndim = centroids[0].size();

            vectord params(2 + num_clusters * ndim + num_clusters);
            params[0] = num_clusters;
            params[1] = ndim;
            
            // max_dist_centroid.clear();
            // max_dist_centroid = vectord(num_clusters);
            cluster_lb.clear();
            cluster_ub.clear();

            vectord cl_size = zvectord(num_clusters);
            vectord cl_min_value = zvectord(num_clusters);
            size_t s = 2;
            for (size_t cl = 0; cl < num_clusters; cl++) {
              size_t end = s + ndim;
              
              vectord cx = centroids[cl];
              subrange(params, s, end) = cx;

              double c_min = 999999999999999; // min value
              // double max_d = 0.02;
              vectord cl_lb = cx - svectord(ndim, 0.04);
              vectord cl_ub = cx + svectord(ndim, 0.04);
              for (size_t j = 0; j < clusters[cl].size(); j++) {
                size_t pt_idx = clusters[cl][j];
                
                double v = Yv[pt_idx];
                if (v < c_min) {
                  c_min = v;
                }

                /*double d = norm_2(Qs[pt_idx] - cx);
                if (d > max_d) {
                  max_d = d;
                }*/
                vectord pt = Qs[pt_idx];
                for (size_t d = 0; d < ndim; d++) {
                  double v = pt[d];
                  if (v < cl_lb[d]) {
                    cl_lb[d] = v;
                  }
                  if (v > cl_ub[d]) {
                    cl_ub[d] = v;
                  }
                }
              }
              
              params[end] = c_min;
              
              cl_size[cl] = clusters[cl].size();
              cl_min_value[cl] = c_min;

              // max_dist_centroid[cl] = max_d * (max_d < 0.04 ? 2 : 1);
              cluster_lb.push_back(cl_lb);
              cluster_ub.push_back(cl_ub);

              std::string c_s = "(";
              for (auto& p : cx) {
                c_s += " " + std::to_string(p) + ",";
              }
              c_s += ")";
              FILE_LOG(logINFO) << "Cluster (" << cl+1 << "):\n"
                                << "\tNum points: " << cl_size[cl] << "\n"
                                << "\tMin value: " << c_min << "\n"
                                << "\tCentroid: " << c_s
                                << "\tBounding box: " << cl_lb << " - " << cl_ub;
              
              s = end + 1;
            }
            
            mModel->updateCriteria(params); // update criteria with cluster info

            /// 3. Compute the clusters score
            vectord clusters_score;
            cluster_score(cl_size, cl_min_value, mModel->getData()->getValueAtMaximum(), clusters_score);

            /// 4. Sample batch
            // 4.1 Get number of points per cluster
            vectord cl_pts = zvectord(num_clusters);
            for (size_t i = 0; i < mParameters.n_parallel_samples; i++){
              double max_sc = -1;
              size_t cl_idx = 0;
              for (size_t cl = 0; cl < num_clusters; cl++) {
                double sc = clusters_score[cl];
                if (sc > max_sc) {
                  max_sc = sc;
                  cl_idx = cl;
                }
              }

              FILE_LOG(logINFO) << "Point " << i+1 << "/" << mParameters.n_parallel_samples;
              FILE_LOG(logINFO) << "\tCluster score: " << clusters_score;
              FILE_LOG(logINFO) << "\tCluster selected: " << cl_idx+1;

              cl_pts[cl_idx]++;
              clusters_score[cl_idx] = clusters_score[cl_idx] * (1 - 0.1 * cl_pts[cl_idx]);
            }

            // 4.2 Sample points
            size_t current_pt = 0;
            for (size_t cl = 0; cl < num_clusters; cl++)
            {
              current_cluster = cl; // update cluster to sampling

              size_t npts = cl_pts[cl];
              if (npts > 0) {
                FILE_LOG(logINFO) << "Sampling Cluster " << cl+1;
                if (npts == 1) {
                  xNextSamples[current_pt] = nextPoint();
                  FILE_LOG(logINFO) << "\tPoint " << current_pt+1 << xNextSamples[current_pt];
                  current_pt++;
                } else {
                  const int nsamples = npts * (mParameters.n_parallel_samples < 5 ? 5 : mParameters.n_parallel_samples);
                  vecOfvec cluster_candidates(nsamples);
                  for(size_t j = 0; j < nsamples; j++) {
                    cluster_candidates[j] = nextPoint();
                  }

                  std::vector<size_t> selection;
                  select_batch_candidates(cluster_candidates, npts, selection);

                  for (auto pt_idx : selection)
                  {
                    xNextSamples[current_pt] = cluster_candidates[pt_idx];
                    FILE_LOG(logINFO) << "\tPoint " << current_pt+1 << xNextSamples[current_pt];
                    current_pt++;
                  }
                }
              }

              mModel->updateCriteria(zvectord(0)); // update current cluster on criteria
            }
            
            /// 5. Reset

            mModel->updateCriteria(vectord(1, -2)); // set criteria to batch combined

            FILE_LOG(logINFO) << "END CLUSTER SAMPLING";
          }
        }
        
        if (default_sampling) {
          if (multiSolsEnabled) {
            if (divfEnabled) mModel->updateCriteria(vectord(1, -2)); // set criteria to batch combined
            mModel->updateCriteria(zvectord(0)); // reset batch

            if (explorative_sampling) {
              FILE_LOG(logINFO) << "=====Sampling batch with explorative strategy=====";

              /// 1. Sample batch_size * N points
              const size_t num_samples_per_pt = 5;
              vecOfvec batchCandidates(mParameters.n_parallel_samples*num_samples_per_pt);
              for(size_t i = 0; i < mParameters.n_parallel_samples; i++)
              {
                for (size_t j = 0; j < num_samples_per_pt; j++) {
                  batchCandidates[i*num_samples_per_pt + j] = nextPoint();
                }
                
                if (isBBO) mModel->updateCriteria(batchCandidates[i*num_samples_per_pt]); // update batch with random candidate
              }

              /// 2. Select batch_size points that maximize euclidean distance
              std::vector<size_t> selection;
              select_batch_candidates(batchCandidates, mParameters.n_parallel_samples, selection);

              for(size_t i = 0; i < selection.size(); i++)
              {
                size_t pt_idx = selection[i];
                FILE_LOG(logINFO) << "Point: " << (i+1) << " / " << mParameters.n_parallel_samples;
                xNextSamples[i] = batchCandidates[pt_idx];
                FILE_LOG(logINFO) << "Point " << pt_idx+1 << ": " << xNextSamples[i];
                mModel->updateCriteria(xNextSamples[i]); // update batch
              }
            }
          }
          
          if (!explorative_sampling) {
            for(size_t i=0; i<mParameters.n_parallel_samples; i++)
            {
              FILE_LOG(logINFO) << "Sampling point: " << (i+1) << " / " << mParameters.n_parallel_samples;
              xNextSamples[i] = nextPoint();
              FILE_LOG(logINFO) << "Point " << (i+1) << ": " << xNextSamples[i];
              if (isBBO) mModel->updateCriteria(xNextSamples[i]); // update batch
            }
          }
        }

        // evaluate samples
        if (isBBO) {
          yNextSamples = evaluateSamplesInternal(xNextSamples);
        } else {
          //#pragma omp parallel for
          for(size_t i=0; i<mParameters.n_parallel_samples; i++)
          {
            yNextSamples[i] = evaluateSampleInternal(xNextSamples[i]);
          }
        }
        
        // add samples to model
        for(size_t i=0; i<mParameters.n_parallel_samples; i++)
          {
            mModel->addSample(xNextSamples[i],yNextSamples[i]);
            plotStepData(mCurrentIter,xNextSamples[i],yNextSamples[i]); // mCurrentIter*mParameters.n_parallel_samples+i
            // TODO(Javier): Modify updateSurrogateModel to deal with multiple new samples at once.
            if(!retrain) mModel->updateSurrogateModel();
          }
        
        //TODO(Javier): Should it use a randomly picked sample or the max yNext?
        xNext = xNextSamples[0];
        yNext = yNextSamples[0];
      }
    // Default behaviour
    else
      {

	if ((mParameters.force_jump > 0) and checkStuck(yNext))
	  {
	    FILE_LOG(logINFO) << "Forced random query!";
	    xNext = samplePoint();			    
	  }
	else
	  {
	    xNext = nextPoint();
	  }

	yNext = evaluateAndAddNextPoint(xNext, retrain);
      }

    if (retrain)  // Full update
      {
        mModel->updateHyperParameters();
        mModel->fitSurrogateModel();
      }


    mModel->updateCriteria(xNext);
    mCurrentIter++;

    mDebugBestPoints.push_back(getFinalResult(false));

    // Save state if required
    saveOptimizationIfRequired();
  }


  void BayesOptBase::initializeOptimization()
  {
    size_t nSamples = mParameters.n_init_samples;

    mCurrentIter = 0;
    mCounterStuck = 0;
    mYPrev = 0.0;

    // Generate xPoints for initial sampling
    matrixd xPoints(nSamples,mDims);
    vectord yPoints(nSamples,0);

    // Save generated xPoints before its evaluation
    generateInitialPoints(xPoints);
    mModel->setSamples(xPoints);
    saveOptimizationIfRequired();

    // Generate non evaluated samples into model
    while(generateNonEvaluatedSample())
      {
        saveOptimizationIfRequired();
      }


    if(mParameters.verbose_level > 0)
      {
        mModel->plotDataset(logDEBUG);
      }

    mModel->updateHyperParameters();
    mModel->fitSurrogateModel();
  }

  vectord BayesOptBase::getFinalResult(bool useMean)
  {
    if (mParameters.input_noise_variance > 0)
      {
        vectord minPoint = mModel->getData()->mX[0];
        double minMean = mUT->integratedOutcome(minPoint);
        mIntegratedIndex = 0;
        
        for(size_t i=1; i < mModel->getData()->getNSamples(); i++)
          {
            vectord point = mModel->getData()->mX[i];
            double mean = mUT->integratedOutcome(point);

            if(mean < minMean)
              {
                minMean = mean;
                minPoint = point;
                mIntegratedIndex = i;
              }
          }
        return remapPoint(minPoint);
      }
    else
      {
        return remapPoint(getPointAtMinimum(useMean));
      }
  }

  double BayesOptBase::getFinalValue(bool useMean)
  {
    if (mParameters.input_noise_variance > 0)
      {
        return mModel->getData()->getSampleY(mIntegratedIndex);
      }
    else
      {
        return getValueAtMinimum();
      }
  }


  void BayesOptBase::getFinalResults(vecOfvec& points, vectord& values, bool useMean) {
    if (mParameters.num_solutions > 1 && mParameters.diversity_level > 0.0) {
      // TODO
      for (size_t i = 0; i < mParameters.num_solutions; i++) {
        points.push_back(getFinalResult());
      }
      values = vectord(mParameters.num_solutions, getFinalValue());
    } else {
      points.push_back(getFinalResult());
      values = vectord(1, getFinalValue());
    }
  }

  // SAVE-RESTORE INTERFACE
  void BayesOptBase::saveOptimization(BOptState &state)
  {
    // BayesOptBase members
    state.mCurrentIter = mCurrentIter;
    state.mCounterStuck = mCounterStuck;
    state.mYPrev = mYPrev;

    state.mParameters = mParameters;

    // Samples
    state.mX = mModel->getData()->mX;
    state.mY = mModel->getData()->mY;

    state.mDebugBestPoints = mDebugBestPoints;
  }

  void BayesOptBase::restoreOptimization(BOptState state)
  {
    // Restore parameters
    mParameters = state.mParameters;

    mCurrentIter = state.mCurrentIter;
    mCounterStuck = state.mCounterStuck;
    mYPrev = state.mYPrev;

    // Posterior surrogate model
    mModel.reset(PosteriorModel::create(mDims, mParameters, mEngine));

    // Put mX and mY into model
    matrixd xPoints(state.mX.size(), state.mX[0].size());
    for(size_t i=0; i<state.mX.size(); i++)
      {
        row(xPoints, i) = state.mX[i];
      }

    mModel->setSamples(xPoints, state.mY);
    saveOptimizationIfRequired();

    // Generate non evaluated samples into model
    while(generateNonEvaluatedSample())
      {
        saveOptimizationIfRequired();
      }

    if(mParameters.verbose_level > 0)
      {
        mModel->plotDataset(logDEBUG);
      }

    // Calculate the posterior model
    mModel->updateHyperParameters();
    mModel->fitSurrogateModel();

    // Check if optimization has already finished
    if(mCurrentIter >= mParameters.n_iterations)
      {
        FILE_LOG(logINFO) << "Optimization has already finished, delete \""
			  << mParameters.load_filename
			  << "\" or give more n_iterations in parameters.";
      }
  }


  // GETTERS AND SETTERS
  // Potential inline functions. Moved here to simplify API and header
  // structure.

  ProbabilityDistribution* BayesOptBase::getPrediction(const vectord& query)
  { return mModel->getPrediction(query); };

  const Dataset* BayesOptBase::getData()
  { return mModel->getData(); };

  Parameters* BayesOptBase::getParameters()
  {return &mParameters;};

  double BayesOptBase::getValueAtMinimum()
  {
    const FixedDataset* data = dynamic_cast<const FixedDataset*>(mModel->getData());
    if (data == NULL) {
      return mModel->getValueAtMinimum();
    } else {
      return data->getValueAtGlobalMinimum();
    }
  };

  double BayesOptBase::evaluateAndAddNextPoint(const vectord& xNext, bool retrain)
  {
    double yNext = evaluateSampleInternal(xNext);
    mModel->addSample(xNext,yNext);
    if (!retrain) mModel->updateSurrogateModel();
    plotStepData(mCurrentIter,xNext,yNext); //TODO(Javier): Remove this from evaluate and add
    return yNext;
  }

  double BayesOptBase::evaluateCriteria(const vectord& query)
  {
    if (checkReachability(query))
      {
        if (mParameters.input_noise_variance > 0 && mParameters.use_unscented_criterion)
          {
            return mUT->integratedCriteria(query);
          }
        return mModel->evaluateCriteria(query);
      }
    else
      {
        return 0.0;
      }
  }

  size_t BayesOptBase::getCurrentIter()
  {return mCurrentIter;};


  // PROTECTED
  vectord BayesOptBase::getPointAtMinimum(bool useMean)
  {
    const FixedDataset* data = dynamic_cast<const FixedDataset*>(mModel->getData());
    if (data == NULL) {
      return mModel->getPointAtMinimum(useMean);
    } else {
      return data->getPointAtGlobalMinimum();
    }
  };

  double BayesOptBase::evaluateSampleInternal( const vectord &query )
  {
    const double yNext = evaluateSample(remapPoint(query));
    if (yNext == HUGE_VAL)
      {
	    throw std::runtime_error("Function evaluation out of range");
      }
    return yNext;
  };

  vectord BayesOptBase::evaluateSamplesInternal( const vecOfvec &queries ) {
    const size_t nQs = queries.size();
    vecOfvec mapped_queries(nQs);
    for (size_t i = 0; i < nQs; i++) {
      mapped_queries[i] = remapPoint(queries[i]);
    }

    vectord yNexts = evaluateSamples(mapped_queries);
    if (yNexts.size() != nQs) {
      throw std::runtime_error("Number of values is different!: Expected " + std::to_string(nQs) + ", obtained " + std::to_string(yNexts.size()));
    }

    for (size_t i = 0; i < nQs; i++) {
      const double yNext = yNexts[i];

      if (yNext == HUGE_VAL) {
	      throw std::runtime_error("Function evaluation out of range in query: " + std::to_string(i+1));
      }
    }

    return yNexts;
    
  }

  void BayesOptBase::plotStepData(size_t iteration, const vectord& xNext,
				     double yNext)
  {
    if(mParameters.verbose_level >0)
      {
        size_t total_samples;
        const FixedDataset* data = dynamic_cast<const FixedDataset*>(mModel->getData());
        if (data == NULL) {
          total_samples = getData()->getNSamples();
        } else {
          total_samples = data->getTotalNSamples();
        }
        FILE_LOG(logINFO) << "Iteration: " << iteration+1 << " of "
                  << mParameters.n_iterations << " | Total samples: "
                  << total_samples;
        FILE_LOG(logINFO) << "Query: "         << remapPoint(xNext); ;
        FILE_LOG(logINFO) << "Query outcome: " << yNext ;
        FILE_LOG(logINFO) << "Best query: "    << getFinalResult(); //TODO(Javier): First fix plotStepData()
        FILE_LOG(logINFO) << "Best outcome: "  << getFinalValue();
      }
  } //plotStepData

  void BayesOptBase::saveOptimizationIfRequired()
  {
    if(mParameters.load_save_flag == 2 || mParameters.load_save_flag == 3)
      {
        BOptState state;
        saveOptimization(state);
        state.saveToFile(mParameters.save_filename);
      }
  }

  bool BayesOptBase::generateNonEvaluatedSample()
  {
    const Dataset* data = mModel->getData();

    if(data->mY.size() < data->mX.size())
      {
        // Index of the first non evaluated sample
        const size_t index = data->mY.size();

        double y = evaluateSampleInternal(data->mX[index]);
        mModel->addSample(y);
        return true;
      }
    else
      {
        return false;
      }
  }

  // PRIVATE MEMBERS
  vectord BayesOptBase::nextPoint()
  {
    //Epsilon-Greedy exploration (see Bull 2011)
    if ((mParameters.epsilon > 0.0) && (mParameters.epsilon < 1.0))
      {
        randFloat drawSample(mEngine,realUniformDist(0,1));
        double result = drawSample();
        FILE_LOG(logINFO) << "Trying random jump with prob:" << result;
        if (mParameters.epsilon > result)
          {
            FILE_LOG(logINFO) << "Epsilon-greedy random query!";
            return samplePoint();
          }
      }

    // If jump is not successful
    vectord Xnext(mDims);

    // GP-Hedge and related algorithms
    if (mModel->criteriaRequiresComparison())
      {
	mModel->setFirstCriterium();
	do
	  {
	    findOptimal(Xnext);
	  }
	while (mModel->setNextCriterium(Xnext));

	std::string name = mModel->getBestCriteria(Xnext);
	FILE_LOG(logINFO) << name << " was selected.";
      }
    else  // Standard "Bayesian optimization"
      {
        switch(mParameters.par_type)
        {
        case PAR_RANDOM:
          {
            randFloat sample( mEngine, realUniformDist(0,1) );
            for(vectord::iterator it = Xnext.begin(); it != Xnext.end(); ++it)
              {
                *it = sample();
              } 
            break;
          }
        case PAR_MCMC:
          {
            sampleCriteria(Xnext);
            break;
          }
        case PAR_REJ_SAMPLING:
          {
            rejectionSamplingCriteria(Xnext);
            break;
          }
        case PAR_PAR_TEMPERING:
          {
            parallelTemperingCriteria(Xnext);
            break;
          }
        case PAR_THOMPSON:
          {
            findOptimal(Xnext);
            break;
          }        
        case PAR_NONE:
        case PAR_BBO:
        case PAR_ERROR:
        default: 
          FILE_LOG(logDEBUG) << "------ Optimizing criteria ------";
          findOptimal(Xnext);
        }
	
      }
    return Xnext;
  }

  void BayesOptBase::setNormalizeCriteria(bool normalize)
  {
    mNormalizeCriteria = normalize;
  }
  
  void BayesOptBase::setBoltzmannPolicy(bool activate, double temperature, double tempReduction, double freezingReduction)
  {
    mUseBoltzmannPolicy = activate;
    if( activate )
      {
        mInitTemperature = temperature;
        mCurrentTemperature = temperature;
        mTempReduction = tempReduction;

        mFreezingTemperature = mInitTemperature;
        mFreezingReduction = freezingReduction;
      }
  }
  
  void BayesOptBase::startAnnealingCycle()
  {
    mCurrentTemperature = mInitTemperature;
    mFreezingTemperature *= mFreezingReduction;
  }
  
  void BayesOptBase::stepAnnealing()
  {
    mCurrentTemperature -= mTempReduction;
  }
  
  double BayesOptBase::modifyCriteria(double value, bool normalize, bool clamp, bool useBoltzmann, double temperature)
  {
    if (normalize) // normalizes the temperature
      {
        double maxY = getData ()->getValueAtMaximum ();
        double minY = getData ()->getValueAtMinimum ();
        temperature = temperature * (maxY - minY);
      }
    if (clamp) // clamp min[-1,0)
      {
        if (value > 0) value = 0;
        else if (value < -1) value = -1;
      }
    if (useBoltzmann)
      {
        value = -exp(-value/temperature);
      }
    return value;
  }

  double BayesOptBase::criteriaCorrection(double value)
  {
    return modifyCriteria(value, mNormalizeCriteria, false, mUseBoltzmannPolicy, mCurrentTemperature);
  }

  bool BayesOptBase::checkStuck(double yNext)
  {
    // If we are stuck in the same point for several iterations, try a
    // random jump!
    if (std::pow(mYPrev - yNext,2) < mParameters.noise)
      {
	mCounterStuck++;
	FILE_LOG(logDEBUG) << "Stuck for "<< mCounterStuck << " steps.";
      }
    else
      {
	mCounterStuck = 0;
      }
    if (mCounterStuck > mParameters.force_jump)
      {
	mCounterStuck = 0;
	return true;
      }
    
    return false;
  }


} //namespace bayesopt

