% 
% -------------------------------------------------------------------------
%    This file is part of BayesOpt, an efficient C++ library for 
%    Bayesian optimization.
%
%    Copyright (C) 2011-2014 Ruben Martinez-Cantin <rmcantin@unizar.es>
%
%    BayesOpt is free software: you can redistribute it and/or modify it 
%    under the terms of the GNU General Public License as published by
%    the Free Software Foundation, either version 3 of the License, or
%    (at your option) any later version.
%
%    BayesOpt is distributed in the hope that it will be useful, but 
%    WITHOUT ANY WARRANTY; without even the implied warranty of
%    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
%    GNU General Public License for more details.
%
%    You should have received a copy of the GNU General Public License
%    along with BayesOpt.  If not, see <http://www.gnu.org/licenses/>.
% ------------------------------------------------------------------------
%
clear all, close all
addpath('walker')

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
disp('Continuous optimization');
fun = 'run_walker'; n = 8;
lb = zeros(n,1);
ub = ones(n,1);

%%% Nonstationary
% d_center = zeros(n*2,2);
% d_std = zeros(n*2,2);
% for i=1:n*2
%     d_center(i,:) = [0.5, 0.5];
%     d_std(i,:) = [1, 100];
% end;
% parfor iter = 10:30
%     params = struct('n_iterations',30,'n_init_samples',10, ...
%                     'l_type','mcmc', 'n_iter_relearn', 1, ...
%                     'random_seed', iter, ...
%                     'kernel_name', 'kNonSt(kMaternARD5,kMaternARD5)', ...
%                     'kernel_domain_center', d_center, ... 
%                     'kernel_domain_std', d_std);
%     bayesoptlog(fun,n,['walkernst',num2str(iter),'.log'],params,lb,ub)
% end

%%% Warped
% h_mean = [ones(1,n*2) * 2  , ones(1,n)];
% h_std  = [ones(1,n*2) * 0.5, ones(1,n) * 10];
% 
% parfor iter = 10:30    
%     params = struct('n_iterations',30,'n_init_samples',10, ...
%                     'l_type','mcmc', 'n_iter_relearn', 1, ...
%                     'random_seed', iter, ...
%                     'kernel_name', 'kWarp(kMaternARD5)', ...
%                     'kernel_hp_mean', h_mean, 'kernel_hp_std', h_std);
%     bayesoptlog(fun,n,['walkerwarp',num2str(iter),'.log'],params,lb,ub)
% end

%%% Standard
parfor iter = 10:30  
    params = struct('n_iterations',30,'n_init_samples',10, ...
                    'l_type','mcmc', 'n_iter_relearn', 1, ...
                    'random_seed', iter);
    bayesoptlog(fun,n,['walkerstd',num2str(iter),'.log'],params,lb,ub)
end;

%%% Active
% parfor iter = 10:30  
%     params = struct('n_iterations',30,'n_init_samples',10, ...
%                     'l_type','mcmc', 'n_iter_relearn', 1, ...
%                     'random_seed', iter, 'active_coef', 0.9);
%     bayesoptlog(fun,n,['walkeractive900_',num2str(iter),'.log'],params,lb,ub)
% end;