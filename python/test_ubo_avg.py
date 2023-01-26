#!/usr/bin/env python
# -------------------------------------------------------------------------
#    This file is part of BayesOpt, an efficient C++ library for
#    Bayesian optimization.
#
#    Copyright (C) 2011-2015 Ruben Martinez-Cantin <rmcantin@unizar.es>
#
#    BayesOpt is free software: you can redistribute it and/or modify it
#    under the terms of the GNU Affero General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    BayesOpt is distributed in the hope that it will be useful, but
#    WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU Affero General Public License for more details.
#
#    You should have received a copy of the GNU Affero General Public License
#    along with BayesOpt.  If not, see <http://www.gnu.org/licenses/>.
# ------------------------------------------------------------------------

import bayesopt
import numpy as np
from rkhs import rkhs_synth

# Python3 compat
if hasattr(__builtins__, 'raw_input'):
    input = raw_input

def inverse_rkhs(x):
    return -rkhs_synth(x)

rkhs_array = np.frompyfunc(rkhs_synth, 1, 1)


input_var = 0.02 ** 2

x_out = [0.0803075611517]
avg = np.mean(rkhs_array(np.random.normal(x_out[0], np.sqrt(input_var), 1000)))

print('{} -> avg: {}'.format(x_out, avg))