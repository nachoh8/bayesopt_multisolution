from keras import backend as K
from keras.datasets import boston_housing
from keras.layers import Dense
from keras.models import Sequential
from keras.optimizers import Adam, RMSprop

import numpy as np
from sklearn.model_selection import KFold

import argparse
import sys
import traceback
import bayesopt

class Function(object):

  def __init__(self, outlier_ratio=0.0):
    train, test = boston_housing.load_data()
    self.X_train, self.Y_train = train
    self.X_test, self.Y_test = test
    self.batch_size = 5
    self.outlier_ratio = outlier_ratio

  def call(self, params):
    print params
    epochs = 5 if np.random.random() < self.outlier_ratio else 20
    # epochs = 20
    # if np.random.random() < self.outlier_ratio:
    #   idx = np.random.randint(50, size=2)
    #   X_train = self.X_train[idx, :]
    #   Y_train = self.Y_train[idx]
    # else:
    #   X_train, Y_train = self.X_train, self.Y_train
    model = Sequential()
    model.add(Dense(int(params[0][3]), kernel_initializer="normal", activation="relu", input_shape=(13,)))
    model.add(Dense(1))
    rmsprop = RMSprop(
      lr=10**params[0][0],
      rho=1-10**params[0][1],
      epsilon=10e-8,
      decay=10**params[0][2],
    )
    adam = Adam(
      lr=10**params[0][0],
      beta_1=1-10**params[0][1],
      beta_2=1-10**params[0][2],
      epsilon=10e-8,
      decay=0.0,
    )
    model.compile(loss='mean_squared_error', optimizer=rmsprop)
    history = model.fit(
      self.X_train,
      self.Y_train,
      batch_size=self.batch_size,
      epochs=epochs,
    #  verbose=0,
    )
    score = model.evaluate(self.X_test, self.Y_test, verbose=0)
    if np.isnan(score):
      score = np.random.rand()*100 + 100
    K.clear_session()
    print score
    return score


def parse_args(raw_args):
    parser = argparse.ArgumentParser(
        description='Robust parsing script with stated inputs.',
    )

    parser.add_argument(
        '--num-trials',
        type=int,
        required=False,
        default=20,
        help='Number of trials for this given function.',
    )

    parser.add_argument(
        '--experiment-output',
        type=str,
        required=False,
        default='./output/',
        help='folder in which results will be saved',
    )

    args = parser.parse_args(raw_args)
    return args

def run_experiment(exp_args, bopt_params={}):
    args = parse_args(exp_args)
    num_trials = args.num_trials
    exp_folder = args.experiment_output

    n = 4
    lb = np.array((-5., -5., -5., 10.))
    ub = np.array((-1., -1., -1., 100.))


    #from pympler.tracker import SummaryTracker
    #tracker = SummaryTracker()

    func_obj = Function()

    def opt_func(X):
        try:
            arr = np.zeros((1, 4))
            arr[0][0] = X[0]
            arr[0][1] = X[1]
            arr[0][2] = X[2]
            arr[0][3] = int(X[3])
            return func_obj.call(arr)
        except Exception as e:
            print 'ERROR:', str(e)
            print 'INPUT:', X
            print(traceback.format_exc())

    for r in range(num_trials):
      params = {}
      params['n_iterations'] = 10
      params['n_iter_relearn'] = 5
      params['n_init_samples'] = 10
      params['n_parallel_samples'] = 4
      params['par_type'] = 'PAR_MCMC'
      params['load_save_flag'] = 2
      params['crit_name'] = 'cEI'

      # Overwrite defaults
      params.update(bopt_params)
      params['random_seed'] = r

      strat = 'batch_sampling'
      if params['crit_name'] == 'cLCBexp' or params['crit_name'] == 'cEIexp':
          temp = 'default'
          if len(params['crit_params']) >= 2:
              temp = str(params['crit_params'][1])
          strat = 'fixedtemp_' + temp

      params['save_filename'] = exp_folder + '{strat}_{crit}_x{par}_{problem}_rep_{rep}.dat'.format(
        crit=params['crit_name'], par=params['n_parallel_samples'], rep=r,
        strat=strat, problem='kerasexample'
      )

      mvalue, x_out, error = bayesopt.optimize(opt_func, n, lb, ub, params)

    #tracker.print_diff()

if __name__ == "__main__":
    run_experiment(sys.argv[1:])
