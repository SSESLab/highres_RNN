#importing keras modules
import keras
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers import Input, Merge
from keras.layers import merge
from keras.models import Model
from keras.callbacks import EarlyStopping
from keras.utils.visualize_util import plot
from keras.wrappers.scikit_learn import KerasClassifier, KerasRegressor
from hyperopt import Trials, STATUS_OK, tpe
from hyperas import optim
from hyperas.distributions import choice, uniform, conditional
from hyperopt import fmin, tpe, hp

#importing graphics and numpy
import numpy
import pydot
print pydot.find_graphviz()

#importing Custom Libraries
#manual functions
import ArrayFunctions
import MathFunctions
import DataFunctions
import NNFunctions
import PlotFunctions
import InterpolateFunctions
import NNFun_PSB
#define hyperopt search space
import hyperopt.pyll.stochastic


seed = 7
numpy.random.seed(seed)

from datetime import  datetime

######## The actual code Starts here

#######Training data: 2015

#EnergyData
date_start = '6/1/15 12:00 AM'
date_end = '6/30/15 11:59 PM'
std_inv = 60 #in minutes

#Read data
data = DataFunctions.read_PSB_csv(date_start, date_end, 5, 20)
conv_critical, crac_critical, crac_normal, conv_normal, HVAC_critical, Elev, HVAC_normal, elec_total = DataFunctions.PSB_elec_split(data)
H_t = DataFunctions.fix_data(conv_normal)
H_t = DataFunctions.fix_energy_intervals(H_t, 5, std_inv)


PlotFunctions.Plot_single(H_t)
#schedules
X_sch_t = DataFunctions.compile_features(H_t, date_start, std_inv)

#Weather Data
weather_file = r'/home/sseslab/Documents/SLC PSB data/WBB Weather Data/WBB_2015_June.csv'
weather_train = DataFunctions.read_weather_csv(weather_file, date_start, date_end)
weather_train = DataFunctions.fix_weather_intervals(weather_train, 5, std_inv)
weather_min, weather_max = DataFunctions.get_normalize_params(weather_train)
#weather_train = DataFunctions.normalize_vector(weather_train, weather_min, weather_max)
weather_train = DataFunctions.interpolate_nans(weather_train)

X_t, Y_t, X_e, Y_e, train_list, test_list = InterpolateFunctions.train_test_split(X_sch_t, H_t)
best = InterpolateFunctions.imputate_optimize(X_t, Y_t)
Y_p = NNFun_PSB.PSB_model_DL(X_t, Y_t, X_e, best)

#Plotting imputated values
PlotFunctions.Plot_double(train_list, Y_t, test_list, Y_p, 'Actual Value', 'Interpolated value', 'ro', 'bo')

#re-organizing
H_t = InterpolateFunctions.organize_pred(H_t, Y_p, test_list)

print weather_train

#######TESST data

#EnergyData
date_start = '6/1/16 12:00 AM'
date_end = '6/30/16 11:59 PM'

#Read data
data = DataFunctions.read_PSB_csv(date_start, date_end, 5, 20)
conv_critical, crac_critical, crac_normal, conv_normal, HVAC_critical, Elev, HVAC_normal, elec_total = DataFunctions.PSB_elec_split(data)
H_e = DataFunctions.fix_data(conv_normal)
H_e = DataFunctions.fix_energy_intervals(H_e, 5, std_inv)

#schedules
X_sch_e = DataFunctions.compile_features(H_e, date_start, std_inv)

#Weather Data
weather_file = r'/home/sseslab/Documents/SLC PSB data/WBB Weather Data/WBB_2016_June.csv'
weather_test = DataFunctions.read_weather_csv(weather_file, date_start, date_end)
weather_test = DataFunctions.fix_weather_intervals(weather_test, 5, std_inv)
#weather_train = DataFunctions.normalize_vector(weather_train, weather_min, weather_max)
weather_test = DataFunctions.interpolate_nans(weather_test)

X_t, Y_t, X_e, Y_e, train_list, test_list = InterpolateFunctions.train_test_split(X_sch_e, H_e)
best = InterpolateFunctions.imputate_optimize(X_t, Y_t)
Y_p = NNFun_PSB.PSB_model_DL(X_t, Y_t, X_e, best)

#Plotting imputated values
PlotFunctions.Plot_double(train_list, Y_t, test_list, Y_p, 'Actual Value', 'Interpolated value', 'ro', 'bo')

H_e = InterpolateFunctions.organize_pred(H_e, Y_p, test_list)

## Optimize hyperparameters
best_opt = InterpolateFunctions.imputate_optimize(X_sch_t, H_t)
print best
Y_pred = NNFun_PSB.PSB_model_DL(X_sch_t, H_t, X_sch_e, best_opt)
e_deep = (MathFunctions.rms_flat(Y_pred - H_e))/(MathFunctions.rms_flat(H_e))


print Y_pred[1:100]
print e_deep
print MathFunctions.rms_flat(H_e)
print MathFunctions.rms_flat(H_e - Y_pred)
PlotFunctions.PlotEnergy(Y_pred, H_e)

e_t = numpy.squeeze(H_e) - Y_pred
print e_t[1:100]
PlotFunctions.Plot_single(e_t)