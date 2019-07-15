#importing keras modules
import keras
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers import Input, Merge
from keras.layers import merge
from keras.models import Model
from keras import backend as K
from keras.callbacks import EarlyStopping

from keras.utils.visualize_util import plot
from keras.wrappers.scikit_learn import KerasClassifier, KerasRegressor
from hyperopt import Trials, STATUS_OK, tpe
from hyperas import optim
from hyperas.distributions import choice, uniform, conditional
from hyperopt import fmin, tpe, hp
from tempfile import TemporaryFile


#importing graphics and numpy
import numpy
import pydot
print pydot.find_graphviz()

#importing Custom Libraries
#manual functions
import ArrayFunctions
import MathFunctions
import TimeScaleFunctions

import DataFunctions
import NNFunctions
import PlotFunctions
import InterpolateFunctions
import PeriodFunctions
import NNFun_PSB
import build_lstm_v1
#define hyperopt search space
import hyperopt.pyll.stochastic

from sklearn.metrics import r2_score
from sklearn.datasets import fetch_20newsgroups_vectorized
from sklearn.feature_selection import chi2

#scipy library
from scipy.stats import pearsonr

seed = 7
numpy.random.seed(seed)

from datetime import  datetime

#EnergyData
date_start = '5/19/15 12:00 AM'
date_end = '5/13/16 11:59 PM'
ref_date = date_start
std_inv = 60 #in minutes
std_inv2 = 10 #in minutes
#delta = 0.03571429 #ideally you'd like to get the value of delta directly
#delta = 0.05
#delta = 0.14285714
delta =  0.11111111
#delta = 0.5

data = DataFunctions.read_PSB_csv(date_start, date_end, 5, 20) #read data at 5 min resolutions
conv_critical, crac_critical, crac_normal, conv_normal, HVAC_critical, Elev, HVAC_normal, elec_total = DataFunctions.PSB_elec_split(data) #split by end uses
H_t2 = DataFunctions.prepare_energy_PSB(conv_normal, std_inv2) #Energy Consumption Data for multi timescale analysis

#performing linear interpolation on H_t2 firs
small_list2, large_list2 = InterpolateFunctions.interpolate_main_v2(H_t2, 30) #Allowing for 30 consecutive timesteps
H_t2[0, :] = 0
H_t2 = InterpolateFunctions.interp_linear(H_t2, small_list2)

#H_t = DataFunctions.prepare_energy_PSB(HVAC_critical, std_inv)
H_t2 = H_t2[:, None]
H_t= DataFunctions.fix_energy_intervals(H_t2, std_inv2, std_inv)
H_t = DataFunctions.fix_high_points(H_t)

#Weather Data for training
weather_train = DataFunctions.prepare_weather_WBB(date_start, date_end, std_inv)
weather_train2 = DataFunctions.prepare_weather_WBB(date_start, date_end, std_inv2)

#feature vectors
X_sch_t = DataFunctions.compile_features(H_t, date_start, std_inv)
X_sch_t2 = DataFunctions.get_feature_low_res(X_sch_t, std_inv, std_inv2)

# Getting timescale features
timescales = numpy.array([325, 351])
T_sch_t = TimeScaleFunctions.timescale_101(X_sch_t, date_start, ref_date, timescales)
T_sch_t2 = DataFunctions.get_feature_low_res(T_sch_t, std_inv, std_inv2)
T_sch_t2 = T_sch_t2[:, 0:2]

train_data = numpy.concatenate((weather_train[:, 0:2], X_sch_t, T_sch_t), axis=1)
train_data2 = numpy.concatenate((weather_train2[:, 0:2], X_sch_t2, T_sch_t2), axis=1)
#train_data = X_sch_t
#train_data2 = X_sch_t2
H_rms = MathFunctions.rms_flat(H_t)

#normalizing data
H_min, H_max = DataFunctions.get_normalize_params(H_t)
H_t = H_t/H_max

#normalizing H_t2
H_min2, H_max2 = DataFunctions.get_normalize_params(H_t2)
H_t2 = H_t2/H_max2

#Computing Entropy
S, unique, pk = DataFunctions.calculate_entropy(H_t)
#print "The entropy value is: ", S

# Block to interpolate
cons_points = 5
s0 = 245
start_day = 250
end_day  = 260
s1 = 265

choice = 1

if choice == 1:
    H_t = numpy.load('H1_file_CONV1.npy')
elif choice == 2:
    X_t, Y_t, X_e, Y_e, train_list, test_list = InterpolateFunctions.train_test_split(train_data, H_t)
    best = InterpolateFunctions.imputate_optimize(X_t, Y_t)
    Y_p = NNFun_PSB.PSB_model_DL(X_t, Y_t, X_e, best)
    H_t = InterpolateFunctions.organize_pred(H_t, Y_p, test_list)
    #numpy.save('Ht_file_HVAC1.npy', H_t)
else:
    H1 = H_t.copy()
    small_list, large_list = InterpolateFunctions.interpolate_main(train_data, H_t, start_day, end_day, cons_points)
    H_t = InterpolateFunctions.interp_linear(H_t, small_list) #H_t is beign changed as numpy arrays are mutable
    H1 = InterpolateFunctions.interp_linear(H1, small_list)

    PlotFunctions.Plot_interp_params()
    H_t = H_t[:, None] #changing numpy array shape to fit the function
    train_interp, dummy1, dummy2 = DataFunctions.normalize_103(train_data, train_data, train_data)
    Y_t, Y_NN = InterpolateFunctions.interp_LSTM(train_interp, H_t, large_list)

    PlotFunctions.Plot_interpolate(H1[s0*24:start_day*24], Y_t[start_day*24:end_day*24], Y_NN[start_day*24:end_day*24], H1[start_day*24:end_day*24], H1[end_day*24:s1*24])
    e_interp = InterpolateFunctions.interpolate_calculate_rms(H1[start_day*24:end_day*24], Y_t[start_day*24:end_day*24])
    e_NN = InterpolateFunctions.interpolate_calculate_rms(H1[start_day*24:end_day*24], Y_NN[start_day*24:end_day*24])

    print e_interp
    print e_NN
    H_t = Y_t.copy()
    #H_t = Y_NN.copy()
    #numpy.save('H1_fill_HVAC1.npy', H_t)


##block to fill in points for H_t2
choice = 0

#filling out the large chunks in H_t2
small_list2, large_list2 = InterpolateFunctions.interpolate_main_v2(H_t2, 30) #Allowing for 30 consecutive timesteps
small_list, large_list = InterpolateFunctions.interpolate_main_v2(H_t, 5)


####Computing derivative of H_t2 before interpolated values are filled out
deriv_Ht2 = numpy.diff(H_t2, axis=0)
S2, unique2, pk2 = DataFunctions.calculate_entropy(deriv_Ht2)
print S2, unique2, pk2

####This block is to fill up using low-res predictions to high resolution missing data
H_t2 = DataFunctions.datafill_low_to_high(H_t.copy()*H_max, H_t2.copy(), large_list2, (std_inv/std_inv2), H_max2)
H1 = H_t2.copy()
H_t2 = DataFunctions.fix_10min_data(H_t2, delta, small_list2, large_list2)
max_unit = numpy.rint(1/delta)
H_t2 = DataFunctions.make_10min_tensor(H_t2, max_unit, delta)


#Aggregating data on a daily basis
H_t = H_t[:, None]
conv_hour_to_day = 24
H_mean_t, H_sum_t, H_min_t, H_max_t = DataFunctions.aggregate_data(H_t, conv_hour_to_day)
w_mean_t, w_sum_t, w_min_t, w_max_t = DataFunctions.aggregate_data(weather_train, conv_hour_to_day)

#gettomg features for a single day
#PlotFunctions.Plot_single(H_mean_t)
X_day_t = DataFunctions.compile_features(H_sum_t, date_start, 24*60)

seq_length = 6*24
Y_t2 = PeriodFunctions.set_period_array(H_t2, seq_length)
Y_t2 = PeriodFunctions.reshape_array(Y_t2, seq_length)

#To determine rho, we need to collapse the solenoid on to its face
#We need to do this for every binary variable
Z = PeriodFunctions.collapse_solenoid(Y_t2, seq_length)
loc_v = PeriodFunctions.compute_site(Z[:, 7])
j = 1
x = PeriodFunctions.get_x_array(loc_v, j, seq_length)
rho = PeriodFunctions.calculate_rho(2*j-1, len(loc_v), x)
rho_max = numpy.amax(rho)

print Z
print loc_v
print x
print rho
print rho_max