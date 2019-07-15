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
from tempfile import TemporaryFile

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
import build_lstm_v1
#define hyperopt search space
import hyperopt.pyll.stochastic


seed = 7
numpy.random.seed(seed)

from datetime import  datetime

######## The actual code Starts here

#######Training data: 2015-16

#EnergyData
date_start = '5/19/15 12:00 AM'
date_end = '5/18/16 11:59 PM'
std_inv = 60 #in minutes

#Read data
data = DataFunctions.read_PSB_csv(date_start, date_end, 5, 20) #read data at 5 min resolutions
conv_critical, crac_critical, crac_normal, conv_normal, HVAC_critical, Elev, HVAC_normal, elec_total = DataFunctions.PSB_elec_split(data) #split by end uses
H_t = DataFunctions.fix_data(HVAC_normal) #substract to find differences
H_t = DataFunctions.fix_energy_intervals(H_t, 5, std_inv) #convert to std_div time intervals
H_t = DataFunctions.fix_high_points(H_t)

PlotFunctions.Plot_single(H_t)

#Weather Data for training
weather_file = DataFunctions.read_weather_files()
weather_train = DataFunctions.read_weather_csv(weather_file, date_start, date_end)
weather_train = DataFunctions.fix_weather_intervals(weather_train, 5, std_inv)
weather_train = DataFunctions.interpolate_nans(weather_train)

print weather_train

#feature vectors
X_sch_t = DataFunctions.compile_features(H_t, date_start, std_inv)

train_data = numpy.concatenate((weather_train[:, 0:2], X_sch_t), axis=1)
#train_data = X_sch_t

choice= 0

if choice == 1:
    H_t = numpy.load('Ht_file.npy')
else:
    X_t, Y_t, X_e, Y_e, train_list, test_list = InterpolateFunctions.train_test_split(train_data, H_t)
    best = InterpolateFunctions.imputate_optimize(X_t, Y_t)
    Y_p = NNFun_PSB.PSB_model_DL(X_t, Y_t, X_e, best)
    H_t = InterpolateFunctions.organize_pred(H_t, Y_p, test_list)
    numpy.save('Ht_file.npy', H_t)

#normalizing data
H_min, H_max = DataFunctions.get_normalize_params(H_t)
H_t = H_t/H_max
X_min, X_max = DataFunctions.get_normalize_params(X_sch_t)
#X_sch_t = DataFunctions.normalize_vector(X_sch_t, X_min, X_max)

print H_t

#Plotting imputated values
#PlotFunctions.Plot_double(train_list, Y_t, test_list, Y_p, 'Actual Value', 'Interpolated value', 'ro', 'bo')


#Aggregating data on a daily basis
conv_hour_to_day = 24
H_mean_t, H_sum_t, H_min_t, H_max_t = DataFunctions.aggregate_data(H_t, conv_hour_to_day)
w_mean_t, w_sum_t, w_min_t, w_max_t = DataFunctions.aggregate_data(weather_train, conv_hour_to_day)

#gettomg features for a single day
#PlotFunctions.Plot_single(H_mean_t)
X_day_t = DataFunctions.compile_features(H_sum_t, date_start, 24*60)


################

###Test data: 2016
date_start = '5/19/16 12:00 AM'
date_end = '8/7/16 11:59 PM'
#std_inv = 60 #in minutes

#Read data
data = DataFunctions.read_PSB_csv(date_start, date_end, 5, 20) #read data at 5 min resolutions
conv_critical, crac_critical, crac_normal, conv_normal, HVAC_critical, Elev, HVAC_normal, elec_total = DataFunctions.PSB_elec_split(data) #split by end uses
H_e = DataFunctions.fix_data(HVAC_normal) #substract to find differences
H_e = DataFunctions.fix_energy_intervals(H_e, 5, std_inv) #convert to std_div time intervals

#Weather Data for training
weather_file = DataFunctions.read_weather_files()
weather_test = DataFunctions.read_weather_csv(weather_file, date_start, date_end) #sort by date and converts to a matrix format
weather_test = DataFunctions.fix_weather_intervals(weather_test, 5, std_inv)
weather_test = DataFunctions.interpolate_nans(weather_test)

#feature vectors
X_sch_e = DataFunctions.compile_features(H_e, date_start, std_inv)

#combining features
test_data = numpy.concatenate((weather_test[:, 0:2], X_sch_e), axis=1)
#test_data = X_sch_e


if choice == 1:
    H_e = numpy.load('He_file.npy')
else:
    X_t, Y_t, X_e, Y_e, train_list, test_list = InterpolateFunctions.train_test_split(test_data, H_e)
    best = InterpolateFunctions.imputate_optimize(X_t, Y_t)
    Y_p = NNFun_PSB.PSB_model_DL(X_t, Y_t, X_e, best)
    H_e = InterpolateFunctions.organize_pred(H_e, Y_p, test_list)
    numpy.save('He_file.npy', H_e)

H_e = H_e/H_max
#X_sch_e = DataFunctions.normalize_vector(X_sch_e, X_min, X_max)

#Plotting imputated values
#PlotFunctions.Plot_double(train_list, Y_t, test_list, Y_p, 'Actual Value', 'Interpolated value', 'ro', 'bo')

#Aggregating data on a daily basis
H_mean_e, H_sum_e, H_min_e, H_max_e = DataFunctions.aggregate_data(H_e, conv_hour_to_day)
w_mean_e, w_sum_e, w_min_e, w_max_e = DataFunctions.aggregate_data(weather_test, conv_hour_to_day)

#gettomg features for a single day
#PlotFunctions.Plot_single(H_mean_e)
X_day_e = DataFunctions.compile_features(H_sum_e, date_start, 24*60)

#####
#Implement daily network here

#  `
#print H_mean_t.shape

# Reshaping array into (#of days, 24-hour timesteps, #features)
train_data = numpy.reshape(train_data, (X_day_t.shape[0], 24, train_data.shape[1]))
test_data = numpy.reshape(test_data, (X_day_e.shape[0], 24, test_data.shape[1]))

X_sch_t = numpy.reshape(X_sch_t, (X_day_t.shape[0], 24, X_sch_t.shape[1]))
X_sch_e = numpy.reshape(X_sch_e, (X_day_e.shape[0], 24, X_sch_e.shape[1]))
H_t = numpy.reshape(H_t, (H_mean_t.shape[0], 24))
H_e = numpy.reshape(H_e, (H_mean_e.shape[0], 24))

#This block is for optimizing LSTM layers
space = {
        'Layer1': hp.quniform('Layer1', 10, 300, 10),
         'Layer2': hp.quniform('Layer2', 10, 300, 10),
         'D1': hp.uniform('D1', 0, 0.5),
         'D2': hp.uniform('D2', 0, 0.5),
         #'layer2_units': hp.quniform('layer2_units', 10, 50, 1)
         }


def objective(params):
    optimize_model = build_lstm_v1.lstm_model_102(params, train_data.shape[2], 24, 24)
    for ep in range(1):
        optimize_history = optimize_model.fit(train_data, H_t, batch_size=1, nb_epoch=1, validation_split=0.3, shuffle=False)
        #optimize_history = optimize_model.fit(train_data, H_t, batch_size=1, nb_epoch=1, validation_split=0.3, shuffle=False)
        optimize_model.reset_states()

    loss_v = optimize_history.history['loss']
    print loss_v

    loss_out = loss_v[-1]

    return {'loss': loss_out, 'status': STATUS_OK}


trials = Trials()
best = fmin(objective, space, algo=tpe.suggest, trials=trials, max_evals=50)

print best

#Building Stateful Model
lstm_hidden = best
tsteps = 24
out_dim = 24

lstm_model = build_lstm_v1.lstm_model_102(lstm_hidden, train_data.shape[2], out_dim, tsteps)

for ep in range(20):
    lstm_history = lstm_model.fit(train_data, H_t, batch_size=1, nb_epoch=1, validation_split=0.3, shuffle=False)
    #lstm_history = lstm_model.fit(train_data, H_t, batch_size=1, nb_epoch=1, validation_split=0.5, shuffle=False)
    lstm_model.reset_states()


lstm_history = lstm_model.fit(train_data, H_t, batch_size=1, nb_epoch=1, validation_split=0.1, shuffle=False)
lstm_model.reset_states()
loss_v = lstm_history.history['loss']

Y_lstm = lstm_model.predict(test_data, batch_size=1, verbose=0)


#### Error analysis
H_t = numpy.reshape(H_t, (H_t.shape[0]*24, 1))
H_e = numpy.reshape(H_e, (H_e.shape[0]*24, 1))
Y_lstm = numpy.reshape(Y_lstm, (Y_lstm.shape[0]*24, 1))
t_train = numpy.arange(0, len(H_t))
t_test = numpy.arange(len(H_t), len(H_t)+len(Y_lstm))
t_array = numpy.arange(0, len(Y_lstm))

e_deep = (MathFunctions.rms_flat(Y_lstm - H_e))/(MathFunctions.rms_flat(H_e))
e_deep2 = (MathFunctions.rms_flat(Y_lstm - H_e))/(MathFunctions.rms_flat(H_t))

print len(H_t)
print len(Y_lstm)

print e_deep
print e_deep2


### Reshape arrays for daily neural network
X_day_t = numpy.reshape(X_day_t, (X_day_t.shape[0], 1, X_day_t.shape[1]))
X_day_e = numpy.reshape(X_day_e, (X_day_e.shape[0], 1, X_day_e.shape[1]))
H_day_t = numpy.concatenate((H_mean_t, H_max_t, H_min_t), axis=1)
H_day_e = numpy.concatenate((H_mean_e, H_max_e, H_min_e), axis=1)

#best = NNFun_PSB.optimize_lstm_daily(X_day_t, H_mean_t, space)

#print best

#lstm_model_day = NNFun_PSB.fit_lstm_daily(X_day_t, H_mean_t, best)

#Y_lstm_day = lstm_model_day.predict(X_day_e, batch_size=1, verbose=0)

#print Y_lstm_day
#print H_mean_e

#err_day = (MathFunctions.rms_flat(Y_lstm_day - H_mean_e))/(MathFunctions.rms_flat(H_mean_e))
#print err_day

#### Plotting
PlotFunctions.Plot_double(t_array, H_e, t_array, Y_lstm, 'Actual conv power','LSTM conv power', 'k-', 'r-')

PlotFunctions.Plot_triple(t_train, H_t, t_test, Y_lstm, t_test, H_e, 'Training Data', 'LSTM predictions', 'Test Data (actual)', 'k-', 'r-', 'b-')

#MathFunctions.plot_fft(H_t)

loss_vector = trials.losses()

loss_vector = numpy.asarray(loss_vector)
iterations = numpy.arange(0, len(loss_vector))


print loss_vector.shape
print iterations.shape

PlotFunctions.Plot_Iterations(iterations, loss_vector)
PlotFunctions.Plot_fft(H_t, numpy.arange(0, len(H_t)))
