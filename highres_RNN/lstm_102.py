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
H_t = DataFunctions.fix_data(elec_total) #substract to find differences
H_t = DataFunctions.fix_energy_intervals(H_t, 5, std_inv) #convert to std_div time intervals
H_t = DataFunctions.fix_high_points(H_t)


S, unique, pk = DataFunctions.calculate_entropy(H_t)

print "Entropy: "
print S

#Weather Data for training
weather_file = DataFunctions.read_weather_files()
weather_train = DataFunctions.read_weather_csv(weather_file, date_start, date_end)
weather_train = DataFunctions.fix_weather_intervals(weather_train, 5, std_inv)
weather_train = DataFunctions.interpolate_nans(weather_train)

#feature vectors
X_sch_t = DataFunctions.compile_features(H_t, date_start, std_inv)

train_data = numpy.concatenate((weather_train[:, 0:2], X_sch_t), axis=1)
#train_data = X_sch_t
#train_data = DataFunctions.normalize_2D(train_data)



choice= 0

if choice == 1:
    H_t = numpy.load('Ht_file_total.npy')
else:
    X_t, Y_t, X_e, Y_e, train_list, test_list = InterpolateFunctions.train_test_split(train_data, H_t)
    best = InterpolateFunctions.imputate_optimize(X_t, Y_t)
    Y_p = NNFun_PSB.PSB_model_DL(X_t, Y_t, X_e, best)
    H_t = InterpolateFunctions.organize_pred(H_t, Y_p, test_list)
    numpy.save('Ht_file_total.npy', H_t)

H_rms = MathFunctions.rms_flat(H_t)
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

#######
#Getting validation data
#date_start = '5/11/16 12:00 AM'
#date_end = '5/18/16 11:59 PM'

#Read data
data = DataFunctions.read_PSB_csv(date_start, date_end, 5, 20) #read data at 5 min resolutions
conv_critical, crac_critical, crac_normal, conv_normal, HVAC_critical, Elev, HVAC_normal, elec_total = DataFunctions.PSB_elec_split(data) #split by end uses
H_val = DataFunctions.fix_data(elec_total) #substract to find differences
H_val = DataFunctions.fix_energy_intervals(H_val, 5, std_inv) #convert to std_div time intervals
H_val = DataFunctions.fix_high_points(H_val)

#PlotFunctions.Plot_single(H_val)

#Weather Data for training
weather_file = DataFunctions.read_weather_files()
weather_val = DataFunctions.read_weather_csv(weather_file, date_start, date_end)
weather_val = DataFunctions.fix_weather_intervals(weather_val, 5, std_inv)
weather_val = DataFunctions.interpolate_nans(weather_val)


#feature vectors
X_sch_val = DataFunctions.compile_features(H_val, date_start, std_inv)

val_data = numpy.concatenate((weather_val[:, 0:2], X_sch_val), axis=1)
#val_data = X_sch_val
#val_data = DataFunctions.normalize_2D(val_data)

choice = 0

if choice == 1:
    H_val = numpy.load('Hv_file_total.npy')
else:
    X_t, Y_t, X_e, Y_e, train_list, test_list = InterpolateFunctions.train_test_split(val_data, H_val)
    best = InterpolateFunctions.imputate_optimize(X_t, Y_t)
    Y_p = NNFun_PSB.PSB_model_DL(X_t, Y_t, X_e, best)
    H_val = InterpolateFunctions.organize_pred(H_val, Y_p, test_list)
    numpy.save('Hv_file_total.npy', H_val)

#normalizing data
H_val = H_val/H_max
#X_sch_t = DataFunctions.normalize_vector(X_sch_t, X_min, X_max)

#Aggregating data on a daily basis
H_mean_v, H_sum_v, H_min_v, H_max_v = DataFunctions.aggregate_data(H_val, conv_hour_to_day)
w_mean_v, w_sum_v, w_min_v, w_max_v = DataFunctions.aggregate_data(weather_val, conv_hour_to_day)

#gettomg features for a single day
X_day_val = DataFunctions.compile_features(H_sum_v, date_start, 24*60)

print "H_Val"
print X_sch_val.shape
print X_day_val.shape


################

###Test data: 2016
date_start = '5/19/16 12:00 AM'
date_end = '8/7/16 11:59 PM'


#Read data
data = DataFunctions.read_PSB_csv(date_start, date_end, 5, 20) #read data at 5 min resolutions
conv_critical, crac_critical, crac_normal, conv_normal, HVAC_critical, Elev, HVAC_normal, elec_total = DataFunctions.PSB_elec_split(data) #split by end uses
H_e = DataFunctions.fix_data(elec_total) #substract to find differences
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
#test_data = DataFunctions.normalize_2D(test_data) #CUT THIS OFF FOR REGRESSION


if choice == 1:
    H_e = numpy.load('He_file_total.npy')
else:
    X_t, Y_t, X_e, Y_e, train_list, test_list = InterpolateFunctions.train_test_split(test_data, H_e)
    best = InterpolateFunctions.imputate_optimize(X_t, Y_t)
    Y_p = NNFun_PSB.PSB_model_DL(X_t, Y_t, X_e, best)
    H_e = InterpolateFunctions.organize_pred(H_e, Y_p, test_list)
    numpy.save('He_file_total.npy', H_e)

H_e = H_e/H_max
#X_sch_e = DataFunctions.normalize_vector(X_sch_e, X_min, X_max)

#Plotting imputated values
#PlotFunctions.Plot_double(train_list, Y_t, test_list, Y_p, 'Actual Value', 'Interpolated value', 'ro', 'bo')

#Aggregating data on a daily basis
H_mean_e, H_sum_e, H_min_e, H_max_e = DataFunctions.aggregate_data(H_e, conv_hour_to_day)
w_mean_e, w_sum_e, w_min_e, w_max_e = DataFunctions.aggregate_data(weather_test, conv_hour_to_day)

#gettomg features for a single day
X_day_e = DataFunctions.compile_features(H_sum_e, date_start, 24*60)

#####
#Saving variables for MLP neural network
X1 = train_data
X2 = val_data
X3 = test_data

H1 = H_t
H2 = H_val
H3 = H_e
#  `
#print H_mean_t.shape

# Reshaping array into (#of days, 24-hour timesteps, #features)
train_data = numpy.reshape(train_data, (X_day_t.shape[0], 24, train_data.shape[1]))
val_data = numpy.reshape(val_data, (X_day_val.shape[0], 24, val_data.shape[1]))
test_data = numpy.reshape(test_data, (X_day_e.shape[0], 24, test_data.shape[1]))

X_sch_t = numpy.reshape(X_sch_t, (X_day_t.shape[0], 24, X_sch_t.shape[1]))
X_sch_val = numpy.reshape(X_sch_val, (X_day_val.shape[0], 24, X_sch_val.shape[1]))
X_sch_e = numpy.reshape(X_sch_e, (X_day_e.shape[0], 24, X_sch_e.shape[1]))
#H_t = numpy.reshape(H_t, (H_mean_t.shape[0], 24, 1))
#H_e = numpy.reshape(H_e, (H_mean_e.shape[0], 24, 1))

H_t = numpy.reshape(H_t, (H_mean_t.shape[0], 24))
H_val = numpy.reshape(H_val, (H_mean_v.shape[0], 24))
H_e = numpy.reshape(H_e, (H_mean_e.shape[0], 24))

#This block is for optimizing LSTM layers
space = {
        'Layer1': hp.quniform('Layer1', 10, 100, 5),
        'Layer2': hp.quniform('Layer2', 10, 100, 5),
        'Layer3': hp.quniform('Layer3', 5, 20, 1),
         #'D1': hp.uniform('D1', 0, 0.5),
         #'D2': hp.uniform('D2', 0, 0.5),
         #'layer2_units': hp.quniform('layer2_units', 10, 50, 1)
         }


def objective(params):
    #optimize_model = build_lstm_v1.lstm_model_102(params, train_data.shape[2], 24, 24)
    #optimize_model = build_lstm_v1.lstm_model_106(params, train_data.shape[2], 24)
    optimize_model = build_lstm_v1.lstm_model_106(params, train_data.shape[2], 24)

    #for epochs in range(5):
    for ep in range(5):
        #optimize_history = optimize_model.fit(X_seq, Y_seq, batch_size=1, nb_epoch=3, validation_split=(X_seq, Y_seq), shuffle=False)
        optimize_history = optimize_model.fit(train_data, H_t, batch_size=1, nb_epoch=1, validation_data=(val_data, H_val), shuffle=False)
        #optimize_history = optimize_model.fit(train_data, H_t, batch_size=1, nb_epoch=1, validation_split=0.3, shuffle=False)
        optimize_model.reset_states()

    loss_v = optimize_history.history['val_loss']
    print loss_v

    loss_out = loss_v[-1]

    return {'loss': loss_out, 'status': STATUS_OK}


trials = Trials()
best = fmin(objective, space, algo=tpe.suggest, trials=trials, max_evals=20)

print best

#Building Stateful Model
lstm_hidden = best
tsteps = 24
out_dim = 24

#lstm_model = build_lstm_v1.lstm_model_102(lstm_hidden, train_data.shape[2], out_dim, tsteps)
lstm_model = build_lstm_v1.lstm_model_106(lstm_hidden, train_data.shape[2], tsteps)
save_model = lstm_model

##callbacks for Early Stopping
callbacks = [EarlyStopping(monitor='val_loss', patience=3)]

#parameters for simulation
attempt_max = 5
epoch_max = 50
min_epoch = 10

#Criterion for early stopping
tau = 3
e_mat = numpy.zeros((epoch_max, attempt_max))
e_temp = numpy.zeros((tau, ))

tol = 0
count = 0
val_loss_v = []
epsilon = 1 #initialzing error

for attempts in range(attempt_max):
    lstm_model = build_lstm_v1.lstm_model_106(lstm_hidden, train_data.shape[2], tsteps)
    print "New model Initialized"

    for ep in range(epoch_max):
        lstm_history = lstm_model.fit(train_data, H_t, batch_size=1, nb_epoch=1, validation_data=(val_data, H_val), shuffle=False, callbacks=callbacks)
        #lstm_history = lstm_model.fit(train_data, H_t, batch_size=1, nb_epoch=1, validation_split=0.5, shuffle=False)

        loss_v = lstm_history.history['val_loss']
        val_loss_check = loss_v[-1]
        val_loss_v.append(val_loss_check)
        e_mat[ep, attempts] = val_loss_check

        if val_loss_v[count] < epsilon:
            epsilon = val_loss_v[count]
            save_model = lstm_model
            lstm_model.reset_states()
            Y_lstm = lstm_model.predict(test_data, batch_size=1, verbose=0)
            e_1, e_2 = DataFunctions.find_error(H_t, H_e, Y_lstm)
            print e_1
            print e_2

        count = count + 1
        lstm_model.reset_states()


        #This block is for early stopping
        if ep>=min_epoch:
            e_temp = e_mat[ep - tau + 1: ep + 1, attempts]
            e_local = e_mat[ep-tau, attempts]

            print e_temp
            print e_local

            if numpy.all(e_temp > e_local):
                break



        #if val_loss_check < tol:
            #break


#lstm_history = lstm_model.fit(train_data, H_t, batch_size=1, nb_epoch=1, validation_split=(train_data, H_t), shuffle=False)
print val_loss_v
#save_model.reset_states()
print "count is"
print count

#Y_lstm = lstm_model.predict(test_data, batch_size=1, verbose=0)
Y_lstm2 = save_model.predict(test_data, batch_size=1, verbose=0)

#### Error analysis
H_t = numpy.reshape(H_t, (H_t.shape[0]*24, 1))
H_e = numpy.reshape(H_e, (H_e.shape[0]*24, 1))
Y_lstm = numpy.reshape(Y_lstm, (Y_lstm.shape[0]*24, 1))
Y_lstm2 = numpy.reshape(Y_lstm2, (Y_lstm2.shape[0]*24, 1))
t_train = numpy.arange(0, len(H_t))
t_test = numpy.arange(len(H_t), len(H_t)+len(Y_lstm))
t_array = numpy.arange(0, len(Y_lstm))

e_deep = (MathFunctions.rms_flat(Y_lstm2 - H_e))/(MathFunctions.rms_flat(H_e))
e_deep2 = (MathFunctions.rms_flat(Y_lstm2 - H_e))/(MathFunctions.rms_flat(H_t))
e_deep3 = (MathFunctions.rms_flat(Y_lstm - H_e))/(MathFunctions.rms_flat(H_e))
e_deep4 = (MathFunctions.rms_flat(Y_lstm - H_e))/(MathFunctions.rms_flat(H_t))



print e_deep
print e_deep2

print e_deep3
print e_deep4




### Reshape arrays for daily neural network
X_day_t = numpy.reshape(X_day_t, (X_day_t.shape[0], 1, X_day_t.shape[1]))
X_day_e = numpy.reshape(X_day_e, (X_day_e.shape[0], 1, X_day_e.shape[1]))
H_day_t = numpy.concatenate((H_mean_t, H_max_t, H_min_t), axis=1)
H_day_e = numpy.concatenate((H_mean_e, H_max_e, H_min_e), axis=1)



#### Implement MLP Neural Network
best_NN = NNFunctions.NN_optimizeNN_v21(X1, H1, X2, H2)
NN_model = NNFunctions.CreateRealSchedule_v21(best_NN, X1.shape[1])
NN_savemodel = NN_model

epsilon = 1
val_loss_v = []

for attempts in range(0, 5):
    NN_model = NNFunctions.CreateRealSchedule_v21(best_NN, X1.shape[1])
    NN_history = NN_model.fit(X1, H1, validation_data=(X1, H1), nb_epoch=50, batch_size=1, verbose=0, callbacks=callbacks)

    loss_v = NN_history.history['val_loss']
    val_loss_check = loss_v[-1]
    #print val_loss_check
    val_loss_v.append(val_loss_check)

    if val_loss_v[attempts] < epsilon:
        epsilon = val_loss_v[attempts]
        NN_savemodel = NN_model



Y_NN = NN_savemodel.predict(X3)
#Y_NN = numpy.reshape(Y_NN, (Y_NN.shape[0]*24, 1))
e_NN = (MathFunctions.rms_flat(Y_NN - H3))/(MathFunctions.rms_flat(H1))

print e_NN



#### Plotting
PlotFunctions.Plot_double(t_array, H_e, t_array, Y_lstm2, 'Actual conv power','LSTM conv power', 'k-', 'r-', "fig_total1a.eps")

PlotFunctions.Plot_triple(t_train, H_t, t_test, Y_lstm2, t_test, H_e, 'Training Data', 'LSTM predictions', 'Test Data (actual)', 'k-', 'r-', 'b-', "fig_total1b.eps")

PlotFunctions.Plot_triple(t_train, H_t, t_test, Y_lstm, t_test, H_e, 'Training Data', 'LSTM predictions', 'Test Data (actual)', 'k-', 'r-', 'b-', "fig_total1c.eps")

PlotFunctions.Plot_quadruple(t_train, H_t, t_test, Y_lstm2, t_test, Y_NN, t_test, H_e, 'Training Data', 'LSTM predictions', 'MLP Predictions', 'Test Data (actual)', 'k-', 'r-', 'y-', 'b-', "fig_Elev1d.eps")


print trials.losses()

print "The RMS value is :"
print H_rms


