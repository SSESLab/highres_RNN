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
import DataFunctions_2
import NNFunctions
import PlotFunctions
import InterpolateFunctions
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

####Read energy and weather data

date_start = '01/01/16 12:00 AM'
date_end = '12/31/16 11:59 PM'
ref_date = date_start
std_inv = 60 #in minutes
std_inv2 = 10 #in minutes
delta = 0.05

###Importing holiday
holiday_list, class_list = DataFunctions.import_holiday(2016)
holiday_flag, spring_flag, summer_flag, fall_flag, recess_flag = DataFunctions.verify_date(DataFunctions.give_time(date_start), 2016)


folder_path = r'/home/sseslab/PycharmProjects/ElectricLoad_v1/MEB_energy/train_data'
data = DataFunctions_2.read_multiple_files(folder_path, 1, 2)
H_t = DataFunctions_2.social_beh_101(data)


#performing linear interpolation on H_t2 firs
small_list, large_list = InterpolateFunctions.interpolate_main_v2(H_t, 5) #Allowing for 30 consecutive timesteps
H_t = InterpolateFunctions.interp_linear(H_t, small_list)
H_t = H_t[:, None]
H2 = H_t.copy()
H_t = DataFunctions_2.fix_high_points(H_t)



#Weather Data for training
folder_path = r'/home/sseslab/Documents/dentistry_bldg/weather_WBB_train'
weather_train = DataFunctions_2.prepare_weather_WBB(folder_path, date_start, date_end, std_inv)
weather_train2 = DataFunctions.prepare_weather_WBB(date_start, date_end, std_inv2)

#schedule features
X_sch_t = DataFunctions.compile_features_U(H_t, date_start, std_inv, 2016)
X_sch_t2 = DataFunctions.get_feature_low_res(X_sch_t, std_inv, std_inv2)

print "X troubleshoot"
print X_sch_t
print X_sch_t.shape


train_data = numpy.concatenate((weather_train[:, 0:4], X_sch_t), axis=1)
train_data2 = numpy.concatenate((weather_train2[:, 0:4], X_sch_t2), axis=1)

H_rms = MathFunctions.rms_flat(H_t)
#normalizing data
H_min, H_max = DataFunctions.get_normalize_params(H_t)
H_t = H_t/H_max

#PlotFunctions.Plot_single(H_t)
#PlotFunctions.Plot_single(H2)
#Computing Entropy
S, unique, pk = DataFunctions.calculate_entropy(H_t)

print S, unique, pk

cons_points = 5
s0 = 245
start_day = 250
end_day  = 251
s1 = 260

choice = 1

if choice == 1:
    H_t = numpy.load('Ht_file_MEB1.npy')
elif choice == 2:
    X_t, Y_t, X_e, Y_e, train_list, test_list = InterpolateFunctions.train_test_split(train_data, H_t)
    best = InterpolateFunctions.imputate_optimize(X_t, Y_t)
    Y_p = NNFun_PSB.PSB_model_DL(X_t, Y_t, X_e, best)
    H_t = InterpolateFunctions.organize_pred(H_t, Y_p, test_list)
    numpy.save('Ht_file_MEB1.npy', H_t)
else:
    H1 = H_t.copy()
    small_list, large_list = InterpolateFunctions.interpolate_main(train_data, H_t, start_day, end_day, cons_points)
    H_t = InterpolateFunctions.interp_linear(H_t, small_list) #H_t is beign changed as numpy arrays are mutable
    H1 = InterpolateFunctions.interp_linear(H1, small_list)

    H_t = H_t[:, None] #changing numpy array shape to fit the function
    train_interp, dummy1, dummy2 = DataFunctions.normalize_103(train_data, train_data, train_data)
    Y_t, Y_NN = InterpolateFunctions.interp_LSTM(train_interp, H_t, large_list)

    PlotFunctions.Plot_interpolate(H1[s0*24:start_day*24], Y_t[start_day*24:end_day*24], Y_NN[start_day*24:end_day*24], H1[start_day*24:end_day*24], H1[end_day*24:s1*24])
    e_interp = InterpolateFunctions.interpolate_calculate_rms(H1[start_day*24:end_day*24], Y_t[start_day*24:end_day*24])
    e_NN = InterpolateFunctions.interpolate_calculate_rms(H1[start_day*24:end_day*24], Y_NN[start_day*24:end_day*24])

    print e_interp
    print e_NN
    H_t = Y_t.copy()
    numpy.save('H1_file_MEB1.npy', H_t)


#Aggregating data on a daily basis
conv_hour_to_day = 24
H_mean_t, H_sum_t, H_min_t, H_max_t = DataFunctions.aggregate_data(H_t, conv_hour_to_day)
w_mean_t, w_sum_t, w_min_t, w_max_t = DataFunctions.aggregate_data(weather_train, conv_hour_to_day)

#gettomg features for a single day
#PlotFunctions.Plot_single(H_mean_t)
X_day_t = DataFunctions.compile_features(H_sum_t, date_start, 24*60)


####Seggregating validation data
date_start = '12/24/16 12:00 AM'
date_end = '12/31/16 11:59 PM'

h_tstep1 = int(DataFunctions.find_hourly_timesteps(ref_date, date_start))
h_tstep2 = int(DataFunctions.find_hourly_timesteps(ref_date, date_end))


#spit train, test
train_data, H_t, val_data, H_val = DataFunctions_2.split_val_data(train_data, H_t, h_tstep1, h_tstep2)

#The validation has no more holes :)
#Aggregating data on a daily basis
H_mean_v, H_sum_v, H_min_v, H_max_v = DataFunctions.aggregate_data(H_val, conv_hour_to_day)
w_mean_v, w_sum_v, w_min_v, w_max_v = DataFunctions.aggregate_data(val_data, conv_hour_to_day)

#gettomg features for a single day
X_day_val = DataFunctions.compile_features(H_sum_v, date_start, 24*60)


#####Getting test data
date_start = '01/01/17 12:00 AM'
date_end = '07/31/17 11:59 PM'


folder_path = r'/home/sseslab/PycharmProjects/ElectricLoad_v1/MEB_energy/test_data'
data = DataFunctions_2.read_multiple_files(folder_path, 1, 2)
H_e = DataFunctions_2.social_beh_101(data)

H2 = H_e.copy()

#performing linear interpolation on H_e firs
small_list, large_list = InterpolateFunctions.interpolate_main_v2(H_e, 5) #Allowing for 30 consecutive timesteps
H_e = InterpolateFunctions.interp_linear(H_e, small_list)
H_e = H_e[:, None]
H_e = DataFunctions_2.fix_high_points(H_e)

PlotFunctions.Plot_single(H_e)
PlotFunctions.Plot_single(H2)

###indexing out test data
idx_e = DataFunctions.find_hourly_timesteps(date_start, date_end)
H_e = H_e[:idx_e+1, :]


#Weather Data for training
folder_path = r'/home/sseslab/Documents/dentistry_bldg/weather_WBB_test'
weather_test = DataFunctions_2.prepare_weather_WBB(folder_path, date_start, date_end, std_inv)
#weather_test2 = DataFunctions.prepare_weather_WBB(date_start, date_end, std_inv2)

print H_e.shape
print weather_test.shape

#schedule features
X_sch_e = DataFunctions.compile_features_U(H_e, date_start, std_inv, 2017)
#padding the test matrix with zeros
X_sch_e = numpy.concatenate((X_sch_e, numpy.zeros((len(X_sch_e), X_sch_t.shape[1] - X_sch_e.shape[1]))), axis=1)


X_sch_e2 = DataFunctions.get_feature_low_res(X_sch_e, std_inv, std_inv2)

test_data = numpy.concatenate((weather_test[:, 0:4], X_sch_e), axis=1)
#test_data2 = numpy.concatenate((weather_test2[:, 0:2], X_sch_e2), axis=1)

#Normalizing Data:
print train_data.shape
print val_data.shape
print test_data.shape

train_data, val_data, test_data = DataFunctions.normalize_103(train_data, val_data, test_data)

#Normalize data
H_e = H_e/H_max
#X_sch_e = DataFunctions.normalize_vector(X_sch_e, X_min, X_max)


#PlotFunctions.Plot_single(H_e)
#PlotFunctions.Plot_single(H2)

choice = 2

if choice == 1:
    H_e = numpy.load('He_file_total.npy')
elif choice==2:
    small_list, large_list = InterpolateFunctions.interpolate_main_v2(H_e, 5)  # Allowing for 30 consecutive timesteps
    H_e = InterpolateFunctions.interp_linear(H_e, small_list)
    print numpy.isnan(H_e)
    H_e[numpy.isnan(H_e)] = numpy.nanmean(H_e)
else:
    X_t, Y_t, X_e, Y_e, train_list, test_list = InterpolateFunctions.train_test_split(test_data, H_e)
    best = InterpolateFunctions.imputate_optimize(X_t, Y_t)
    Y_p = NNFun_PSB.PSB_model_DL(X_t, Y_t, X_e, best)
    H_e = InterpolateFunctions.organize_pred(H_e, Y_p, test_list)

    #numpy.save('He_file_total.npy', H_e)




#indexing out forecasting period
#Plotting imputated values
#PlotFunctions.Plot_double(train_list, Y_t, test_list, Y_p, 'Actual Value', 'Interpolated value', 'ro', 'bo')
H_e = H_e[:, None]

#Aggregating data on a daily basis
H_mean_e, H_sum_e, H_min_e, H_max_e = DataFunctions.aggregate_data(H_e, conv_hour_to_day)
w_mean_e, w_sum_e, w_min_e, w_max_e = DataFunctions.aggregate_data(weather_test, conv_hour_to_day)

#gettomg features for a single day
X_day_e = DataFunctions.compile_features(H_sum_e, date_start, 24*60)

#Saving variables for MLP neural network
X1 = train_data.copy()
X2 = val_data.copy()
X3 = test_data.copy()

H1 = H_t.copy()
H2 = H_val.copy()
H3 = H_e.copy()



#Saving files
#numpy.save('Ht_SB1.npy', H_t)
#numpy.save('Hv_SB1.npy', H_val)
#numpy.save('He_SB1.npy', H_e)


#Reshaping array into (#of days, 24-hour timesteps, #features)
train_data = numpy.reshape(train_data, (int(len(train_data)/24), 24, train_data.shape[1]))
val_data = numpy.reshape(val_data, (int(len(val_data)/24), 24, val_data.shape[1]))
test_data = numpy.reshape(test_data, (int(len(test_data)/24), 24, test_data.shape[1]))

H_t = numpy.reshape(H_t, (int(len(H_t)/24), 24))
H_val = numpy.reshape(H_val, (len(H_val)/24, 24))
H_e = numpy.reshape(H_e, (len(H_e)/24, 24))

#This block is for optimizing LSTM layers
#This block is for optimizing LSTM layers
space = {
        'Layer1': hp.quniform('Layer1', 10, 100, 5),
        'Layer2': hp.quniform('Layer2', 10, 100, 5),
        'Layer3': hp.quniform('Layer3', 5, 20, 1),
        'activ_l3': hp.choice('activ_l3', ['relu', 'sigmoid']),
        'activ_l4': hp.choice('activ_l4', ['sigmoid'])
         }


def objective(params):
    optimize_model = build_lstm_v1.lstm_model_110(params, train_data.shape[2], 24)

    #for epochs in range(5):
    for ep in range(20):
        optimize_history = optimize_model.fit(train_data, H_t, batch_size=1, nb_epoch=1, validation_data=(val_data, H_val), shuffle=False)
        optimize_model.reset_states()

    loss_v = optimize_history.history['val_loss']
    print loss_v

    loss_out = loss_v[-1]

    return {'loss': loss_out, 'status': STATUS_OK}


trials = Trials()
best = fmin(objective, space, algo=tpe.suggest, trials=trials, max_evals=20)

##################################################################################################

#Building Stateful Model
lstm_hidden = hyperopt.space_eval(space, best)
print lstm_hidden
tsteps = 24
out_dim = 24

#lstm_model = build_lstm_v1.lstm_model_102(lstm_hidden, train_data.shape[2], out_dim, tsteps)
lstm_model = build_lstm_v1.lstm_model_110(lstm_hidden, train_data.shape[2], tsteps)
save_model = lstm_model

##callbacks for Early Stopping
callbacks = [EarlyStopping(monitor='val_loss', patience=3)]

#parameters for simulation
attempt_max = 3
epoch_max = 100
min_epoch = 10

#Criterion for early stopping
tau = 10
e_mat = numpy.zeros((epoch_max, attempt_max))
e_temp = numpy.zeros((tau, ))

tol = 0
count = 0
val_loss_v = []
epsilon = 1 #initialzing error
loss_old = 1
loss_val = 1

for attempts in range(attempt_max):
    lstm_model = build_lstm_v1.lstm_model_110(lstm_hidden, train_data.shape[2], tsteps)
    print "New model Initialized"

    val_err_temp = []

    for ep in range(epoch_max):
        lstm_history = lstm_model.fit(train_data, H_t, batch_size=1, nb_epoch=1, validation_split=0, shuffle=False)

        loss_old = loss_val
        loss_val = lstm_history.history['loss']

        # testing alternative block
        #lstm_model.reset_states()
        y_val = lstm_model.predict(val_data, batch_size=1, verbose=0)
        e1, e2 = DataFunctions.find_error(H_t, H_val, y_val)
        print e1, e2
        val_loss_check = e2
        val_err_temp.append(e2)
        val_loss_v.append(val_loss_check)
        e_mat[ep, attempts] = val_loss_check

        if val_loss_v[count] < epsilon and loss_val < loss_old:
            val_final = numpy.asarray(val_err_temp)
            epsilon = val_loss_v[count]
            save_model = lstm_model
            test_model = lstm_model
            Y_lstm = test_model.predict(test_data, batch_size=1, verbose=0)
            e_1, e_2 = DataFunctions.find_error(H_t, H_e, Y_lstm)
            test_model.reset_states()
            print e_1
            print e_2

        count = count + 1
        lstm_model.reset_states()


        #This block is for early stopping
        if ep>=min_epoch:
            e_temp = e_mat[ep - tau + 1: ep + 1, attempts]
            e_local = e_mat[ep-tau, attempts]

            #print e_temp
            #print e_local

            if numpy.all(e_temp > e_local):
                break



        #if val_loss_check < tol:
            #break


print val_loss_v

#Y_lstm = lstm_model.predict(test_data, batch_size=1, verbose=0)
#Y1 = save_model.predict(train_data, batch_size=1, verbose=0) #get the states up to speed
#Y2 = save_model.predict(val_data, batch_size=1, verbose=0) #get the states up to speed
Y_lstm2 = save_model.predict(test_data, batch_size=1, verbose=0)
#numpy.save('Y_file_CONV1.npy', Y_lstm2)

#### Error analysis
H_t = numpy.reshape(H_t, (H_t.shape[0]*24, 1))
H_val = numpy.reshape(H_val, (H_val.shape[0]*24, 1))
H_e = numpy.reshape(H_e, (H_e.shape[0]*24, 1))
Y_lstm = numpy.reshape(Y_lstm, (Y_lstm.shape[0]*24, 1))
Y_lstm2 = numpy.reshape(Y_lstm2, (Y_lstm2.shape[0]*24, 1))
t_train = numpy.arange(0, len(H_t))
t_test = numpy.arange(len(H_t), len(H_t)+len(Y_lstm2))
t_array = numpy.arange(0, len(Y_lstm2))

e_deep = (MathFunctions.rms_flat(Y_lstm2 - H_e))/(MathFunctions.rms_flat(H_e))
e_deep2 = (MathFunctions.rms_flat(Y_lstm2 - H_e))/(MathFunctions.rms_flat(H_t))

print e_deep
print e_deep2


#### Implement MLP Neural Network
best_NN = NNFunctions.NN_optimizeNN_v21(X1, H1, X2, H2)
NN_model = NNFunctions.CreateRealSchedule_v21(best_NN, X1.shape[1])
NN_savemodel = NN_model

epsilon = 1
val_loss_v = []

for attempts in range(0, 5):
    NN_model = NNFunctions.CreateRealSchedule_v21(best_NN, X1.shape[1])
    NN_history = NN_model.fit(X1, H1, validation_data=(X2, H2), nb_epoch=50, batch_size=1, verbose=0, callbacks=callbacks)

    loss_v = NN_history.history['val_loss']
    val_loss_check = loss_v[-1]
    #print val_loss_check
    val_loss_v.append(val_loss_check)

    if val_loss_v[attempts] < epsilon:
        epsilon = val_loss_v[attempts]
        NN_savemodel = NN_model



Y_NN = NN_savemodel.predict(X3)
#Y_NN = numpy.reshape(Y_NN, (Y_NN.shape[0]*24, 1))
e_NN = (MathFunctions.rms_flat(Y_NN - H3))/(MathFunctions.rms_flat(H3))
e_NN2 = (MathFunctions.rms_flat(Y_NN - H3))/(MathFunctions.rms_flat(H1))
#Saving files

numpy.save('Ht_MEB1.npy', H_t)
numpy.save('Hv_MEB1.npy', H_val)
numpy.save('He_MEB1.npy', H_e)
numpy.save('MEB1_BU.npy', Y_lstm2)
numpy.save('MEB1_2BU.npy', Y_lstm)
numpy.save('MEB1_NNU.npy', Y_NN)

print "Max H, ", H_max
print "RMS H: ", H_rms
print e_NN
print e_NN2
