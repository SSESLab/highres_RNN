#THIS FILE IS TO COMPUTE THE Multi-timescale method
#importing keras modules
import keras


from keras.callbacks import EarlyStopping
from keras import backend as K


#importing hyperopt modules
from hyperopt import Trials, STATUS_OK, tpe
from hyperas import optim
from hyperas.distributions import choice, uniform, conditional
from hyperopt import fmin, tpe, hp
from tempfile import TemporaryFile
import hyperopt.pyll.stochastic

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

from sklearn.metrics import r2_score
from sklearn.datasets import fetch_20newsgroups_vectorized
from sklearn.feature_selection import chi2

#scipy library
from scipy.stats import pearsonr

seed = 7
numpy.random.seed(seed)

#EnergyData
#Processing hourly data
date_start = '01/01/15 12:00 AM'
date_end = '12/31/15 11:59 PM'
std_inv1 = 60 #in minutes

#Extracting Energy Data
folder_path = r'/home/sseslab/PycharmProjects/ElectricLoad_v1/dataport_data/dataport_hour_train3'
H_t, H_max1 = DataFunctions.read_multiple_dataport(folder_path, 24*365)
H_t = H_t/H_max1
H_rms1 = MathFunctions.rms_flat(H_t)

#Computing Entropy
S, unique, pk = DataFunctions.calculate_entropy(H_t)
print "The entropy value is: ", S

#EnergyData Processing 1-min data data
std_inv2 = 1 #in minutes

folder_path = r'/home/sseslab/PycharmProjects/ElectricLoad_v1/dataport_data/dataport_1min_train3'
H_t2, H_max2 = DataFunctions.read_multiple_dataport(folder_path, 24*365*60)
H_t2 = H_t2/H_max2
H_rms2 = MathFunctions.rms_flat(H_t2)

file_ = r'~/PycharmProjects/ElectricLoad_v1/dataport_data/austin_weather_2015.csv'
weather_train = DataFunctions.preprocess_weatherData_101(file_, date_start, date_end)
X_sch_t = DataFunctions.compile_features(H_t, date_start, std_inv1)

#Concatenating and converting data to 1 min resolution
train_data = numpy.concatenate((weather_train[:, 0:2], X_sch_t), axis=1)
train_data2 = DataFunctions.get_weather_1min(train_data, std_inv1, std_inv2)

#Normalizing Data
train_data = DataFunctions.normalize_2D(train_data)
train_data2 = DataFunctions.normalize_2D(train_data2)

#Geting Validation sets
#val_data, H_val  = DataFunctions.find_val_set2(0, train_data, H_t, 1)
val_data, H_val, val_data2, H_val2  = DataFunctions.find_val_set2(0, train_data, H_t, train_data2, H_t2, 60)

conv_hour_to_day =  24
H_mean_t, H_sum_t, H_min_t, H_max_t = DataFunctions.aggregate_data(H_t, conv_hour_to_day)
X_day_t = DataFunctions.compile_features(H_sum_t, date_start, 24*60)

#Getting Daily features
H_mean_v, H_sum_v, H_min_v, H_max_v = DataFunctions.aggregate_data(H_val, conv_hour_to_day)
X_day_val = DataFunctions.compile_features(H_sum_v, date_start, 24*60)

################################33
###Test Data
date_start = '01/01/16 12:00 AM'
date_end = '12/31/16 11:59 PM'

folder_path = r'/home/sseslab/PycharmProjects/ElectricLoad_v1/dataport_data/dataport_hour_test3'
H_e, dummy = DataFunctions.read_multiple_dataport(folder_path, 24*366)
H_e = H_e/H_max1

folder_path = r'/home/sseslab/PycharmProjects/ElectricLoad_v1/dataport_data/dataport_1min_test3'
H_e2, dummy2 = DataFunctions.read_multiple_dataport(folder_path, 60*24*366)
H_e2 = H_e2/H_max2

#############Features
file_ = r'~/PycharmProjects/ElectricLoad_v1/dataport_data/austin_weather_2016.csv'
weather_test = DataFunctions.preprocess_weatherData_101(file_, date_start, date_end)
X_sch_e = DataFunctions.compile_features(H_e, date_start, std_inv1)


#Concatenating and convertring weather data to 1 resolution
test_data = numpy.concatenate((weather_test[:, 0:2], X_sch_e), axis=1)
test_data2 = DataFunctions.get_weather_1min(test_data, std_inv1, std_inv2)
#test_data2 = test_data2[:, -1]
#test_data2 = test_data2[:, None]

#Normalizing Data
test_data = DataFunctions.normalize_2D(test_data)
test_data2 = DataFunctions.normalize_2D(test_data2)

#Getting Daily features
H_mean_e, H_sum_e, H_min_e, H_max_e = DataFunctions.aggregate_data(H_e, conv_hour_to_day)
X_day_e = DataFunctions.compile_features(H_sum_e, date_start, 24*60)

#Saving variables for MLP neural network
X1 = train_data.copy()
X2 = val_data.copy()
X3 = test_data.copy()
H1 = H_t.copy()
H2 = H_val.copy()
H3 = H_e.copy()

####Reshaping into an array
#Reshaping array into (#of days, 24-hour timesteps, #features)
train_data = numpy.reshape(train_data, (X_day_t.shape[0], 24, train_data.shape[1]))
val_data = numpy.reshape(val_data, (X_day_val.shape[0], 24, val_data.shape[1]))
test_data = numpy.reshape(test_data, (X_day_e.shape[0], 24, test_data.shape[1]))
H_t = numpy.reshape(H_t, (H_mean_t.shape[0], 24))
H_val = numpy.reshape(H_val, (H_mean_v.shape[0], 24))
H_e = numpy.reshape(H_e, (H_mean_e.shape[0], 24))


#Reshaping array into (#of days, 60-min timesteps, #features)
train_data2 = numpy.reshape(train_data2, (conv_hour_to_day*X_day_t.shape[0], 60, train_data2.shape[1]))
val_data2 = numpy.reshape(val_data2, (conv_hour_to_day*X_day_val.shape[0], 60, val_data2.shape[1]))
test_data2 = numpy.reshape(test_data2, (conv_hour_to_day*X_day_e.shape[0], 60, test_data2.shape[1]))
H_t2 = numpy.reshape(H_t2, (conv_hour_to_day*H_mean_t.shape[0], 60))
H_val2 = numpy.reshape(H_val2, (conv_hour_to_day*H_mean_v.shape[0], 60))
H_e2 = numpy.reshape(H_e2, (conv_hour_to_day*H_mean_e.shape[0], 60))


####Optimizing hyper-parameters
#Declaring parameter space
#####THis step is to optimize hyper-parameters
#This block is for optimizing LSTM layers
space = {
        'Layer1': hp.quniform('Layer1', 10, 100, 5),
        'Layer2': hp.quniform('Layer2s', 10, 100, 5),
        'Layer3': hp.quniform('Layer3', 5, 100, 1),
        'activ_l3': hp.choice('activ_l3', ['relu', 'tanh']),
        'activ_l4': hp.choice('activ_l4', ['hard_sigmoid', 'linear'])

         }


space2 = {
        'Layer1': hp.quniform('Layer1', 10, 100, 5),
        'Layer2': hp.quniform('Layer2', 10, 100, 5),
        'Layer3': hp.quniform('Layer3', 5, 100, 1),
        'activ_l3': hp.choice('activ_l3', [ 'tanh', 'relu']),
        'activ_l4': hp.choice('activ_l4', ['hard_sigmoid', 'linear'])

         }


#################
#Optimizing for space 1
def objective(params):
    optimize_model = build_lstm_v1.lstm_model_110(params, train_data.shape[2], 24)
    loss_out = NNFunctions.model_optimizer_101(optimize_model, train_data, H_t, val_data, H_val, 10)
    return {'loss': loss_out, 'status': STATUS_OK}


trials = Trials()
best = fmin(objective, space, algo=tpe.suggest, trials=trials, max_evals=5)

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
epoch_max = 30
min_epoch = 10

#Criterion for early stopping
tau = 5
e_mat = numpy.zeros((epoch_max, attempt_max))
e_temp = numpy.zeros((tau, ))

tol = 0
count = 0
val_loss_v = []
epsilon = 1 #initialzing error
loss_old = 1
loss_val = 1


###
#Fitting model for hourly resolution

for attempts in range(attempt_max):
    lstm_model = build_lstm_v1.lstm_model_110(lstm_hidden, train_data.shape[2], tsteps)
    print "New model Initialized"
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
        val_loss_v.append(val_loss_check)
        e_mat[ep, attempts] = val_loss_check

        if val_loss_v[count] < epsilon and loss_val < loss_old:
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

            if numpy.all(e_temp > e_local):
                break



        #if val_loss_check < tol:
            #break




###########Extracting Data for multi-timescale Analysis
tsteps = 60
get_1st_layer_output = K.function([save_model.layers[0].input], [save_model.layers[3].output])
lstm_h2 = int(lstm_hidden['Layer2'])
h_input = numpy.zeros((len(train_data2), tsteps, lstm_h2))
h_val = numpy.zeros((len(val_data2), tsteps, lstm_h2))
h_test = numpy.zeros((len(test_data2), tsteps, lstm_h2))

#Initialising array for short-term prediction
Y_lstm2 = numpy.zeros((H_mean_e.shape[0], 24))

###Trouble shoot###
print "Len(train1)", len(train_data)
print "Len(train2)", len(train_data2)

for i in range(0, len(train_data)):
    X_temp = train_data[i, :, :]
    X_temp = X_temp[None, :, :]
    h_temp = numpy.asarray(get_1st_layer_output([X_temp]))
    h_dim = h_temp.shape[3]
    h_temp2 = numpy.reshape(h_temp, (24, h_dim))

    for j in range(0, tsteps):
        h_input[i*24:(i+1)*24, j, :] = h_temp2


###Getting validation data
for i in range(0, len(val_data)):
    X_temp = val_data[i, :, :]
    X_temp = X_temp[None, :, :]
    h_dim = h_temp.shape[3]
    h_temp2 = numpy.reshape(h_temp, (24, h_dim))

    for j in range(0, tsteps):
        h_val[i*24:(i+1)*24, j, :] = h_temp2


for i in range(0, len(test_data)):
    X_temp = test_data[i, :, :]
    X_temp = X_temp[None, :, :]
    Y_lstm2[i, :] = save_model.predict(X_temp, batch_size=1, verbose=0)
    Y_temp = H_e[i, :]
    Y_temp = Y_temp[None, :]
    #for ep in range(0, 5):
    lstm_history = save_model.fit(X_temp, Y_temp, batch_size=1, nb_epoch=1, validation_split=0, shuffle=False, verbose=0)

    h_temp = numpy.asarray(get_1st_layer_output([X_temp]))
    h_dim = h_temp.shape[3]
    h_temp2 = numpy.reshape(h_temp, (24, h_dim))

    for j in range(0, tsteps):
        h_test[i*24:(i+1)*24, j, :] = h_temp2


#####Post-possessing block
#Short-term


#### Error analysis
H_t = numpy.reshape(H_t, (H_t.shape[0]*24, 1))
H_e = numpy.reshape(H_e, (H_e.shape[0]*24, 1))
Y_lstm = numpy.reshape(Y_lstm, (Y_lstm.shape[0]*24, 1))
Y_lstm2 = numpy.reshape(Y_lstm2, (Y_lstm2.shape[0]*24, 1))
t_train = numpy.arange(0, len(H_t))
t_test = numpy.arange(len(H_t), len(H_t)+len(Y_lstm2))
t_array = numpy.arange(0, len(Y_lstm2))

e_deep = (MathFunctions.rms_flat(Y_lstm2 - H_e))/(MathFunctions.rms_flat(H_e))
e_deep2 = (MathFunctions.rms_flat(Y_lstm2 - H_e))/(MathFunctions.rms_flat(H_t))

print Y_lstm2.shape
print H_e.shape
print t_array.shape
print e_deep
print e_deep2
print t_array.shape



####Plotting for min-wise
PlotFunctions.Plot_double(t_array, H_e, t_array, Y_lstm2, 'Actual conv power','LSTM conv power', 'k-', 'r-', "fig_5houseI.eps")


#####################
###Build 1-min model:


def objective(params):
    optimize_model = build_lstm_v1.lstm_multi_101b(params, train_data2.shape[2], lstm_h2, 60) #Check code here, relu entering
    loss_out = NNFunctions.model_optimizer_101(optimize_model, [train_data2, h_input], H_t2, [val_data2, h_val], H_val2, 5)
    return {'loss': loss_out, 'status': STATUS_OK}


trials = Trials()
best2 = fmin(objective, space2, algo=tpe.suggest, trials=trials, max_evals=3)


#Building Stateful Model
lstm_hidden = hyperopt.space_eval(space2, best2)
print lstm_hidden
tsteps = 60
out_dim = 60

#Building model
lstm_model = build_lstm_v1.lstm_multi_101b(lstm_hidden, train_data2.shape[2], lstm_h2, tsteps)
save_model = lstm_model

##callbacks for Early Stopping
callbacks = [EarlyStopping(monitor='val_loss', patience=5)]

#parameters for simulation
attempt_max = 1
epoch_max = 20
min_epoch = 10

#Criterion for early stopping
tau = 5
e_mat = numpy.zeros((epoch_max, attempt_max))
e_temp = numpy.zeros((tau, ))

tol = 0
count = 0
val_loss_v = []
epsilon = 1 #initialzing error
loss_old = 1
loss_val = 1


for attempts in range(attempt_max):
    lstm_model = build_lstm_v1.lstm_multi_101b(lstm_hidden, train_data2.shape[2], lstm_h2, tsteps)
    print "New model Initialized"

    for ep in range(epoch_max):
        lstm_history = lstm_model.fit([train_data2, h_input], H_t2, batch_size=1, nb_epoch=1, validation_split=0, shuffle=False)
        loss_old = loss_val
        loss_val = lstm_history.history['loss']

        # testing alternative block
        y_val2 = lstm_model.predict([val_data2, h_val], batch_size=1, verbose=0)
        e1, e2 = DataFunctions.find_error2(H_t2, H_val2, y_val2, 60)
        print e1, e2
        val_loss_check = e2
        val_loss_v.append(val_loss_check)
        e_mat[ep, attempts] = val_loss_check

        if val_loss_v[count] < epsilon and loss_val < loss_old:
            epsilon = val_loss_v[count]
            save_model = lstm_model
            #test_model = lstm_model

        count = count + 1
        lstm_model.reset_states()


        #This block is for early stopping
        if ep>=min_epoch:
            e_temp = e_mat[ep - tau + 1: ep + 1, attempts]
            e_local = e_mat[ep-tau, attempts]

            if numpy.all(e_temp > e_local):
                break




#The prediction step will change for short-term prediction
#Defining arrays for short-term prediction
Y_lstm2 = numpy.zeros((conv_hour_to_day*H_mean_e.shape[0], 60))

for i in range(0, len(test_data2)):
    #for ep in range(0, 5):
     X_temp = test_data2[i, :, :]
     X_temp = X_temp[None, :, :]
     h_temp = h_test[i, :, :]
     h_temp = h_temp[None, :, :]
     Y_lstm2[i, :] = save_model.predict([X_temp, h_temp], batch_size=1, verbose=0)
     Y_temp = H_e2[i, :]
     Y_temp = Y_temp[None, :]
     lstm_history = save_model.fit([X_temp,  h_temp], Y_temp, batch_size=1, nb_epoch=1, validation_split=0, shuffle=False)


#### Error analysis
H_t2 = numpy.reshape(H_t2, (H_t.shape[0]*60, 1))
H_e2 = numpy.reshape(H_e2, (H_e.shape[0]*60, 1))
#Y_lstm = numpy.reshape(Y_lstm, (Y_lstm.shape[0]*60, 1))
Y_lstm2 = numpy.reshape(Y_lstm2, (Y_lstm2.shape[0]*60, 1))
t_train2 = numpy.arange(0, len(H_t2))
t_test2 = numpy.arange(len(H_t2), len(H_t2)+len(Y_lstm2))
t_array2 = numpy.arange(0, len(Y_lstm2))

numpy.save('Y_ST1.npy', Y_lstm2)
e_deep = (MathFunctions.rms_flat(Y_lstm2 - H_e2))/(MathFunctions.rms_flat(H_e2))
e_deep2 = (MathFunctions.rms_flat(Y_lstm2 - H_e2))/(MathFunctions.rms_flat(H_t2))

print e_deep
print e_deep2

Y_lstm2 = Y_lstm2*H_max2 #Converting to actual value
Y_lstm2 = Y_lstm2/60 #Converting to KW-h
Y_hour = DataFunctions.reduce_by_sum(Y_lstm2, 60)/H_max1
numpy.save('Y_ST2.npy', Y_hour)

print "H values"
print Y_hour
print H_e

e_deep3 = (MathFunctions.rms_flat(Y_hour - H_e))/(MathFunctions.rms_flat(H_e))
e_deep4 = (MathFunctions.rms_flat(Y_hour - H_e))/(MathFunctions.rms_flat(H_t))

print "Errors: "
print e_deep3
print e_deep4

#Plotting for hourly
#PlotFunctions.Plot_double(t_array, H_e, t_array, Y_lstm2, 'Actual conv power','LSTM conv power', 'k-', 'r-', "fig_77a.eps")
#PlotFunctions.Plot_triple(t_train, H_t, t_test, Y_lstm2, t_test, H_e, 'Training Data', 'LSTM predictions', 'Test Data (actual)', 'k-', 'r-', 'b-', "fig_77b.eps")

####Plotting for min-wise
PlotFunctions.Plot_double(t_array, H_e, t_array, Y_hour, 'Actual conv power','LSTM conv power', 'k-', 'r-', "fig_5houseII.eps")
numpy.save('Y_hour_10houseII.npy', Y_hour)
numpy.save('Y_lstm_10houseII.npy', Y_lstm2)