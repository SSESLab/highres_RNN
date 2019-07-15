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

#Sklearn library for statistics
from sklearn.metrics import r2_score
from sklearn.datasets import fetch_20newsgroups_vectorized
from sklearn.feature_selection import chi2

#scipy library
from scipy.stats import pearsonr

#seeding
seed = 7
numpy.random.seed(seed)

####### The actual code Starts here
#######Training data: 2015-16

#EnergyData
#Processing hourly data
date_start = '01/01/15 12:00 AM'
date_end = '12/31/15 11:59 PM'
std_inv = 60 #in minutes

file_ = r'/home/sseslab/Documents/dataport_data/dataport26.csv'
data = DataFunctions.read_dataport_csv(file_)
H_t = data[0:len(data)-1] #removing las tentry

#Normalizing H_t
H_min, H_max = DataFunctions.get_normalize_params(H_t)
H_t = H_t/H_max
H_rms = MathFunctions.rms_flat(H_t)

print "The RMS is: ", H_rms

###Weather Data for Training
file_ = r'/home/sseslab/Documents/dataport_data/austin_weather_2015.csv'
df, weather_data = DataFunctions.read_weather_austin(file_, date_start, date_end)
weather_data = DataFunctions.interpolate_nans(weather_data)

#Appending the initial weather file
w0 = weather_data[0, :]
w0 = w0[None, :]
weather_train= numpy.append(w0, weather_data, axis=0)

#feature vectors
X_sch_t = DataFunctions.compile_features(H_t, date_start, std_inv)

#Getting Training Data
train_data = numpy.concatenate((weather_train[:, 0:2], X_sch_t), axis=1)

train_data = DataFunctions.normalize_2D(train_data)
X_val, Y_val  = DataFunctions.find_val_set(0, train_data, H_t)


#Getting Daily features
conv_hour_to_day = 24
H_mean_t, H_sum_t, H_min_t, H_max_t = DataFunctions.aggregate_data(H_t, conv_hour_to_day)
w_mean_t, w_sum_t, w_min_t, w_max_t = DataFunctions.aggregate_data(weather_train, conv_hour_to_day)
X_day_t = DataFunctions.compile_features(H_sum_t, date_start, 24*60)

#Saving module
numpy.save('X_t.npy', train_data)
numpy.save('Y_t.npy', H_t)

#############################################################################
#####Validation data
date_start = '01/01/16 12:00 AM'
date_end = '01/15/16 12:00 AM'
std_inv = 60 #in minutes

file_ = r'/home/sseslab/Documents/dataport_data/dataport26b.csv'
data = DataFunctions.read_dataport_csv(file_)
H_val = data[0:len(data)-1] #removing las tentry

#Normalizing H_t
H_val = H_val/H_max

###Weather Data for Training
file_ = r'/home/sseslab/Documents/dataport_data/austin_weather_2016.csv'
df, weather_data = DataFunctions.read_weather_austin(file_, date_start, date_end)
weather_data = DataFunctions.interpolate_nans(weather_data)

#Appending the initial weather file
weather_val = weather_data

#feature vectors
X_sch_val = DataFunctions.compile_features(H_val, date_start, std_inv)

#Getting Training Data
val_data = numpy.concatenate((weather_val[:, 0:2], X_sch_val), axis=1)
val_data = DataFunctions.normalize_2D(val_data)

###Section to over-ride specified validaiton data
val_data = X_val
H_val = Y_val

#Getting Daily features
H_mean_v, H_sum_v, H_min_v, H_max_v = DataFunctions.aggregate_data(H_val, conv_hour_to_day)
w_mean_v, w_sum_v, w_min_v, w_max_v = DataFunctions.aggregate_data(weather_val, conv_hour_to_day)
X_day_val = DataFunctions.compile_features(H_sum_v, date_start, 24*60)




############################################################################
####TEST DATA

date_start = '01/15/16 12:00 AM'
date_end = '12/31/16 12:00 AM'
std_inv = 60 #in minutes

file_ = r'/home/sseslab/Documents/dataport_data/dataport77c.csv'
data = DataFunctions.read_dataport_csv(file_)
H_e = data[0:len(data)-1] #removing las tentry

#Normalizing H_t
H_e =  H_e/H_max

###Weather Data for Training
file_ = r'/home/sseslab/Documents/dataport_data/austin_weather_2016.csv'
df, weather_data = DataFunctions.read_weather_austin(file_, date_start, date_end)
weather_data = DataFunctions.interpolate_nans(weather_data)

#Appending the initial weather file
weather_test = weather_data
print H_e
print H_e.shape
print weather_test.shape

#feature vectors
X_sch_e = DataFunctions.compile_features(H_e, date_start, std_inv)
print X_sch_e.shape
#Getting Training Data
test_data = numpy.concatenate((weather_test[:, 0:2], X_sch_e), axis=1)
test_data = DataFunctions.normalize_2D(test_data)

#Getting Daily features
H_mean_e, H_sum_e, H_min_e, H_max_e = DataFunctions.aggregate_data(H_e, conv_hour_to_day)
w_mean_e, w_sum_e, w_min_e, w_max_e = DataFunctions.aggregate_data(weather_test, conv_hour_to_day)
X_day_e = DataFunctions.compile_features(H_sum_e, date_start, 24*60)

#Saving module
numpy.save('X_e.npy', test_data)
numpy.save('Y_e.npy', H_e)

##############################################
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

X_sch_t = numpy.reshape(X_sch_t, (X_day_t.shape[0], 24, X_sch_t.shape[1]))
#X_sch_val = numpy.reshape(X_sch_val, (X_day_val.shape[0], 24, X_sch_val.shape[1]))
X_sch_e = numpy.reshape(X_sch_e, (X_day_e.shape[0], 24, X_sch_e.shape[1]))
#H_t = numpy.reshape(H_t, (H_mean_t.shape[0], 24, 1))
#H_e = numpy.reshape(H_e, (H_mean_e.shape[0], 24, 1))

H_t = numpy.reshape(H_t, (H_mean_t.shape[0], 24))
H_val = numpy.reshape(H_val, (H_mean_v.shape[0], 24))
H_e = numpy.reshape(H_e, (H_mean_e.shape[0], 24))


#####THis step is to optimize hyper-parameters
#This block is for optimizing LSTM layers
space = {
        'Layer1': hp.quniform('Layer1', 10, 100, 5),
        'Layer2': hp.quniform('Layer2', 10, 100, 5),
        'Layer3': hp.quniform('Layer3', 5, 20, 1),
        'activ_l3': hp.choice('activ_l3', ['relu', 'tanh']),
        'activ_l4': hp.choice('activ_l4', ['relu', 'sigmoid'])

    #'D1': hp.uniform('D1', 0, 0.5),
         #'D2': hp.uniform('D2', 0, 0.5),
         #'layer2_units': hp.quniform('layer2_units', 10, 50, 1)
         }


def objective(params):
    optimize_model = build_lstm_v1.lstm_model_110(params, train_data.shape[2], 24)

    #for epochs in range(5):
    for ep in range(20):
        optimize_history = optimize_model.fit(train_data, H_t, batch_size=1, nb_epoch=1, validation_data=(val_data, H_val), shuffle=False)
        #optimize_history = optimize_model.fit(train_data, H_t, batch_size=1, nb_epoch=1, validation_split=0.3, shuffle=False)
        optimize_model.reset_states()

    loss_v = optimize_history.history['val_loss']
    print loss_v

    loss_out = loss_v[-1]

    return {'loss': loss_out, 'status': STATUS_OK}


trials = Trials()
best = fmin(objective, space, algo=tpe.suggest, trials=trials, max_evals=40)


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
attempt_max = 5
epoch_max = 200
min_epoch = 20

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

    for ep in range(epoch_max):
        #lstm_history = lstm_model.fit(train_data, H_t, batch_size=1, nb_epoch=1, validation_data=(val_data, H_val), shuffle=False, callbacks=callbacks)
        #lstm_history = lstm_model.fit(train_data, H_t, batch_size=1, nb_epoch=1, validation_split=0.5, shuffle=False)
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


#lstm_history = lstm_model.fit(train_data, H_t, batch_size=1, nb_epoch=1, validation_split=(train_data, H_t), shuffle=False)
print val_loss_v
#save_model.reset_states()
print "count is"
print count

Y_lstm2 = save_model.predict(test_data, batch_size=1, verbose=0)

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

print e_deep
print e_deep2


#### Implement MLP Neural Network
best_NN = NNFunctions.NN_optimizeNN_v21(X1, H1, X2, H2)
NN_model = NNFunctions.CreateRealSchedule_v21(best_NN, X1.shape[1])
NN_savemodel = NN_model

epsilon = 1
val_loss_v = []

for attempts in range(0, 5):
    NN_model = NNFunctions.CreateRealSchedule_v22(best_NN, X1.shape[1])
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
e_NN = (MathFunctions.rms_flat(Y_NN - H_e))/(MathFunctions.rms_flat(H_e))
e_NN2 = (MathFunctions.rms_flat(Y_NN - H_e))/(MathFunctions.rms_flat(H_t))


print e_NN
print e_NN2
print "R2: "
print r2_score(Y_lstm2, H_e)

#Calculate p-value
a1, a2 = pearsonr(Y_lstm2, H_e)
print "LSTM rho-value: "
print a1

b1, b2 = pearsonr(Y_NN, H_e)
print "NN rho-value"
print b1




####Plotting
PlotFunctions.Plot_double(t_array, H_e, t_array, Y_lstm2, 'Actual conv power','LSTM conv power', 'k-', 'r-', "fig_77a.eps")
PlotFunctions.Plot_triple(t_train, H_t, t_test, Y_lstm2, t_test, H_e, 'Training Data', 'LSTM predictions', 'Test Data (actual)', 'k-', 'r-', 'b-', "fig_77b.eps")
PlotFunctions.Plot_quadruple(t_train, H_t, t_test, Y_lstm2, t_test, Y_NN, t_test, H_e, 'Training Data', 'LSTM predictions', 'MLP Predictions', 'Test Data (actual)', 'k-', 'r-', 'y-', 'b-', "fig_77d.eps")
