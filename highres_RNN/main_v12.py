from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers import Merge
from keras.utils.visualize_util import plot
from keras.wrappers.scikit_learn import KerasClassifier, KerasRegressor

from sklearn.cross_validation import StratifiedKFold
from sklearn.cross_validation import cross_val_score
from sklearn.grid_search import GridSearchCV
import numpy
import pydot
import keras
print pydot.find_graphviz()

#MATLAB functions
from pymatbridge import Matlab
mlab = Matlab()

#hyperas modules
import pymongo
import hyperas
import hyperopt
from hyperopt import Trials, STATUS_OK, tpe
from hyperas import optim
from hyperas.distributions import choice, uniform


from sklearn.metrics import mean_squared_error
from numpy import linalg as LA

from math import sqrt
# importing theano and running on gpu

#manual functions
import ArrayFunctions
import MathFunctions
import DataFunctions
import NNFunctions

#Data Extraction
ClusterData = DataFunctions.ExtractData()

#Creating ScheduleModel
ScheduleModel = NNFunctions.CreateScheduleModel()
BinScheduleMOdel = NNFunctions.CreateBinaryScheduleModel()
MainModel = NNFunctions.CreateMainModel()

#Choices of hyperparameters
epoch_choice = numpy.array([10, 20])
batch_choice = numpy.array([5, 10, 20])
dropout_choice = [0.0, 0.2, 0.5]

# Seeding random number stream
seed = 7
numpy.random.seed(seed)

#Constants for indexing
N_day = 365
y_NN = numpy.zeros((N_day*365, ))
y_sim = numpy.zeros((N_day*365, ))

#constants specific to Electric Load Prediction
weather_idxMax = 4 # Maximum  number of weather features
Sch_idxMax = 7

# Initialize matrices for error
e_sch = numpy.zeros((Sch_idxMax, 3))

for day in range(0, 3):
    D = ClusterData[day].DataTrain
    X, Y, idx, W, Sch, t = ArrayFunctions.SeparateCSV(D, weather_idxMax, Sch_idxMax)

    D2 = ClusterData[day].DataTest
    X_e, Y_e, idx_e, W_e, Sch_e, t = ArrayFunctions.SeparateCSV(D2, weather_idxMax, Sch_idxMax)

    N = len(Sch) #N is the total number of observations in the given cluster
    Sch_test = numpy.zeros((N, Sch_idxMax))

    #Assigning training and test to class
    ClusterData[day].X_train = X
    ClusterData[day].Y_train = Y
    ClusterData[day].idx = idx
    ClusterData[day].X_test = X_e
    ClusterData[day].Y_test = Y_e

#Fit Model
    for j in range(0, Sch_idxMax):
        #SchCV_model = KerasRegressor(build_fn=NNFunctions.CreateScheduleModel)
        # Find best parameters
        #param_grid = dict(nb_epoch=epoch_choice, batch_size=batch_choice, dropout_rate=dropout_choice)
        #grid = GridSearchCV(estimator=SchCV_model, param_grid=param_grid, verbose=0, scoring='mean_squared_error')
        #grid_result = grid.fit(t, Sch[:, j])

        #print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
        #for params, mean_score, scores in grid_result.grid_scores_:
            #print("%f (%f) with: %r" % (scores.mean(), scores.std(), params))

        earlyStopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=0, verbose=0, mode='auto')

        if j > 3:
            BinScheduleMOdel.fit(t, Sch[:, j], nb_epoch=100, batch_size=20, validation_split=0.2, verbose=0, callbacks=[earlyStopping])
            print (BinScheduleMOdel.predict(t)).shape
            Sch_test[:, j] = numpy.squeeze(BinScheduleMOdel.predict(t))
            #Sch_test[:, j] = DataFunctions.ConvertToBinary(Sch_test[:, j])
        else:
            ScheduleModel.fit(t, Sch[:, j], nb_epoch=100, batch_size=20, validation_split=0.2, verbose=0, callbacks=[earlyStopping])
            print (ScheduleModel.predict(t)).shape
            Sch_test[:, j] = numpy.squeeze(ScheduleModel.predict(t))

        Sch_copy = Sch[:, j]
        Sch_copy = numpy.squeeze(Sch_copy[:, None])
        e_t = Sch_test[:, j] - Sch_copy
        e_sch[j, day] = (MathFunctions.rms_flat(e_t))/(MathFunctions.rms_flat(Sch[:, j]))

        #score = ScheduleModel.evaluate(t, Sch_copy, verbose=0)
        #print('Test accuracy:', score)
        print t[0:24]
        print Sch_copy[0:24]
        print Sch_test[0:24, j]


        #print Sch_copy
    #Replacing X_train
    ClusterData[day].X_train[:, weather_idxMax:(weather_idxMax + Sch_idxMax)] = Sch_test


for day in range(0, 3):
    X_train = ClusterData[day].X_train
    Y_train = ClusterData[day].Y_train
    X_test = ClusterData[day].X_test
    Y_test = ClusterData[day].Y_test

    MainModel.fit(X_train, Y_train, nb_epoch=20, batch_size=20, verbose=0)
    print X_test.shape
    y_pred = numpy.squeeze(MainModel.predict(X_test))
    ClusterData[day].y_pred = y_pred

    for i in range(len(ClusterData[day].idx)):
        y_NN[ClusterData[day].idx[i]] = y_pred[i]
        y_sim[ClusterData[day].idx[i]] = Y_train[i]

#Outputs
print e_sch
print y_NN
print y_sim
e_rms = (MathFunctions.rms_flat(y_NN - y_sim))/(MathFunctions.rms_flat(y_sim))
print e_rms

#debug
#print Sch
#print Sch_test
#print Sch[0:47, 1]
#print weather_idxMax
#print Sch_idxMax
#print j
#print ClusterData[0].X_train[0:23, :]
#print ClusterData[1].X_train[0:23, :]
#print ClusterData[2].X_train[0:23, :]
#mlab.start()
#results = mlab.run_code('a=1;')
#print mlab.get_variable('a')

#res = mlab.run_func('jk.m', {'arg1': [2, 4], 'arg2': [3, 7]})
#res = mlab.run_func('funLearnSchedule_v1.m', {'hyparam': [10, 3], 'schedule_train': Sch[0:47, 1], 'schedule_test': Sch[0:47, 1]})
#print(res['result'])