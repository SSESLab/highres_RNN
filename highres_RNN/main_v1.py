from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers import Merge
from keras.utils.visualize_util import plot
import numpy
import pydot
import keras
print pydot.find_graphviz()

#hyperas modules
import pymongo
import hyperas
import hyperopt

from sklearn.metrics import mean_squared_error
from numpy import linalg as LA

from math import sqrt
# importing theano and running on gpu

#manual functions
import ArrayFunctions
import MathFunctions

#Importing all .csv files
D1_train = numpy.loadtxt("WBB_SaturdayData_train.csv", delimiter=',')
D1_test = numpy.loadtxt("WBB_SaturdayData_test.csv", delimiter=',')
D2_train = numpy.loadtxt("WBB_SundayData_train.csv", delimiter=',')
D2_test = numpy.loadtxt("WBB_SundayData_test.csv", delimiter=',')
D3_train = numpy.loadtxt("WBB_WeekdayData_train.csv", delimiter=',')
D3_test = numpy.loadtxt("WBB_WeekdayData_test.csv", delimiter=',')

#Creating class that can be used to create a list of objects
class DaySelect(object):
    """__init__() functions as the class constructor"""

    def __init__(self, DataTrain=None, DataTest=None):
        self.DataTrain = DataTrain
        self.DataTest = DataTest
print

#Seggregating by cluster, where each cluster represents, Saturday, Sunday and Weekdays

ClusterData = []
ClusterData.append(DaySelect(D1_train, D1_test))
ClusterData.append(DaySelect(D2_train, D2_test))
ClusterData.append(DaySelect(D3_train, D3_test))

#Building a schedule model
ScheduleModel = Sequential()
ScheduleModel.add(Dense(20, input_dim=1))
ScheduleModel.add(Activation('hard_sigmoid'))
ScheduleModel.add(Dense(1))
ScheduleModel.add(Activation('linear'))

#Compile model
#keras.optimizers.SGD(lr=0.01, momentum=0.0, decay=0.0, nesterov=False)
ScheduleModel.compile(loss='mse', optimizer='adam')
plot(ScheduleModel, to_file='ScheduleModel.png')

#Building the model for prediction model
MainModel = Sequential()
MainModel.add(Dense(15, input_dim=11))
MainModel.add(Activation('sigmoid'))
MainModel.add(Dense(1))

#Compile model
MainModel.compile(loss='mse', optimizer='adam')
plot(MainModel, to_file='MainModel.png')


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
        #earlyStopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=2, verbose=0, mode='auto') #setting early stopping criteria
        ScheduleModel.fit(t, Sch[:, j], nb_epoch=100, batch_size=20, validation_split=0.1, verbose=0, callbacks=[earlyStopping])
        print (ScheduleModel.predict(t)).shape
        Sch_test[:, j] = numpy.squeeze(ScheduleModel.predict(t))
        Sch_copy = Sch[:, j]
        Sch_copy = numpy.squeeze(Sch_copy[:, None])
        e_t = Sch_test[:, j] - (Sch_copy)
        e_sch[j, day] = (MathFunctions.rms_flat(e_t))/(MathFunctions.rms_flat(Sch[:, j]))

        score = ScheduleModel.evaluate(t, Sch_copy, verbose=0)
        print('Test accuracy:', score)
        print t[0:47]
        print Sch_copy[0:47]
        print Sch_test[0:47, j]
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



print y_NN
print y_sim
e_rms = (MathFunctions.rms_flat(y_NN - y_sim))/(MathFunctions.rms_flat(y_sim))
print e_rms
#print Sch_test.shape
#print Sch_test
#print ClusterData[day].X_train.shape
#print ClusterData[day].X_train

#print "The Schedule error is: "
print e_sch
#print ClusterData[2].X_train[0:47, weather_idxMax:(weather_idxMax + Sch_idxMax)]
#print Sch_test[0:47, :]












