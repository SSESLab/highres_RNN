from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers import Input, Merge
from keras.layers import merge
from keras.models import Model
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

#Creating Class for Schedules
class ScheduleData(object):
    """__init__() functions as the class constructor"""

    def __init__(self, model=None):
        self.model = model


print


#manual functions
import ArrayFunctions
import MathFunctions
import DataFunctions
import NNFunctions
import PlotFunctions

#Data Extraction
ClusterData = DataFunctions.ExtractData()

#Creating ScheduleModel
ScheduleModel = NNFunctions.CreateScheduleModel()
BinScheduleMOdel = NNFunctions.CreateBinaryScheduleModel()
MainModel = NNFunctions.CreateMainModel()

#number of real schedules
real_num = 4
RealSchModel, real_out = NNFunctions.CreateRealSchedule_v2(1, real_num)
real_schedule_model = []

for i in range(0, real_num):
    #real_schedule.append(ScheduleData(ScheduleModel))
    real_schedule_model.append(NNFunctions.CreateScheduleModel())

#number of binary schedules
binary_num = 3
BinSchModel, bin_out = NNFunctions.CreateBinSchedule_v2(1, binary_num)
bin_schedule_model = []

for i in range(0, binary_num):
    #real_schedule.append(ScheduleData(ScheduleModel))
    bin_schedule_model.append(NNFunctions.CreateBinaryScheduleModel())


#Choices of hyperparameters
epoch_choice = numpy.array([10, 20])
batch_choice = numpy.array([5, 10, 20])
dropout_choice = [0.0, 0.2, 0.5]

# Seeding random number stream
seed = 7
numpy.random.seed(seed)

#Constants for indexing, initialize vectors
N_day = 365
y_NN = numpy.zeros((N_day*24, ))
y_sim = numpy.zeros((N_day*24, ))
y_deep = numpy.zeros((N_day*24, ))

#constants specific to Electric Load Prediction
weather_idxMax = 4 # Maximum  number of weather features
Sch_idxMax = 7

# Initialize matrices for error
e_sch = numpy.zeros((Sch_idxMax, 3))
err_real = numpy.zeros((real_num, 3))
err_bin = numpy.zeros((binary_num, 3))

for day in range(0, 3):
    D = ClusterData[day].DataTrain
    X, Y, idx, W, Sch, t = ArrayFunctions.SeparateCSV(D, weather_idxMax, Sch_idxMax)
    w_num = W.shape[1]
    RealSch_t = Sch[:, 0:real_num]
    BinSch_t = Sch[:, real_num: real_num+binary_num]


    D2 = ClusterData[day].DataTest
    X_e, Y_e, idx_e, W_e, Sch_e, t = ArrayFunctions.SeparateCSV(D2, weather_idxMax, Sch_idxMax)
    RealSch_e = Sch_e[:, 0:real_num]
    BinSch_e = Sch_e[:, real_num: real_num + binary_num]

    N = len(Sch) #N is the total number of observations in the given cluster
    Sch_test = numpy.zeros((N, Sch_idxMax))

    #Assigning training and test to class
    ClusterData[day].X_train = X
    ClusterData[day].Y_train = Y
    ClusterData[day].idx = idx
    ClusterData[day].X_test = X_e
    ClusterData[day].Y_test = Y_e

    earlyStopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=0, verbose=0, mode='auto')
    BinSchModel.fit(t, BinSch_t, nb_epoch=100, batch_size=20, validation_split=0.2, verbose=0, callbacks=[earlyStopping])
    BinSch_pred = BinSchModel.predict(t)
    BinSch_pred = DataFunctions.ConvertToBinary(BinSch_pred)

    r1 = RealSchModel.fit(t, RealSch_t, nb_epoch=100, batch_size=20, validation_split=0.2, verbose=0, callbacks=[earlyStopping])
    RealSch_pred = RealSchModel.predict(t)
    RealSch_pred = DataFunctions.FixRealValue(RealSch_pred)


    e_t1 = RealSch_t - RealSch_pred

    for j in range(0, real_num):
        err_real[j, day] = (MathFunctions.rms_flat(e_t1)/(MathFunctions.rms_flat(RealSch_t[:, j])))

    Sch_cat = numpy.concatenate((RealSch_pred, BinSch_pred), axis=1)

#Fit Model

    #Create the main model
    #MainModel2.add
    #Real Model
    t_input = Input(shape=(1,))
    x = Dense(50, activation='hard_sigmoid')(t_input)
    real_out = Dense(real_num, activation='linear')(x)

    #Bin Model
    t_input2 = Input(shape=(1,))
    x = Dense(512, activation='tanh')(t_input2)
    bin_out = Dense(binary_num, activation='sigmoid')(x)

    #weather_input
    weather_input = Input(shape=(weather_idxMax, ), name='weather_input')

    x = merge([real_out, bin_out, weather_input], mode='concat')
    x = Dense(20, activation='hard_sigmoid')(x)
    main_out = Dense(1, activation='linear')(x)

    main_model = Model(input=[t_input, t_input2, weather_input], output=main_out)
    main_model.compile(loss='mse', optimizer='adam')
    main_model.fit([t, t, W], ClusterData[day].Y_train, nb_epoch=200, batch_size=20, verbose=0)
    Y_p = numpy.squeeze(main_model.predict([t, t, W_e]))

    print Y_p[0:23]
    print Y_e[0:23]

    print ClusterData[day].idx[0:30]
    print Y_p.shape
    print y_deep.shape

    for i in range(len(ClusterData[day].idx)):
        y_deep[ClusterData[day].idx[i]-1] = Y_p[i]


            #print Sch_copy
    #Replacing X_train
    ClusterData[day].X_train[:, weather_idxMax:(weather_idxMax + Sch_idxMax)] = Sch_cat


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
        y_NN[ClusterData[day].idx[i]-1] = y_pred[i]
        y_sim[ClusterData[day].idx[i]-1] = Y_test[i]

#Outputs
print y_NN
print y_sim
print y_deep
e_rms = (MathFunctions.rms_flat(y_NN - y_sim))/(MathFunctions.rms_flat(y_sim))
e_deep = (MathFunctions.rms_flat(y_deep - y_sim))/(MathFunctions.rms_flat(y_sim))
print e_rms
print e_deep


PlotFunctions.PlotEnergy(y_NN, y_sim, y_deep)
j1, j2 = DataFunctions.find_day_idx(4)
PlotFunctions.PlotEnergyDaily(y_NN[j1:j2], y_sim[j1:j2], y_deep[j1:j2])
j1, j2 = DataFunctions.find_day_idx(180)
PlotFunctions.PlotEnergyDaily(y_NN[j1:j2], y_sim[j1:j2], y_deep[j1:j2])
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


print err_real
print j1
print j2
print y_NN[j1:j2]
print y_NN[j1:j2].shape