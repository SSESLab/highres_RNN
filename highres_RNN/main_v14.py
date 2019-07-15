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

#importing sklearn modules
from sklearn.cross_validation import KFold
from sklearn.cross_validation import cross_val_score
from sklearn.grid_search import GridSearchCV


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

#define hyperopt search space
import hyperopt.pyll.stochastic

seed = 7
numpy.random.seed(seed)


#--------------------------------------------------------------------------------------------
#Data Import and Processing
#getting training data
weather_file = 'WeatherData_WBB2013.xlsx'
energy_file = 'EnergySmallOffice_WBB2013.xlsx'

weather_out, weather_max, weather_min = DataFunctions.get_weatherdata(weather_file)
H = DataFunctions.get_energydata(energy_file)
H_max = numpy.amax(H)

#normalize weather data for training
weather_train = DataFunctions.normalize_vector(weather_out, weather_min, weather_max)
output_train = H/H_max


#Getting test Data
weather_file_e = 'WeatherData_WBB2014.xlsx'
energy_file_e = 'EnergySmallOffice_WBB2014.xlsx'

weather_out, filler1, filler2 = DataFunctions.get_weatherdata(weather_file_e)
H = DataFunctions.get_energydata(energy_file_e)

#normalize weather data for training
weather_test = DataFunctions.normalize_vector(weather_out, weather_min, weather_max)
output_test = H/H_max

#Seggregate by data
ClusterData = DataFunctions.separate_data_weekday(weather_train, output_train, weather_test, output_test)

#-----------------------------------------------------------------------
#Create DL network
#number of real schedules
real_num = 4
#number of binary schedules
binary_num = 3
#weather variables
weather_idxMax = 4

# Seeding random number stream
#seed = 7
#numpy.random.seed(seed)

#Constants for indexing, initialize vectors
N_day = 365
y_NN = numpy.zeros((N_day*24, ))
y_sim = numpy.zeros((N_day*24, ))
y_deep = numpy.zeros((N_day*24, ))
e_cv = numpy.zeros((3, 1))
#-----------------------------------------------------------
#This section is for optimizing hyper parameters
#reading csv for applying GP
read = DataFunctions.read_csvfile('hyp_1.csv')
#Space for hyper parameters
space = {
         'bin_units': hp.quniform('bin_units', 7, 10, 1),
         'real_units': hp.quniform('real_units', 30, 60, 1),
         'layer2_units': hp.quniform('layer2_units', 10, 50, 1)
         }


real_hyp = numpy.zeros((3, 1))
bin_hyp = numpy.zeros((3, 1))
layer2_hyp = numpy.zeros((3, 1))

DataFunctions.create_csvfile(1, 'hyp_1.csv') #choice = 1 if you want to over-write

for day in range(0, 3):
    col1 = 1
    row2, col2 = ClusterData[day].X_train.shape

    def objective(params):
        t = ClusterData[day].t
        X = ClusterData[day].X_train
        Y = ClusterData[day].Y_train
        cvscores = NNFunctions.NN_optimizeNN(t, X, Y, real_num, binary_num, weather_idxMax, params)
        e_temp = cvscores.mean()
        print e_temp
        return {'loss': e_temp, 'status': STATUS_OK}


    trials = Trials()

    best = fmin(objective, space, algo=tpe.suggest, trials=trials, max_evals=25)


    best_values = best.values()
    print best_values

    real_hyp[day] = best_values[0]
    bin_hyp[day] = best_values[1]
    layer2_hyp[day] = best_values[2]
    #print (trials.best_trial)

    #Writing best params to file
    input_dim = col1
    arr_write = numpy.array([input_dim, real_hyp[day]])
    DataFunctions.write_csvfile('hyp_1.csv', arr_write)


#---------------------
for day in range(0, 3):
    t_input1, real_out = NNFunctions.CreateRealSchedule_DL(1, real_num, real_hyp[day])
    t_input2, bin_out = NNFunctions.CreateBinSchedule_DL(1, binary_num, bin_hyp[day])

    #merging models
    # weather_input
    weather_input = Input(shape=(weather_idxMax,), name='weather_input')
    x = merge([real_out, bin_out, weather_input], mode='concat')
    x = Dense(layer2_hyp[day], activation='hard_sigmoid')(x)
    main_out = Dense(1, activation='linear')(x)

    main_model = Model(input=[t_input1, t_input2, weather_input], output=main_out)
    main_model.compile(loss='mse', optimizer='adam')

    #Re-training model multiple times
    main_model_new, cvscores = NNFunctions.fit_model(main_model, ClusterData[day].t, ClusterData[day].X_train, ClusterData[day].Y_train)

    main_model_new.fit([ClusterData[day].t, ClusterData[day].t, ClusterData[day].X_train], ClusterData[day].Y_train, nb_epoch=200, batch_size=20, verbose=0)
    Y_p = numpy.squeeze(main_model_new.predict([ClusterData[day].t, ClusterData[day].t, ClusterData[day].X_test]))


    for i in range(len(ClusterData[day].idx)):
        y_deep[ClusterData[day].idx[i]] = Y_p[i]
        y_sim[ClusterData[day].idx[i]] = ClusterData[day].Y_test[i]



#print y_deep
e_deep = (MathFunctions.rms_flat(y_deep - y_sim))/(MathFunctions.rms_flat(y_sim))
print e_deep
print e_cv
print real_hyp
print bin_hyp

PlotFunctions.PlotEnergy(y_deep, y_sim)
j1, j2 = DataFunctions.find_day_idx(4)
PlotFunctions.PlotEnergyDaily(y_deep[j1:j2], y_sim[j1:j2])
j1, j2 = DataFunctions.find_day_idx(180)
PlotFunctions.PlotEnergyDaily(y_deep[j1:j2], y_sim[j1:j2])
