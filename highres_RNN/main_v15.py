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
import NNFun_PSB
#define hyperopt search space
import hyperopt.pyll.stochastic

seed = 7
numpy.random.seed(seed)

from datetime import  datetime

######## The actual code Starts here

#######Training data: 2015

#EnergyData
date_start = '5/14/16 12:00 AM'
date_end = '5/31/16 11:59 PM'
std_inv = 30 #in minutes

#Read data
data = DataFunctions.read_PSB_csv(date_start, date_end, 5, 20)
conv_critical, crac_critical, crac_normal, conv_normal, HVAC_critical, Elev, HVAC_normal, elec_total = DataFunctions.PSB_elec_split(data)
H_t = DataFunctions.fix_data(elec_total)


H_t = DataFunctions.fix_energy_intervals(H_t, 5, std_inv)
PlotFunctions.Plot_single(H_t)

energy_min, energy_max = DataFunctions.get_normalize_params(H_t)
#H_t = DataFunctions.normalize_vector(H_t, 0, energy_max)
H_t = H_t/energy_max

#Weather Data
weather_file = r'/home/sseslab/Documents/SLC PSB data/WBB Weather Data/WBB_2016_May.csv'
weather_train = DataFunctions.read_weather_csv(weather_file, date_start, date_end)
weather_train = DataFunctions.fix_weather_intervals(weather_train, 5, std_inv)

weather_min, weather_max = DataFunctions.get_normalize_params(weather_train)
#weather_train = DataFunctions.normalize_vector(weather_train, weather_min, weather_max)

#Schedule_data
X_sch_t = DataFunctions.compile_features(weather_train, date_start, std_inv)

print weather_train.shape
print X_sch_t.shape

###Test data

#EnergyData
date_start = '6/1/16 12:00 AM'
date_end = '6/30/16 11:59 PM'

#Read data
data = DataFunctions.read_PSB_csv(date_start, date_end, 5, 20)
conv_critical, crac_critical, crac_normal, conv_normal, HVAC_critical, Elev, HVAC_normal, elec_total = DataFunctions.PSB_elec_split(data)
H_e = DataFunctions.fix_data(conv_normal)

#PlotFunctions.plot_PSB_daily(H_e, 5)
H_e = DataFunctions.fix_energy_intervals(H_e, 5, std_inv)
#H_e = DataFunctions.normalize_vector(H_e, energy_min, energy_max)
H_e = H_e/energy_max
#Weather Data
weather_file = r'/home/sseslab/Documents/SLC PSB data/WBB Weather Data/WBB_2016_June.csv'
weather_test = DataFunctions.read_weather_csv(weather_file, date_start, date_end)
weather_test = DataFunctions.fix_weather_intervals(weather_test, 5, std_inv)

weather_test = DataFunctions.normalize_vector(weather_test, weather_min, weather_max)

#Schedule_data
X_sch_e = DataFunctions.compile_features(weather_test, date_start, std_inv)

#print H_t
#print H_e
#PlotFunctions.PlotEnergy(H_t[1:500], H_e[1:500])
#-----------------------------------------------------------------------------------------------
#Create DL network
#number of real schedules
#real_num = 4
#number of binary schedules
#binary_num = 4
#weather variables
data_length, weather_idxMax = weather_train.shape

# Seeding random number stream
#seed = 7
#numpy.random.seed(seed)

#Constants for indexing, initialize vectors


#-----------------------------------------------------------
#This section is for optimizing hyper parameters
#reading csv for applying GP
read = DataFunctions.read_csvfile('hyp_1.csv')
#Space for hyper parameters
space = {
        'bin_units': hp.quniform('bin_units', 5, 60, 1),
         'real_units': hp.quniform('real_units', 5, 60, 1),
         'real_num': hp.quniform('real_num', 1, 25, 1),
         'bin_num': hp.quniform('bin_num', 1, 25, 1),
         #'layer2_units': hp.quniform('layer2_units', 10, 50, 1)
         }



DataFunctions.create_csvfile(1, 'hyp_1.csv') #choice = 1 if you want to over-write


def objective(params):
    cvscores = NNFun_PSB.NN_optimizeNN_v2(X_sch_t, weather_train, H_t, params)
    e_temp = cvscores.mean()
    print e_temp

    return {'loss': e_temp, 'status': STATUS_OK}


trials = Trials()

best = fmin(objective, space, algo=tpe.suggest, trials=trials, max_evals=1)
print best

best_values = best.values()
print best_values

binary_num = best['bin_num']
bin_hyp = best['bin_units']

real_num = best['real_num']
real_hyp = best['real_units']



#---------------------------
#Fit model
row_sch, col_max = X_sch_t.shape
t_input1, real_out = NNFun_PSB.CreateRealSchedule_conv_PSB(col_max, real_num, real_hyp)
t_input2, bin_out = NNFun_PSB.CreateBinSchedule_conv_PSB(col_max, binary_num, bin_hyp)

#merging models
# weather_input
#weather_input = Input(shape=(weather_idxMax,), name='weather_input')
x = merge([real_out, bin_out], mode='concat')
#x = Dense(layer2_hyp, activation='hard_sigmoid')(x)
main_out = Dense(1, activation='linear')(x)

main_model = Model(input=[t_input1, t_input2], output=main_out)
main_model.compile(loss='mse', optimizer='adam')

#Re-training model multiple times
main_model, cvscores = NNFun_PSB.fit_model(main_model, X_sch_t, H_t)

main_model.fit([X_sch_t, X_sch_t], H_t, nb_epoch=100, batch_size=20, verbose=0)
Y_p = numpy.squeeze(main_model.predict([X_sch_e, X_sch_e]))
print Y_p
e_deep = (MathFunctions.rms_flat(Y_p - H_e))/(MathFunctions.rms_flat(H_e))

print e_deep


#PlotFunctions.PlotEnergy(Y_p, H_e)
#PlotFunctions.PlotEnergy(Y_p[0:144], H_e[0:144])
PlotFunctions.PlotEnergy(Y_p, H_e)
PlotFunctions.PlotEnergy(Y_p[0:96], H_e[0:96])
#print Y_p[0:47]
#print H_e[0:47]


for i in range(0, len(H_e)):
    if H_e[i] > 2 or H_e[i] < 0.2:
        H_e[i] = 0.6
        Y_p[i] = 0.6

e_deep2 = (MathFunctions.rms_flat(Y_p - H_e))/(MathFunctions.rms_flat(H_e))

print e_deep2