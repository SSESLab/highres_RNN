import matplotlib.pyplot as plt
import numpy
import pandas as pd
import MathFunctions
import scipy
import PlotFunctions
import DataFunctions

#scipy library
from scipy.stats import pearsonr

y_t = numpy.load('HVAC_critical_train.npy')
#y_t = numpy.load('1_train.npy')
y_val = numpy.load('HVAC_critical_val.npy')
y_t = numpy.squeeze(y_t)
print y_t.shape


y_t = numpy.concatenate((y_t, y_val), axis=0)

y_e = numpy.load('HVAC_critical_test.npy')
y2 = numpy.load('HVAC_critical_B.npy')
#y2 = numpy.reshape(y2, (y2.shape[0]*24, 1))
y3 = numpy.load('HVAC_critical_MLP.npy')

choose_day =30
day_cons = 35

print y3.shape

y_eA = y_e[choose_day*24:(choose_day+day_cons)*24]
#y_1A = y1[choose_day*24:(choose_day+day_cons)*24]
y_2A = y2[choose_day*24:(choose_day+day_cons)*24]
y_3A = y3[choose_day*24:(choose_day+day_cons)*24]
t_array = numpy.arange(0, len(y_2A))

###plot 1

#### Plotting
#plt.plot(t_array, y_1A, 'b-.', label='Model A Predictions', linewidth=5.00)
plt.rc('font', size=13)
plt.plot(t_array, y_2A, 'r:', label='Deep RNN Predictions',linewidth=2.00)
plt.plot(t_array, y_3A, 'g--', label='MLP Predictions', linewidth=2.00)
plt.plot(t_array, y_eA, 'k-', label='Ground Truth', linewidth=2.00)
# plt.plot(t/24, y3, 'r-', label='Case II: Modular Network without schedules')
#plt.ylim([0, 1.5])
plt.xlabel('Hours')
plt.ylim([0.2, 1.1])
plt.ylabel('Hourly heating load (normalized)')
# plt.axis([0, 2200, 0, y_max])
plt.legend(loc='upper right')
#plt.savefig('fig_HVAC2A.eps')
plt.show()


###Figure 2
t_train = numpy.arange(0, len(y_t))
t_test = numpy.arange(len(y_t), len(y_t)+len(y2))
t_lim = t_train[-1]
t_max = t_test[-1]

plt.plot(t_train, y_t, 'k-', label="Ground Truth")
plt.plot(t_test, y_e, 'k-')
#plt.plot(t_test, y1, 'b-', label='Model A Predictions')
plt.plot(t_test, y2, 'r-', label='Model B Predictions')
plt.plot(t_test, y3, 'g-', label='MLP Predictions')
plt.plot((t_lim, t_lim), (0, 1.8), 'k--')

# plt.plot(t/24, y3, 'r-', label='Case II: Modular Network without schedules')

# Adding Annotations
plt.annotate('', xy=(t_lim, 1.2), xycoords='data', xytext=(0, 1.2), textcoords='data', arrowprops={'arrowstyle': '<->'})
plt.annotate('Training Phase', xy=(int(t_lim / 2), 1.0), xycoords='data', xytext=(4000, 1.15), textcoords='data')
plt.annotate('', xy=(t_max, 1.2), xycoords='data', xytext=(t_lim, 1.2), textcoords='data',
             arrowprops={'arrowstyle': '<->'})
plt.annotate('Test Phase', xy=(int((t_max - t_lim) / 2), 1.0), xycoords='data', xytext=(9000, 1.15), textcoords='data')

plt.xlabel('Hours')
plt.ylabel('Hourly electric load (normalized)')
plt.axis([0, t_max, 0.0, 1.8])
plt.legend(loc='upper right')
plt.savefig('fig_HVAC2A.eps')
plt.show()



###
print y2.shape
print y_e.shape

month_list = [31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
err_B = numpy.zeros((12, 1))
err_MLP = numpy.zeros((12, 1))

idx_1 = 0
idx_2 = 0

for i in range(0, len(month_list)):

    print idx_1
    print idx_2


    idx_1 = idx_2
    idx_2 = (month_list[i] + idx_2)

    err_B[i] = (MathFunctions.rms_flat(y2[24*idx_1:24*idx_2] - y_e[24*idx_1:24*idx_2])) / (MathFunctions.rms_flat(y_e[24*idx_1:24*idx_2]))

    #err_B[i] = (MathFunctions.rms_flat(y2[24 * idx_1:24 * idx_2] - y_e[24 * idx_1:24 * idx_2])) / (
    #MathFunctions.rms_flat(y_t))


    err_MLP[i] = (MathFunctions.rms_flat(y3[24 * idx_1:24 * idx_2] - y_e[24 * idx_1:24 * idx_2])) / (
    MathFunctions.rms_flat(y_e[24 * idx_1:24 * idx_2]))

    #err_MLP[i] = (MathFunctions.rms_flat(y3[24 * idx_1:24 * idx_2] - y_e[24 * idx_1:24 * idx_2])) / (
        #MathFunctions.rms_flat(y_t))

print err_B
print err_MLP

fig, ax = plt.subplots()
index = numpy.arange(len(month_list))
bar_width = 0.4

opacity = 0.4

rects1 = plt.bar(index, err_B, bar_width,  color='b', label= 'Model B Prerdictions' )
rects2 = plt.bar(index+bar_width, err_MLP, bar_width, color='r', label= 'MLP Prerdictions' )

ax.legend((rects1[0], rects2[0]), ('Model B Errors', 'MLP Errors'))
plt.xlabel('Month')
plt.xticks(index + bar_width, ('Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'))
plt.ylabel('Relative error, $e_1$')
plt.ylim([0, 0.45])
plt.savefig('fig_bar1.eps')
plt.show()



###################################################

HVAC1_R = [0.165, 0.154, 0.159, 0.183]
HVAC1_U = [0.171, 0.151, 0.192, 0.149]

fig, ax = plt.subplots()
index = numpy.arange(len(HVAC1_R))

rects1 = plt.bar(index, HVAC1_R, bar_width,  color='b', label= 'Real-valued' )
rects2 = plt.bar(index+bar_width, HVAC1_U, bar_width, color='r', label= 'Unary')

ax.legend((rects1[0], rects2[0]), ('Real-valued', 'Unary'))
plt.xlabel('Strategy')
plt.xticks(index + bar_width, ('Strategy 0', 'Strategy I', 'Strategy II', 'Strategy III'))
plt.ylabel('Relative error, $e_1$')
plt.ylim([0.14, 0.22])
plt.savefig('fig_barI.eps')
plt.show()

HVAC2_R = [0.178, 0.177, 0.208, 0.186]
HVAC2_U = [0.255, 0.155, 0.509, 0.190]

fig, ax = plt.subplots()

rects1 = plt.bar(index, HVAC2_R, bar_width,  color='b', label= 'Real-valued' )
rects2 = plt.bar(index+bar_width, HVAC2_U, bar_width, color='r', label= 'Unary')

ax.legend((rects1[0], rects2[0]), ('Real-valued', 'Unary'))
plt.xlabel('Strategy')
plt.xticks(index + bar_width, ('Strategy 0', 'Strategy I', 'Strategy II', 'Strategy III'))
plt.ylabel('Relative error, $e_1$')
plt.ylim([0.14, 0.58])
plt.savefig('fig_barII.eps')
plt.show()



CONV1_R = [0.162, 0.163, 0.169, 0.177]
CONV1_U = [0.169, 0.170, 0.173, 0.169]

fig, ax = plt.subplots()

rects1 = plt.bar(index, CONV1_R, bar_width,  color='b', label= 'Real-valued' )
rects2 = plt.bar(index+bar_width, CONV1_U, bar_width, color='r', label= 'Unary')

ax.legend((rects1[0], rects2[0]), ('Real-valued', 'Unary'))
plt.xlabel('Strategy')
plt.xticks(index + bar_width, ('Strategy 0', 'Strategy I', 'Strategy II', 'Strategy III'))
plt.ylabel('Relative error, $e_1$')
plt.ylim([0.15, 0.20])
plt.savefig('fig_barIII.eps')
plt.show()


CONV2_R = [0.158, 0.165, 0.158, 0.177]
CONV2_U = [0.182, 0.186, 0.187, 0.182]

fig, ax = plt.subplots()

rects1 = plt.bar(index, CONV2_R, bar_width,  color='b', label= 'Real-valued' )
rects2 = plt.bar(index+bar_width, CONV2_U, bar_width, color='r', label= 'Unary')

ax.legend((rects1[0], rects2[0]), ('Real-valued', 'Unary'))
plt.xlabel('Strategy')
plt.xticks(index + bar_width, ('Strategy 0', 'Strategy I', 'Strategy II', 'Strategy III'))
plt.ylabel('Relative error, $e_1$')
plt.ylim([0.15, 0.20])
plt.savefig('fig_barIV.eps')
plt.show()


CRAC1_R = [0.427, 0.968, 1.018, 0.766]
CRAC1_U = [0.745, 1.08, 0.476, 0.391]

fig, ax = plt.subplots()

rects1 = plt.bar(index, CRAC1_R, bar_width,  color='b', label= 'Real-valued' )
rects2 = plt.bar(index+bar_width, CRAC1_U, bar_width, color='r', label= 'Unary')

ax.legend((rects1[0], rects2[0]), ('Real-valued', 'Unary'))
plt.xlabel('Strategy')
plt.xticks(index + bar_width, ('Strategy 0', 'Strategy I', 'Strategy II', 'Strategy III'))
plt.ylabel('Relative error, $e_1$')
plt.ylim([0.3, 1.3])
plt.savefig('fig_barV.eps')
plt.show()



CRAC1_R = [0.427, 0.968, 1.018, 0.766]
CRAC1_U = [0.745, 1.08, 0.476, 0.391]

fig, ax = plt.subplots()

rects1 = plt.bar(index, CRAC1_R, bar_width,  color='b', label= 'Real-valued' )
rects2 = plt.bar(index+bar_width, CRAC1_U, bar_width, color='r', label= 'Unary')

ax.legend((rects1[0], rects2[0]), ('Real-valued', 'Unary'))
plt.xlabel('Strategy')
plt.xticks(index + bar_width, ('Strategy 0', 'Strategy I', 'Strategy II', 'Strategy III'))
plt.ylabel('Relative error, $e_1$')
plt.ylim([0.3, 1.3])
plt.savefig('fig_barV.eps')
plt.show()

CRAC1_R = [0.427, 0.968, 1.018, 0.766]
CRAC1_U = [0.745, 1.08, 0.476, 0.391]

fig, ax = plt.subplots()

rects1 = plt.bar(index, CRAC1_R, bar_width,  color='b', label= 'Real-valued' )
rects2 = plt.bar(index+bar_width, CRAC1_U, bar_width, color='r', label= 'Unary')

ax.legend((rects1[0], rects2[0]), ('Real-valued', 'Unary'))
plt.xlabel('Strategy')
plt.xticks(index + bar_width, ('Strategy 0', 'Strategy I', 'Strategy II', 'Strategy III'))
plt.ylabel('Relative error, $e_1$')
plt.ylim([0.3, 1.3])
plt.savefig('fig_barV.eps')
plt.show()


CRAC1_R = [0.427, 0.968, 1.018, 0.766]
CRAC1_U = [0.745, 1.08, 0.476, 0.391]

fig, ax = plt.subplots()

rects1 = plt.bar(index, CRAC1_R, bar_width,  color='b', label= 'Real-valued' )
rects2 = plt.bar(index+bar_width, CRAC1_U, bar_width, color='r', label= 'Unary')

ax.legend((rects1[0], rects2[0]), ('Real-valued', 'Unary'))
plt.xlabel('Strategy')
plt.xticks(index + bar_width, ('Strategy 0', 'Strategy I', 'Strategy II', 'Strategy III'))
plt.ylabel('Relative error, $e_1$')
plt.ylim([0.3, 1.3])
plt.savefig('fig_barV.eps')
plt.show()

CRAC1_R = [0.36, 1.192, 0.779, 0.308]
CRAC1_U = [0.334, 1.08, 0.548, 1.238]

fig, ax = plt.subplots()

rects1 = plt.bar(index, CRAC1_R, bar_width,  color='b', label= 'Real-valued' )
rects2 = plt.bar(index+bar_width, CRAC1_U, bar_width, color='r', label= 'Unary')

ax.legend((rects1[0], rects2[0]), ('Real-valued', 'Unary'))
plt.xlabel('Strategy')
plt.xticks(index + bar_width, ('Strategy 0', 'Strategy I', 'Strategy II', 'Strategy III'))
plt.ylabel('Relative error, $e_1$')
plt.ylim([0.3, 1.4])
plt.savefig('fig_barVI.eps')
plt.show()