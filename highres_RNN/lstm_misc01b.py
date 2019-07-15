import matplotlib.pyplot as plt
import numpy
import pandas as pd
import MathFunctions
import scipy
import PlotFunctions
import DataFunctions

#scipy library
from scipy.stats import pearsonr

#y_t = numpy.load('CRAC_normal_train.npy')
y_t = numpy.load('Ht_MD1.npy')
y_val = numpy.load('Hv_MD1.npy')
y_t = numpy.squeeze(y_t)

print y_t.shape
print y_val.shape

#y_t = numpy.concatenate((y_t, y_val), axis=0)

y_e = numpy.load('He_MD1.npy')
#y1 = numpy.load('HVAC_normal_A.npy')
y2 = numpy.load('MD1_B.npy')
#y2 = numpy.reshape(y2, (y2.shape[0]*24, 1))
y3 = numpy.load('MD1_NN.npy')

choose_day =21
day_cons = 05


y_eA = y_e[choose_day*24:(choose_day+day_cons)*24]
#y_1A = y1[choose_day*24:(choose_day+day_cons)*24]
y_2A = y2[choose_day*24:(choose_day+day_cons)*24]
y_3A = y3[choose_day*24:(choose_day+day_cons)*24]
t_array = numpy.arange(0, len(y_2A))

print "Array Shapes"
print y2.shape
print t_array.shape
#print y_1A.shape

#sim_score = (MathFunctions.rms_flat(y_e - y_t[0:len(y_e)]))/(numpy.amax(y_t) - numpy.amin(y_t))
#print "The Similarity Score is: ", sim_score

#### Plotting
#plt.plot(t_array, y_1A, 'b-.', label='Model A Predictions', linewidth=5.00)
plt.rc('font', size=13)
plt.plot(t_array, y_2A, 'r:', label='Deep RNN Predictions',linewidth=2.00)
plt.plot(t_array, y_3A, 'g--', label='MLP Predictions', linewidth=2.00)
plt.plot(t_array, y_eA, 'k-', label='Ground Truth', linewidth=2.00)
# plt.plot(t/24, y3, 'r-', label='Case II: Modular Network without schedules')
#plt.ylim([0, 1.5])
plt.xlabel('Hours')
plt.ylim([0.4, 1.1])
plt.ylabel('Hourly heating load (normalized)')
# plt.axis([0, 2200, 0, y_max])
plt.legend(loc='upper right')
plt.savefig('fig_MD1A.eps')
plt.show()


#################################
###Figure 2
t_train = numpy.arange(0, len(y_t))
t_test = numpy.arange(len(y_t), len(y_t)+len(y2))
t_lim = t_train[-1]
t_max = t_test[-1]

print t_test.shape
print y_e.shape

plt.plot(t_train, y_t, 'k-', label="Ground Truth")
plt.plot(t_test, y_e, 'k-')
#plt.plot(t_test, y1, 'b-', label='Model A Predictions')
plt.plot(t_test, y2, 'r-', label='Deep RNN Predictions')
plt.plot(t_test, y3, 'g-', label='MLP Predictions')
plt.plot((t_lim, t_lim), (0, 1.8), 'k--')
plt.plot(((365 + choose_day)*24, (365 + choose_day)*24), (0, 1.8), 'b--')
plt.plot(((365 + choose_day + day_cons)*24, (365 + choose_day + day_cons)*24), (0, 1.8), 'b--')

# plt.plot(t/24, y3, 'r-', label='Case II: Modular Network without schedules')

# Adding Annotations
plt.annotate('', xy=(t_lim, 1.2), xycoords='data', xytext=(0, 1.2), textcoords='data', arrowprops={'arrowstyle': '<->'})
plt.annotate('Training Phase', xy=(int(t_lim / 2), 1.0), xycoords='data', xytext=(4000, 1.15), textcoords='data')
plt.annotate('', xy=(t_max, 1.2), xycoords='data', xytext=(t_lim, 1.2), textcoords='data',
             arrowprops={'arrowstyle': '<->'})
plt.annotate('Test Phase', xy=(int((t_max - t_lim) / 2), 1.0), xycoords='data', xytext=(10000, 1.15), textcoords='data')

plt.xlabel('Hours')
plt.ylabel('Hourly heating load (normalized)')
plt.axis([0, t_max, 0.0, 1.8])
plt.legend(loc='upper right')
plt.savefig('fig_MD2A.eps')
plt.show()

##############################
#Computing error
print "Heating Loads: "
print y2.shape
print y3.shape
e_B = (MathFunctions.rms_flat(y2- y_e))/(MathFunctions.rms_flat(y_t))
e_NN = (MathFunctions.rms_flat(y3- y_e))/(MathFunctions.rms_flat(y_t))
print "errors: :"
print e_B, e_NN




###################
idx_1 = 31*24
idx_2 = 59*24

y_b0 = y2[idx_1:idx_2]
y_nn = y3[idx_1:idx_2]
y_e0 = y_e[idx_1:idx_2]

e_B0 = (MathFunctions.rms_flat(y_b0- y_e0))/(MathFunctions.rms_flat(y_t))
e_NN0 = (MathFunctions.rms_flat(y_nn - y_e0))/(MathFunctions.rms_flat(y_t))
print e_B0, e_NN0


###
e_b = numpy.asarray([0.226, 0.2850, 0.286, 0.238, 0.439, 0.282, 0.199])
e_nn = numpy.asarray([0.306, 0.3622, 0.288, 0.22, 0.262, 0.260, 0.220])

###Bar chart
ind = numpy.arange(len(e_b))  # the x locations for the groups
width = 0.35       # the width of the bars
fig, ax = plt.subplots()
rects1 = ax.bar(ind, e_b, width, color='r')
rects2 = ax.bar(ind + width, e_nn, width, color='g')
ax.set_ylim([0, 0.6])
ax.set_xlabel('Month')
ax.set_ylabel('$e_1$')
ax.set_xticks(ind + width)
ax.set_xticklabels(('Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul'))
ax.legend((rects1[0], rects2[0]), ('Deep RNN Predictions', 'MLP Predictions'))
plt.savefig('fig_bar1_new.eps')
plt.show()

y_tp = y_t[0:59*24]
y_ep = y_e[0:59*24]
s = MathFunctions.rms_flat(y_tp  - y_ep)/(numpy.amax(y_ep) - numpy.amin(y_ep))
print "s = ", s