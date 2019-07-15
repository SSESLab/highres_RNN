import matplotlib.pyplot as plt
import numpy
import pandas as pd
import MathFunctions
import scipy
import PlotFunctions
import DataFunctions

#scipy library
from scipy.stats import pearsonr, pointbiserialr

#y_t = numpy.load('CRAC_normal_train.npy')
y_t = numpy.load('Ht2_CRAC1.npy')
Yt = y_t.copy()
y_val = numpy.load('Hv2_CRAC1.npy')
y_t = numpy.concatenate((y_t, y_val), axis=0)
y_t = numpy.squeeze(y_t)
y_e = numpy.load('He2_CRAC1.npy')

y1 = numpy.load('CRAC_critical_10min0.npy')
y2 = numpy.load('CRAC_critical_10min2B0.npy')
y3 = numpy.load('CRAC_critical_10min2C0.npy')
y4 = numpy.load('CRAC_critical_10min2D0.npy')


y5 = numpy.load('CRAC_critical_10min2A.npy')
y6 = numpy.load('CRAC_critical_10min2B.npy')
y7 = numpy.load('CRAC_critical_10min2C.npy')
y8 = numpy.load('CRAC_critical_10min2D.npy')

y1 = y1.flatten()
y2 = y2.flatten()
y3 = y3.flatten()
y4 = y4.flatten()

###Finding pearson and pointbiserial


Xt = numpy.load('X1_HVAC1.npy')
S = numpy.load('S1_HVAC1.npy')

Yt = numpy.squeeze(Yt)

rho = pearsonr(Xt[:, 0], Yt)
print "Pearson: ", rho

r_vector = []

for i in range(0, 7):
    r = pointbiserialr(S[:, i], Yt)
    r = numpy.absolute(r[0])
    r_vector.append(r)


print r_vector
r_vector = numpy.asarray(r_vector)
r_val = numpy.mean(r_vector)

print "PB:", r_val

choose_day =30
day_cons = 5

y_eA = y_e[choose_day*24*6:(choose_day+day_cons)*24*6]
y_1A = y1[choose_day*24*6:(choose_day+day_cons)*24*6]
y_2A = y2[choose_day*24*6:(choose_day+day_cons)*24*6]
y_3A = y3[choose_day*24*6:(choose_day+day_cons)*24*6]
y_4A = y3[choose_day*24*6:(choose_day+day_cons)*24*6]
t_array = numpy.arange(0, len(y_2A))


y_5A = y5[choose_day*24*6:(choose_day+day_cons)*24*6]
y_6A = y6[choose_day*24*6:(choose_day+day_cons)*24*6]
y_7A = y7[choose_day*24*6:(choose_day+day_cons)*24*6]
y_8A = y8[choose_day*24*6:(choose_day+day_cons)*24*6]

print y_1A.shape

#### Plotting
t_array = t_array/6
plt.plot(t_array, y_eA, 'k-', label='Ground Truth', linewidth=2.00)
plt.plot(t_array, y_1A, 'b-', label='Strategy 0', linewidth=2.00)
plt.plot(t_array, y_2A, 'r-', label='Strategy I',linewidth=2.00)
#plt.plot(t_array, y_3A, 'g-', label='Strategy II', linewidth=2.00)
#plt.plot(t_array, y_4A, 'm-', label='Strategy III', linewidth=2.00)

plt.plot(t_array, y_5A, 'b--',  linewidth=2.00)
plt.plot(t_array, y_6A, 'r--',  linewidth=2.00)
#plt.plot(t_array, y_7A, 'g--',  linewidth=2.00)
#plt.plot(t_array, y_8A, 'm--',  linewidth=2.00)


# plt.plot(t/24, y3, 'r-', label='Case II: Modular Network without schedules')
#plt.ylim([0, 1.5])
plt.xlabel('Hours')
plt.ylim([0.0, 1.8])
plt.ylabel('Hourly electric load (normalized)')
# plt.axis([0, 2200, 0, y_max])
plt.legend(loc='upper right')
plt.savefig('fig_CRAC10B.eps')
plt.show()





#####################################
##Figure 2
t_train = numpy.arange(0, len(y_t))
t_test = numpy.arange(len(y_t), len(y_t)+len(y2))
t_train = t_train/6
t_test = t_test/6
t_lim = t_train[-1]
t_max = t_test[-1]




plt.plot(t_train, y_t, 'k-', label="Ground Truth")
plt.plot(t_test, y_e, 'k-')
plt.plot(t_test, y1, 'b-', label='Strategy 0')
plt.plot(t_test, y2, 'r-', label='Strategy I')
#plt.plot(t_test, y3, 'g-', label='Strategy II')
#plt.plot(t_test, y4, 'm-', label='Strategy III')
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
plt.axis([0, t_max, 0.0, 2.0])
plt.legend(loc='upper right')
plt.savefig('fig_CRAC10A.eps')
plt.show()




################################



plt.plot(t_train, y_t, 'k-', label="Ground Truth")
plt.plot(t_test, y_e, 'k-')
plt.plot(t_test, y5, 'b--', label='Strategy 0')
plt.plot(t_test, y6, 'r-', label='Strategy I')
plt.plot(t_test, y7, 'g--', label='Strategy II')
plt.plot(t_test, y8, 'm--', label='Strategy III')
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
plt.axis([0, t_max, 0.0, 2.0])
plt.legend(loc='upper right')
plt.savefig('fig_CRAC1_10A.eps')
plt.show()


###
##Strategy I Plots:

e_1h = numpy.asarray([0.141, 0.128, 0.105, 0.0813, 0.191, 0.245])
e_st0 = numpy.asarray([0.165, 0.178, 0.162, 0.158, 0.427, 0.360])
e_st1 = numpy.asarray([0.154, 0.177, 0.163, 0.165, 0.968, 1.192])

plt.scatter(e_1h, e_st0, s=120, c='b', label='Strategy 0')
plt.scatter(e_1h, e_st1, s=150, c='r', label='Strategy I')

plt.xlabel("$e_{1,1-h}$")
plt.ylabel("$e_{1,10-min}$")
plt.legend(loc=2)
plt.savefig('fig_str1.eps')
plt.show()
