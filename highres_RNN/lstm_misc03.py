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
y_t = numpy.load('HVAC_critical_train.npy')
y_val = numpy.load('HVAC_critical_val.npy')
y_t = numpy.squeeze(y_t)


#y_t = numpy.concatenate((y_t, y_val), axis=0)

y_e = numpy.load('HVAC_critical_test.npy')
y1 = numpy.load('HVAC_critical_A.npy')
y2 = numpy.load('HVAC_critical_B.npy')
y2 = numpy.reshape(y2, (y2.shape[0]*24, 1))
y3 = numpy.load('HVAC_critical_MLP.npy')


choose_day = 30
day_cons = 5

y_eA = y_e[choose_day*24:(choose_day+day_cons)*24]
y_1A = y1[choose_day*24:(choose_day+day_cons)*24]
y_2A = y2[choose_day*24:(choose_day+day_cons)*24]
y_3A = y3[choose_day*24:(choose_day+day_cons)*24]
t_array = numpy.arange(0, len(y_2A))

print "Array Shapes"
print y_t.shape
print y_val.shape
print y_e.shape

print y1.shape
print y2.shape
print y3.shape

#sim_score = (MathFunctions.rms_flat(y_e - y_t[0:len(y_e)]))/(numpy.amax(y_t) - numpy.amin(y_t))
#print "The Similarity Score is: ", sim_score

#### Plotting
#plt.plot(t_array, y_1A, 'b-.', label='Model A Predictions', linewidth=5.00)

plt.plot(t_array, y_2A, 'r:', label='Model B Predictions',linewidth=5.00)
plt.plot(t_array, y_3A, 'g--', label='MLP Predictions', linewidth=3.00)
plt.plot(t_array, y_eA, 'k-', label='Ground Truth', linewidth=3.00)
# plt.plot(t/24, y3, 'r-', label='Case II: Modular Network without schedules')
#plt.ylim([0, 1.5])
plt.xlabel('Hours')
plt.ylim([0.0, 0.9])
plt.ylabel('Hourly electric load (normalized)')
# plt.axis([0, 2200, 0, y_max])
plt.legend(loc='upper right')
#plt.savefig('fig_res1C.eps')
plt.show()


#################################
###Figure 2
t_train = numpy.arange(0, len(y_t))
t_test = numpy.arange(len(y_t), len(y_t)+len(y2))
t_lim = t_train[-1]
t_max = t_test[-1]

plt.plot(t_train, y_t, 'k-', label="Ground Truth")
plt.plot(t_test, y_e, 'k-')
plt.plot(t_test, y1, 'b-', label='Model A Predictions')
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
plt.savefig('fig_CRAC2A.eps')
plt.show()



#####-------------------------------------####-----------------------
n = numpy.array([1,3 ,5, 10, 15, 20, 25, 30])
rms_array = numpy.array([1.966, 5.041, 7.326, 14.89, 19.94, 26.74, 32.24, 42.05])

e_rnn = numpy.array([45.3, 30.0, 30.0, 22.5, 21.9, 21.4, 21.1, 20.8])
e_mlp = numpy.array([46.1, 30.6, 30.1, 22.7, 18.4, 17.4, 17.8, 16.6])


plt.plot(rms_array, e_rnn, 'ro', label='Model B', markersize=8)
plt.plot(rms_array, e_mlp, 'ks', label='MLP model', markersize=8)
plt.xlabel('RMS average of aggregated electricity consumption, kWh')
plt.ylabel('$e_2 $(%)')
plt.legend(loc='upper right')
plt.savefig('fig_analysis1a.eps')
plt.show()

##########################################
s_1 = numpy.array([0.363, 0.171, 0.0962, 0.1007, 0.3947, 0.397])
e_21 = numpy.array([14.3, 12.8, 10.5, 8.13, 19.1, 24.5])
e_mlp1 = numpy.array([63.2, 23.2, 15.9, 9.38, 21.0, 22.5])
s_2 = numpy.array([0.155, 0.120, 0.1107, 0.1026, 0.100,  0.0978, 0.0976, 0.10023])

#plt.plot(s_2, e_rnn, 'ro', label='Deep RNN model', markersize=8)
#plt.plot(s_2, e_rnn, 'ro', label='Deep RNN model', markersize=8)
fig, ax = plt.subplots()
#ax.scatter(s_1, e_21, 'rs', label='Deep RNN model', markersize=8)
#ax.scatter(s_1, e_mlp1, 'ks', label='Deep RNN model', markersize=8)
plt.scatter(s_1, e_21, color='r', s=121, marker='^', alpha=.9, label='Model B (PSB)')
plt.scatter(s_1, e_mlp1, color='g', s=121, marker='^', alpha=.9, label='MLP (PSB)')
plt.scatter(s_2, e_rnn, color='r', s=121, marker='o', alpha=.9, label='Model B (Aggregate Residential)')
plt.scatter(s_2, e_mlp, color='g', s=121, marker='o', alpha=.9, label='MLP (Aggregate Residential)')
plt.ylim([0, 100])
plt.xlabel('s-value')
plt.ylabel('$e_2 $(%)')
plt.legend(loc='upper right')
plt.savefig('fig_sim1a.eps')
plt.show()


####################
file_name = '~/PycharmProjects/ElectricLoad_v1/dataport_data/dataport-metadata.csv'
fixed_df = pd.read_csv(file_name)
data_id = fixed_df['dataid']

print fixed_df
print data_id


##-----------------
#10-min predictions
#delta = 0.03571429``````````````````````````````````````````````
#delta = 0.11111111
#delta = 0.14285714
#delta = 0.05
#delta = 0.05
delta = 0.5

c1 = '0.75'
y_t = numpy.load('Ht2_CRAC1.npy')
y_train = y_t.copy()
#numpy.savetxt('CONV2.csv', y_t, delimiter=",")
y_val = numpy.load('Hv2_CRAC1.npy')
y_t = numpy.concatenate((y_t, y_val), axis=0)
y_e = numpy.load('He2_CRAC1.npy')
y_lstm = numpy.load('CRAC_critical_10min2A.npy')

deriv_Ht2 = numpy.diff(y_t, axis=0)
Delta = MathFunctions.rms_flat(deriv_Ht2)/MathFunctions.rms_flat(y_t)

print "Delta: ", Delta


X_t = numpy.load('X1_HVAC1.npy')
S_t = numpy.load('S1_HVAC1.npy')



T_db = X_t[:, 0]
Sch = S_t[:, 1:8]

y_train = numpy.squeeze(y_train)


print T_db.shape
print y_train.shape
print Sch
rho_T = pearsonr(T_db, y_train)

r_vector = []


for i in range(0, 7):
    print Sch.shape
    print y_train.shape
    r, r2 = pointbiserialr(Sch[:, i], y_train)

    r_vector.append(r)


print "rho_T: "
print rho_T

r_vector = numpy.absolute(numpy.asarray(r_vector))
r_mean = numpy.mean(r_vector)





print r_mean
print "X-values"
print X_t
#print S_t

#y0 = numpy.load('HVAC_normal_10min0.npy')
y6 = numpy.load('HVAC_critical_10min2F.npy')

yI = y6[0, :, :, :]
yI = DataFunctions.fix_bindata(yI)
y6 = DataFunctions.make_realdata(yI, delta)



y0 = numpy.load('CRAC_critical_10min0.npy')

#y0 = y0[0, :, :, :]


y1 = numpy.load('CRAC_critical_10min2A.npy')

print y2.shape

y2 = numpy.load('CRAC1_1h.npy')
#y2 = numpy.load('HVAC_critical_10min2B.npy')
y3 = numpy.load('CRAC_critical_10min2C.npy')
y4 = numpy.load('CRAC_critical_10min2D.npy')
#y5 = numpy.load('CRAC_critical_10min2E.npy')
y_mlp = numpy.load('CRAC_critical_10minMLP.npy')

#y0 = y0[:, None]
#y1 = numpy.squeeze(y1)
print y1.shape
y0 = y0.flatten()
y1 = y1.flatten()
y2 = y2.flatten()
y3 = y3.flatten()
y4 = y4.flatten()
y6 = y0.copy()
print y6.shape


y0 = y0[:, None]
y1 = y1[:, None]
y2 = y2[:, None]
y3 = y3[:, None]
y4 = y4[:, None]
#y5 = y5[:, None]
y6 = y6[:, None]

T = 1*6

print "Model 0:"

e_deep01 = (MathFunctions.rms_flat(y0- y_e))/(MathFunctions.rms_flat(y_e))
e_deep02 = (MathFunctions.rms_flat(y0- y_e))/(MathFunctions.rms_flat(y_t))
rho_0, a0 = pearsonr(y0, y_e)
print e_deep01, e_deep02, rho_0


print "Strategy I:"

e_deepA1 = (MathFunctions.rms_flat(y1- y_e))/(MathFunctions.rms_flat(y_e))
e_deepA2 = (MathFunctions.rms_flat(y1- y_e))/(MathFunctions.rms_flat(y_t))
rho_A, a2 = pearsonr(y1, y_e)
epsilon_1, epsilon_2 = DataFunctions.compute_peak_metric(y1, y_e, T)

print e_deepA1, e_deepA2, rho_A, epsilon_1, epsilon_2

print y2.shape
print "Strategy II: "
e_deepB1 = (MathFunctions.rms_flat(y2- y_e))/(MathFunctions.rms_flat(y_e))
e_deepB2 = (MathFunctions.rms_flat(y2- y_e))/(MathFunctions.rms_flat(y_t))
epsilon_1, epsilon_2 = DataFunctions.compute_peak_metric(y2, y_e, T)
rho_B, a2 = pearsonr(y2, y_e)
print e_deepB1, e_deepB2, rho_B, epsilon_1, epsilon_2

print "Strategy III: "
e_deepC1 = (MathFunctions.rms_flat(y3- y_e))/(MathFunctions.rms_flat(y_e))
e_deepC2 = (MathFunctions.rms_flat(y3- y_e))/(MathFunctions.rms_flat(y_t))
rho_C, a2 = pearsonr(y3, y_e)
epsilon_1, epsilon_2 = DataFunctions.compute_peak_metric(y3, y_e, T)
print e_deepC1, e_deepC2, rho_C, epsilon_1, epsilon_2


print "Strategy IV: "
e_deepD1 = (MathFunctions.rms_flat(y4- y_e))/(MathFunctions.rms_flat(y_e))
e_deepD2 = (MathFunctions.rms_flat(y4- y_e))/(MathFunctions.rms_flat(y_t))
rho_D, a2 = pearsonr(y4, y_e)
epsilon_1, epsilon_2 = DataFunctions.compute_peak_metric(y4, y_e, T)
print e_deepD1, e_deepD2, rho_D, epsilon_1, epsilon_2

print "Strategy V: "
e_deepE1 = (MathFunctions.rms_flat(y6- y_e))/(MathFunctions.rms_flat(y_e))
e_deepE2 = (MathFunctions.rms_flat(y6- y_e))/(MathFunctions.rms_flat(y_t))
rho_E, a2 = pearsonr(y6, y_e)
epsilon_1, epsilon_2 = DataFunctions.compute_peak_metric(y6, y_e, T)
print e_deepE1, e_deepE2, rho_E, epsilon_1, epsilon_2


print "MLP: "
e_nn = (MathFunctions.rms_flat(y_mlp - y_e))/(MathFunctions.rms_flat(y_e))
e_nn2 = (MathFunctions.rms_flat(y_mlp - y_e))/(MathFunctions.rms_flat(y_t))
rho_nn, a2 = pearsonr(y_mlp, y_e)
epsilon_1, epsilon_2 = DataFunctions.compute_peak_metric(y_mlp, y_e, T)
print e_nn, e_nn2, rho_nn, epsilon_1, epsilon_2


#### Plotting
t_train = numpy.arange(0, len(y_t), dtype=numpy.float)/6
t_test = numpy.arange(len(y_t), len(y_t)+len(y_lstm), dtype=numpy.float)/6
t_lim = t_train[-1]
t_max = t_test[-1]





#####################

##FIGURES: 1-h MODEL
fig, ax = plt.subplots()


plt.plot(t_train, y_t, 'k-', label="Ground Truth")

print "Model 0:"

e_deep01 = (MathFunctions.rms_flat(y0- y_e))/(MathFunctions.rms_flat(y_e))
e_deep02 = (MathFunctions.rms_flat(y0- y_e))/(MathFunctions.rms_flat(y_t))
rho_0, a0 = pearsonr(y0, y_e)
print e_deep01, e_deep02, rho_0

plt.plot(t_test, y_e, 'k-')
#plt.plot(t_test, y0, c1, label='Strategy 0')
plt.plot(t_test, y0, 'r-', label='10-minute Model')
#plt.plot(t_test, y_mlp, 'y-', label='MLP')
plt.plot(t_test, y2, 'b-', label='1-hour Model')
plt.plot(t_test, y_mlp, 'g-', label='MLP Model')
plt.plot((t_lim, t_lim), (0, 2.5), 'k--')


plt.xlabel('Hours')
plt.ylabel('Hourly electric load (normalized)')
plt.axis([0, t_max, 0, 2.5])

# Adding Annotations
plt.annotate('', xy=(t_lim, 1.2), xycoords='data', xytext=(0, 1.2), textcoords='data', arrowprops={'arrowstyle': '<->'})
plt.annotate('Training Phase', xy=(int(t_lim / 2), 1.0), xycoords='data', xytext=(4000, 1.15), textcoords='data')
plt.annotate('', xy=(t_max, 1.2), xycoords='data', xytext=(t_lim, 1.2), textcoords='data',
             arrowprops={'arrowstyle': '<->'})
plt.annotate('Test Phase', xy=(int((t_max - t_lim) / 2), 1.0), xycoords='data', xytext=(9000, 1.15), textcoords='data')

plt.legend(loc='upper right')
plt.savefig('CRAC1_1h.eps')
plt.show()




################



plt.plot(t_train, y_t, 'k-', label="Actual Data")
plt.plot(t_test, y_e, 'k-')
#plt.plot(t_test, y0, c1, label='Strategy 0')
plt.plot(t_test, y1, 'r-', label='Strategy I')
#plt.plot(t_test, y_mlp, 'y-', label='MLP')
plt.plot(t_test, y2, 'g-', label='Strategy II')
#plt.plot(t_test, y3, 'g-', label='Strategy III')
plt.plot(t_test, y4, 'b-', label='Strategy IV')
#plt.plot(t_test, y6, 'c-', label='Strategy V')
plt.plot(t_test, y_mlp, 'y-', label='MLP')
plt.plot((t_lim, t_lim), (0, 1.8), 'k--')
plt.ylim([0, 1.5])
# plt.plot(t/24, y3, 'r-', label='Case II: Modular Network without schedules')

# Adding Annotations
plt.annotate('', xy=(t_lim, 1.2), xycoords='data', xytext=(0, 1.2), textcoords='data', arrowprops={'arrowstyle': '<->'})
plt.annotate('Training Phase', xy=(int(t_lim / 2), 1.0), xycoords='data', xytext=(4000, 1.15), textcoords='data')
plt.annotate('', xy=(t_max, 1.2), xycoords='data', xytext=(t_lim, 1.2), textcoords='data',
             arrowprops={'arrowstyle': '<->'})
plt.annotate('Test Phase', xy=(int((t_max - t_lim) / 2), 1.0), xycoords='data', xytext=(9000, 1.15), textcoords='data')

plt.xlabel('Hours')
plt.ylabel('Hourly electric load (normalized)')
plt.axis([0, t_max, 0, 2.25])
plt.legend(loc='upper right')
plt.savefig('HVAC2_10minA.eps')
plt.show()






choose_day = 30
day_cons = 3

y_eA = y_e[choose_day*6*24:(choose_day+day_cons)*6*24]
y_0A = y0[choose_day*6*24:(choose_day+day_cons)*6*24]
y_1A = y1[choose_day*6*24:(choose_day+day_cons)*6*24]
y_2A = y2[choose_day*6*24:(choose_day+day_cons)*6*24]
y_3A = y3[choose_day*6*24:(choose_day+day_cons)*6*24]
y_4A = y4[choose_day*6*24:(choose_day+day_cons)*6*24]
#y_5A = y5[choose_day*6*24:(choose_day+day_cons)*6*24]
y_6A = y6[choose_day*6*24:(choose_day+day_cons)*6*24]
y_mlpA = y_mlp[choose_day*6*24:(choose_day+day_cons)*6*24]
t_array = numpy.arange(0, len(y_1A), dtype=numpy.float)/6

#print "Array Shapes"
#print y2.shape
#print t_array.shape
#print y_1A.shape

#sim_score = (MathFunctions.rms_flat(y_e - y_t[0:len(y_e)]))/(numpy.amax(y_t) - numpy.amin(y_t))
#print "The Similarity Score is: ", sim_score

#### Plotting
#plt.plot(t_array, y_0A, c1, label='Model 0', linewidth=1.00)
plt.plot(t_array, y_1A, 'r-', label='Strategy I', linewidth=1.00)
plt.plot(t_array, y_2A, 'g-', label='Strategy II',linewidth=1.00)
#plt.plot(t_array, y_3A, 'g--', label='Strategy III', linewidth=1.00)
plt.plot(t_array, y_4A, 'b-', label='Strategy IV', linewidth=1.00)
#plt.plot(t_array, y_6A, 'c--', label='Strategy V', linewidth=1.00)
plt.plot(t_array, y_mlpA, 'y-', label='MLP', linewidth=1.00)
plt.plot(t_array, y_eA, 'k-', label='Test Data', linewidth=1.00)
# plt.plot(t/24, y3, 'r-', label='Case II: Modular Network without schedules')
#plt.ylim([0, 1.2])
plt.xlabel('Hours')
plt.ylim([0., 1.2])
plt.ylabel('Hourly electric load (normalized)')
# plt.axis([0, 2200, 0, y_max])
plt.legend(loc='upper right')
plt.savefig('HVAC2_10minB.eps')
#plt.savefig('fig_conv2_102min.eps')
plt.show()





##Loading for t-tests
y_e = numpy.load('He2_CONV2.npy')
y0 = numpy.load('CONV_normal_10min0.npy')
y4 = numpy.load('CONV_normal_10min2A.npy')
y0 = y0.flatten()
y4 = y4.flatten()
y0 = y0[:, None]
y4 = y4[:, None]
print y0.shape


####t-test
print "t-test"
print "Strategy II: "
t1, p1, m1 = DataFunctions.compute_paired_ttest(y_e, y0, y4)
t2, p2, m2 = DataFunctions.compute_paired_ttest2(y_e, y0, y4, 6)

print t1, p1, m1
print t2, p2, m2

print t_train
print t_test.shape
print "yI shape"
print yI.shape
#print y0.shape
color = '0.90'


####Comparison of validation errors
#10-min predictions
#delta = 0.03571429
#delta = 0.11111111
delta = 0.05
c1 = '0.75'
y_t = numpy.load('Ht2_CRAC1.npy')
#numpy.savetxt('CONV2.csv', y_t, delimiter=",")
y_val = numpy.load('Hv2_CRAC1.npy')

y0 = numpy.load('CRAC1F_val0.npy')
y0 = y0[0, :, :, :]
y0 = y0.flatten()
y1 = numpy.load('CRAC1_val.npy')

y0 = y0[:, None]
y1 = y1[:, None]




print y0.shape
print y_val.shape

print "Validation Errors: "
print "Strategy 0:"

e_deepA1 = (MathFunctions.rms_flat(y0- y_val))/(MathFunctions.rms_flat(y_val))
e_deepA2 = (MathFunctions.rms_flat(y0- y_val))/(MathFunctions.rms_flat(y_t))
rho_A, a2 = pearsonr(y0, y_val)
epsilon_1, epsilon_2 = DataFunctions.compute_peak_metric(y0, y_val, T)

print e_deepA1, e_deepA2, rho_A,
print "epsilon_1", epsilon_1
print "epsilon_2", epsilon_2

print "Validation Errors: "
print "Strategy I:"

e_deepA1 = (MathFunctions.rms_flat(y1- y_val))/(MathFunctions.rms_flat(y_val))
e_deepA2 = (MathFunctions.rms_flat(y1- y_val))/(MathFunctions.rms_flat(y_t))
rho_A, a2 = pearsonr(y1, y_val)
epsilon_1, epsilon_2 = DataFunctions.compute_peak_metric(y1, y_val, T)

print e_deepA1, e_deepA2, rho_A,
print "epsilon_1", epsilon_1
print "epsilon_2", epsilon_2