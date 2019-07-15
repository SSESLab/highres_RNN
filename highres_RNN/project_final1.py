import numpy

import PlotFunctions

H_t = numpy.load('H_t10.npy')
H_e = numpy.load('H_e10.npy')
Y_lstm2 = numpy.load('Y_ST2B.npy')
t_train = numpy.arange(0, len(H_t))
t_test = numpy.arange(len(H_t), len(H_t)+len(Y_lstm2))
t_array = numpy.arange(0, len(Y_lstm2))



PlotFunctions.Plot_triple(t_train, H_t, t_test, Y_lstm2, t_test, H_e, 'Training Data', 'Approach I predictions', 'Test Data (actual)', 'k-', 'r-', 'b-', "fig_sample.eps")