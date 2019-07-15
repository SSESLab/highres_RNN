import numpy
import matplotlib.pyplot as plt
import plotly.plotly as py
import matplotlib.pyplot as plt
import scipy.fftpack

import DataFunctions

# Number of samplepoints

def get_time_charasteristics(Y, tau):
    # tau is the charestertistic time-scale:

    T = len(Y)  # length of the entire
    seq_num = int(numpy.trunc(T/tau))  # number of sequences to be repeated

    min_vector = numpy.zeros(seq_num, )
    max_vector = numpy.zeros(seq_num, )
    phase_vector = numpy.zeros(seq_num, )
    mean_vector = numpy.zeros(seq_num, )

    Y_seq = numpy.zeros((tau, seq_num))

    for i in range(0, seq_num):
        min_vector[i] = numpy.amin(Y[i*tau: (i+1)*tau])
        max_vector[i] = numpy.amax(Y[i*tau: (i+1)*tau])
        phase_vector[i] = numpy.argmax(Y[i*tau: (i+1)*tau])
        mean_vector[i] = numpy.mean(Y[i*tau: (i+1)*tau])

        Y_seq[:, i] = Y[i*tau: (i+1)*tau]

    return min_vector, max_vector, phase_vector, mean_vector, Y_seq


def Plot_fft_101(t, y, sampling_rate):
    n = len(y)
    k = numpy.arange(n)
    T = n/sampling_rate
    print "T: "
    print T

    k = k.astype(float)
    frq = k / float(T)  # two sides frequency range
    frq = frq[range(n / 2)]  # one side frequency range

    frq = frq.astype(float)
    Y = numpy.fft.fft(y) / n  # fft computing and normalization
    Y = Y[range(n / 2)]

    plt.plot(frq, abs(Y), 'r')
    plt.show()

    return frq, abs(Y)


def timescale_101(X, data_string, ref_date, timescale):
    #from __future__ import  division
    #This function works on an hourly basis
    a_seconds = DataFunctions.give_time(data_string) - DataFunctions.give_time(ref_date)
    a_hour = int(a_seconds.total_seconds()/3600)

    idx = a_hour
    row_max = X.shape[0]
    col_max = len(timescale)

    time_features = numpy.zeros((row_max, col_max))

    for i in range(0, row_max):
        for j in range(0, col_max):
            temp_val = (idx + i) % timescale[j]
            temp_val = temp_val.astype(float)
            time_features[i, j] = temp_val/timescale[j]

    return time_features









