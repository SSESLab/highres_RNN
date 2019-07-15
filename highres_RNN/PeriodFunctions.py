#from __future__ import division

import scipy
import numpy
import pandas
import openpyxl
import csv
import glob
import os
import random
import math

from datetime import  datetime
import dateutil.parser as parser
from datetime import timedelta

from scipy.special import comb

from keras.utils import np_utils
from sklearn.preprocessing import LabelEncoder

import MathFunctions


def set_period_array(Y, P):

    #P is in timesteps
    N = float(len(Y))
    seq_num = int(math.trunc(N/float(P)))

    Y_new = Y[0:P*seq_num, :]



    return Y_new


def reshape_array(Y, P):

    Y_new = Y.copy()
    Y_new = numpy.reshape(Y_new, (len(Y_new)/P, P, Y_new.shape[1]))

    return Y_new


def collapse_solenoid(Y, P):

    var_num = Y.shape[2]

    #Z is the collapsed circle
    Z = numpy.zeros((P, var_num))

    for q in range(0, var_num):
        for p in range(0, P):
            Z[p, q] = numpy.sum(Y[:, p, q])


    return Z


def calculate_rho(i, n, x):

    c1 = (n-1)*comb(i, (n-2), exact=False)
    a1 = numpy.power(x, i)
    a2 = numpy.power((1-x), n-2-i)

    rho = c1*numpy.multiply(a1, a2)

    return rho


def compute_site(Y):

    N = numpy.sum(Y) #N is the total number of points in a circles


    N_v = numpy.zeros((len(Y)+1, ))

    for i in range(0, len(Y)+1):
        N_v[i] = numpy.sum(Y[0:i])

    #N_v = numpy.insert(N_v, [0], axis=0)
    #now i need to attach an ID to every site
    ID_v = numpy.arange(0, N)
    loc_v = numpy.zeros((N, ))

    for i in range(0, int(N)):
        for p in range(0, len(N_v)-1):
          if i>=N_v[p] and i<N_v[p+1]:
              loc_v[i] = p+1



    return loc_v



def compute_x(loc_v, i, j, P):

    N = len(loc_v)
    print N
    x_pos = 0
    x_neg = 0

    #Trying to find the positive
    if i + j >= N:
        p_prime = i + j  - N
        pos_1 = loc_v[i]
        pos_2 = loc_v[p_prime]
        x_pos = pos_2 + P - pos_1
    else:
        pos_1 = loc_v[i+j]
        pos_2 = loc_v[i]
        x_pos = pos_2 - pos_1


    #Try to find the negative
    if i - j < 0:
        p_prime = i - j + N
        pos_1 = loc_v[i]
        pos_2 = loc_v[p_prime]
        x_neg = pos_1 + P - pos_2
    else:
        pos_1 = loc_v[i]
        pos_2 = loc_v[i-j]
        x_neg = pos_1 - pos_2

    x = x_pos + x_neg

    return x


def get_x_array(loc_v, j, P):

    x_array = numpy.zeros((len(loc_v), ))

    for p in range(0, len(loc_v)):
        x_array[p] = compute_x(loc_v, p, j, P)

    x_array = x_array/P

    return x_array