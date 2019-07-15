#from __future__ import division

import scipy
from scipy import stats
import numpy
import pandas
import openpyxl
import csv
import glob
import os
import random
import math

import DataFunctions
from datetime import  datetime
import dateutil.parser as parser
from datetime import timedelta

from keras.utils import np_utils
from sklearn.preprocessing import LabelEncoder

import MathFunctions




def read_multiple_files(folder_path, col1, col2):

    allfiles = glob.glob(folder_path + "/*.csv")
    # final_data = numpy.zeros((1, col_idx2-col_idx1))
    final_data = pandas.DataFrame()
    count = 0

    for file_ in sorted(allfiles):
        df = pandas.read_csv(file_, delimiter=',')
        #print df
        for col in df.columns[1:]:
            df[col] = df[col].str.extract(r'(\d+\.*\d*)').astype(numpy.float)

        df = df.apply(pandas.to_numeric, args=('coerce',))
        data_arr = df.as_matrix(columns=df.columns[col1:col2])


        #if len(data_arr)>num:
         #   data_arr = data_arr[:-1]

        #if count==0:
            #final_data = data_arr
        #else:
            #final_data = numpy.concatenate((final_data, data_arr), axis=0)

        #count = count + 1

        final_data = final_data.append(df)

    final_data = final_data.apply(pandas.to_numeric, args=('coerce',))
    data_arr = final_data.as_matrix(columns=df.columns[col1:col2])

    return data_arr




def denstry_bldg_101(data_arr):

    #this function processes everything related to bldg data
    Q = (6.3e-5)*data_arr[:, 0]
    T1 = (data_arr[:, 1] - 32.00)*(5.00/9.00)
    T2 = (data_arr[:, 2] - 32.00) * (5.00 / 9.00)

    H = Q*(T2 - T1)

    return H








def social_beh_101(data_arr):

    #this function processes everything related to bldg data
    H = data_arr[:, 0]

    return H




def read_weather_files(folder_path):
    #folder_path = r'/home/sseslab/Documents/SLC PSB data/WBB Weather Data/WBB'
    allfiles = glob.glob(folder_path + "/*.csv")
    #final_data = numpy.zeros((1, col_idx2-col_idx1))
    final_data = pandas.DataFrame()

    for file_ in sorted(allfiles):
        df = pandas.read_csv(file_, delimiter=',', skiprows=7)
        #df = df[:-1]
        #df = df.apply(pandas.to_numeric, args=('coerce',))
        #data_arr = df.as_matrix(columns=df.columns[col_idx1:col_idx2])

        #data_arr = data_arr[0:-2, :]
        #final_data = numpy.concatenate((final_data, data_arr), axis=0)
        final_data = final_data.append(df)

    #final_data = numpy.delete(final_data, (0), axis=0) #deleting first 0 row

    final_data = final_data[:-1] #deleting last row

    return final_data


def prepare_weather_WBB(folder_path, date_start, date_end, std_inv):

    weather_file = read_weather_files(folder_path)
    weather_train = DataFunctions.read_weather_102(weather_file, date_start, date_end, 2, 7)
    weather_train = DataFunctions.fix_weather_intervals(weather_train, 5, std_inv)
    weather_train = DataFunctions.interpolate_nans(weather_train)

    return weather_train

def fix_high_points(X):

    for row in range(0, len(X)):
        if X[row] > 5*numpy.nanmean(X):
            X[row] = numpy.nanmean(X)

        if X[row] < 0:
            X[row] = 0


    return X



def give_time(date_string):
    format = '%m/%d/%y %I:%M %p'
    my_date = datetime.strptime(date_string, format)


    return my_date


def split_val_data(train_data, H_t, t1, t2):

    val_data = train_data[t1:t2+1, :]
    H_val = H_t[t1:t2+1, :]

    X_t = train_data[:t1, :]
    Y_t = H_t[:t1, :]


    return X_t, Y_t, val_data, H_val

