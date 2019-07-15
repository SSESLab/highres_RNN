import numpy

def create_dataset(dataset, time_lag):
    dataX, dataY = [], []

    for i in range(len(dataset)-time_lag-1):
        a = dataset[i:(i+time_lag), 0]
        dataX.append(a)
        dataY.append(dataset[i+time_lag, 0])

    return numpy.array(dataX), numpy.array(dataY)

