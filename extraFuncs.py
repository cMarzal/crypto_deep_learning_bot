from __future__ import print_function
import torch.utils.data
import torch
import config
from binance.client import Client
import numpy as np


def prepareData(arr):
    s1 = arr[:, [0]]
    s2 = arr[:, 1:5]
    s3 = arr[:, [5]]
    s1 = np.interp(s1, (s1.min(), s1.max()), (0, +1))
    s2 = np.interp(s2, (s2.min(), s2.max()), (0, +1))
    s3 = np.interp(s3, (s3.min(), s3.max()), (0, +1))
    s2 = np.append(s2, s3, axis=1)
    s2 = np.append(s1, s2, axis=1)
    temp = torch.from_numpy(s2).type(torch.FloatTensor)
    temp = temp.unsqueeze(1)#.cuda()
    return temp


def obtainData(symbol, date1, date2):
    client = Client(config.API_KEY, config.API_SECRET)
    klines = client.get_historical_klines(symbol, Client.KLINE_INTERVAL_15MINUTE, date1, date2)
    klines = np.array(klines)
    klines = klines[:, 0:6]
    # Format of data is Time, Open, High, Low, Close, Volume
    print('Length of obtained data is: ' + str(len(klines)))
    return klines


def save_file(arr, file_name):
    np.save('data/' + file_name + '.npy', arr)  # save


def load_file(file_name):
    new_arr = np.load('data/' + file_name + '.npy')  # load
    return new_arr


def changeList(arr):
    li = []
    for i, sample in enumerate(arr):
        s1 = sample[:, [0]]
        s2 = sample[:, 1:5]
        s3 = sample[:, [5]]
        s1 = np.interp(s1, (s1.min(), s1.max()), (0, +1))
        s2 = np.interp(s2, (s2.min(), s2.max()), (0, +1))
        s3 = np.interp(s3, (s3.min(), s3.max()), (0, +1))
        s2 = np.append(s2, s3, axis=1)
        s2 = np.append(s1, s2, axis=1)
        newv = torch.from_numpy(s2).type(torch.FloatTensor)
        #newv = newv.unsqueeze_(1)
        li.append(newv)
    return li


def getWeights(arr):
    we = []
    tots = [0, 0, 0, 0, 0]
    le = len(arr)
    for num in arr:
        tots[num] += 1
    m = max(tots)
    for x in tots:
        we.append(m / x)
    we[0] = we[0] * 1.1
    we[4] = we[4] * 1.5
    return we


if __name__ == '__main__':
    trainSet = load_file('trainBTC1')
    testSet = load_file('testBTC1')
    trainSet = changeList(trainSet)
    testSet = changeList(testSet)
