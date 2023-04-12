from __future__ import print_function
import random
import numpy as np
import torch
import os
import torch.nn as nn
import torch.utils.data
import matplotlib.pyplot as plt
import model
from args import arg_parse
from extraFuncs import obtainData, save_file, load_file


def loss_win(mod, arr, mon):
    # load args
    args = arg_parse()

    prediction = args.prediction
    period = args.period

    money = mon
    i = 0

    print('Starting simulation:')
    while i < (len(arr) - prediction - period):
        temp = arr[i:i+period, :].astype(float)
        s1 = temp[:, [0]]
        s2 = temp[:, 1:5]
        s3 = temp[:, [5]]
        s1 = np.interp(s1, (s1.min(), s1.max()), (0, +1))
        s2 = np.interp(s2, (s2.min(), s2.max()), (0, +1))
        s3 = np.interp(s3, (s3.min(), s3.max()), (0, +1))
        s2 = np.append(s2, s3, axis=1)
        s2 = np.append(s1, s2, axis=1)
        temp = torch.from_numpy(s2).type(torch.FloatTensor)
        temp = temp.unsqueeze(1).cuda()
        out = mod(temp, [period])
        out_label = torch.argmax(out, 1).cpu().data
        if out_label[0] == 4:
            buy = max(money, mon)
            trig = 0
            for c in range(1, prediction):
                if(arr[i + period, 4] * 0.9) > arr[i + period + c, 3]:
                    sell = buy * 0.97
                    trig = 1
                    i += c
                    break
                '''
                elif (arr[i + period, 3] * 1.05) < arr[i + period + c, 1]:
                    sell = buy * 1.05
                    trig = 1
                    i += c
                '''
            if trig == 0:
                sell = (arr[i + period + prediction, 4]/arr[i + period, 4])*buy
                i += prediction
            money -= buy#*1.00055
            money += sell
            # print('Buying at, ', buy, '. Selling at: ', sell)
            print('New money ammount: ', money)
        else:
            i += 1
    return money


if __name__ == '__main__':
    # load args
    args = arg_parse()
    prediction = args.prediction
    period = args.period
    step = args.step
    prtg = args.percentage_buy
    money_spent = 300
    #params

    model_state = os.path.join('mods', 'RNN_model.pth.tar')
    my_model = model.Model(6).eval().cuda()
    my_model.load_state_dict(torch.load(model_state))

    print('Obtaining data')
    if 1:
        symbol = "BTCUSDT"
        start_date = "10/8/2022"
        end_date = "4/8/2023"

        klines = obtainData(symbol, start_date, end_date)
        klines = np.array(klines).astype(float)
        save_file(klines, 'validationBTC1')
    else:
        klines = load_file('validationBTC1')
    totMon = (klines[-1, 4]/klines[0, 4])*money_spent
    print('money if hold: ', totMon)
    print('starting calculations')
    money = loss_win(my_model, klines, money_spent)
    print('Total money won/loss: ', money)

