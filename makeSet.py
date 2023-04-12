from __future__ import print_function
import numpy as np
from args import arg_parse
from math import floor
from extraFuncs import save_file, obtainData

if __name__ == '__main__':
    # load args
    args = arg_parse()
    # params
    prediction = args.prediction
    period = args.period
    step = args.step

    symbol = "BTCUSDT"
    start_date_train = "1/1/2019"
    end_date_train = "10/31/2022"
    start_date_test = "11/1/2022"
    end_date_test = "4/10/2023"

    q1 = -1.6
    q2 = -0.7
    q3 = 0.6
    q4 = 1.7

    output_train = []
    dataset_train = []
    output_test = []
    dataset_test = []

    # Obtain data from binance
    print('Obtaining training data from Binance...')
    klines_train = obtainData(symbol, start_date_train, end_date_train)
    print('Obtaining testing data from Binance...')
    klines_test = obtainData(symbol, start_date_test, end_date_test)

    '''
    print('Starting data loop for Train')
    for i in range(floor((len(klines_train) - period - prediction) / step)):
        n = i * step
        n2 = (i * step) + period
        cutlines = klines_train[n:n2, :]
        numIn = float(klines_train[(n2), 3])
        numOut = float(klines_train[(n2 + prediction), 3])
        # outputs percentage winnings (negative if loss)
        out = ((numOut/numIn)-1)*100
        output_train.append(out)
        dataset_train.append(cutlines)

    print('Starting data loop for Test')
    for i in range(floor((len(klines_test) - period - prediction) / step)):
        n = i * step
        n2 = (i * step) + period
        cutlines = klines_test[n:n2, :]
        numIn = float(klines_test[(n2), 3])
        numOut = float(klines_test[(n2 + prediction), 3])
        # outputs percentage winnings (negative if loss)
        out = ((numOut / numIn) - 1) * 100
        output_test.append(out)
        dataset_test.append(cutlines)
    '''

    print('Starting data loop for Train')
    for i in range(floor((len(klines_train) - period - prediction) / step)):
        n = i * step
        n2 = (i * step) + period
        cutlines = klines_train[n:n2, :]
        # set time to time of the week
        cutlines = cutlines.astype(float)
        cutlines[:, 0] = cutlines[:, 0] % 604800000
        numIn = float(klines_train[n2, 4])
        numOut = float(klines_train[(n2 + prediction), 4])
        # outputs percentage winnings (negative if loss)
        out = ((numOut / numIn) - 1) * 100
        tag = 4
        if out < q1:
            tag = 0
        elif out < q2:
            tag = 1
        elif out < q3:
            tag = 2
        elif out < q4:
            tag = 3
        output_train.append(tag)
        dataset_train.append(cutlines)

    print('Starting data loop for Test')
    for i in range(floor((len(klines_test) - period - prediction) / step)):
        n = i * step
        n2 = (i * step) + period
        cutlines = klines_test[n:n2, :]
        cutlines = cutlines.astype(float)
        cutlines[:, 0] = cutlines[:, 0] % 604800000
        numIn = float(klines_test[n2, 4])
        numOut = float(klines_test[(n2 + prediction), 4])
        # outputs percentage winnings (negative if loss)
        out = ((numOut / numIn) - 1) * 100
        tag = 4
        if out < q1:
            tag = 0
        elif out < q2:
            tag = 1
        elif out < q3:
            tag = 2
        elif out < q4:
            tag = 3
        output_test.append(tag)
        dataset_test.append(cutlines)

    print('Saving Files')
    dataset_train = np.array(dataset_train).astype(float)
    output_train = np.array(output_train)
    dataset_test = np.array(dataset_test).astype(float)
    output_test = np.array(output_test)
    save_file(dataset_train, 'trainBTC1')
    save_file(output_train, 'ytrainBTC1')
    save_file(dataset_test, 'testBTC1')
    save_file(output_test, 'ytestBTC1')
