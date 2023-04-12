import websocket, json, pprint, numpy
import config
from binance.client import Client
from binance.enums import *
import os
import numpy as np
import torch.nn as nn
import torch.utils.data
import model
from args import arg_parse
from extraFuncs import obtainData, save_file, load_file, changeList, prepareData


def buyOrder(symbol, value):
    print('Buying ', symbol, 'at price: ', value)

def sellOrder(symbol, value):
    print('Selling ', symbol)

# load args
args = arg_parse()

# SET PARAMETERS
client = Client(config.API_KEY, config.API_SECRET, tld='com')
SOCKET = "wss://stream.binance.com:9443/ws/ethusdt@kline_1m"
TRADE_SYMBOL = 'ETHUSDT'
TRADE_QUANTITY = 0.05
INITIAL_MONEY = 50
period = args.period

# LOAD MODEL
model_state = os.path.join('model_bck', 'modelLS8.pth.tar')
my_model = model.Model(5).eval().cuda()
my_model.load_state_dict(torch.load(model_state))

# OBTAIN INITIAL DATA
global data_arr
data_arr = client.get_klines(symbol = TRADE_SYMBOL, interval=Client.KLINE_INTERVAL_1MINUTE, limit=period)
data_arr = np.array(data_arr).astype(np.float)
data_arr = data_arr[:, 1:6]

in_position = False
money = INITIAL_MONEY
time_passed = 0
value = 0
bought = 0


def on_open(ws):
    print('opened connection')


def on_close(ws):
    print('closed connection')


def on_message(ws, message):
    global time_passed, in_position, money, value, bought
    global data_arr

    #print('received message')
    json_message = json.loads(message)

    candle = json_message['k']
    is_candle_closed = candle['x']

    if is_candle_closed:
        newData = [candle['o'], candle['h'], candle['l'], candle['c'], candle['v']]
        data_arr = data_arr.tolist()
        data_arr.append(newData)
        data_arr = np.array(data_arr).astype(np.float)
        data_arr = data_arr[1:, :]
        if in_position:
            time_passed += 1
            print('Time since BUY: ', time_passed)
            if time_passed == 30:
                print('Sold at ', candle['c'])
                money += (float(candle['c'])/bought)*value
                sellOrder(TRADE_SYMBOL, money)
                in_position = False
                print('Current money: ', money)
        else:
            tensorData = prepareData(data_arr)
            out = my_model(tensorData, [period])
            out_label = torch.argmax(out, 1).cpu().data
            print('skrrrr')
            if out_label[0] == 4:
                value = max(money, INITIAL_MONEY)
                bought = float(candle['c'])
                money -= value
                print('Buying', value, ' at price: ', bought)
                buyOrder(TRADE_SYMBOL, value)
                time_passed = 0
                in_position = True


ws = websocket.WebSocketApp(SOCKET, on_open=on_open, on_close=on_close, on_message=on_message)
ws.run_forever()
