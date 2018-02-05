#-*- coding: utf-8 -*-
import argparse

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import SequenceModel

import sys

import pandas as pd
from pandas_datareader import data
from datetime import datetime
import fix_yahoo_finance as yf
import pickle
yf.pdr_override()

start_date = datetime(2013, 1, 1) 
end_date = datetime(2017, 12, 31) 
tickers = ['067160.KQ', '035420.KS']
#afreeca = data.get_data_yahoo(tickers[0], start_date)
# naver = data.get_data_yahoo(tickers[1], start_date)

#naver_1 = data.DataReader(tickers[1] ,'yahoo' ,datetime(2008, 1, 1) ,datetime(2008, 12, 31) )
naver = data.DataReader(tickers[1] ,'yahoo' ,start_date,end_date )


naver.to_csv('./naver.csv')
stored_data = './naver.csv'
#print(pd.read_csv('./naver.csv'))

raw_data = pd.read_csv(stored_data)

raw_data.info()


fix_data = raw_data['Open']
#print(fix_data)
fix_data.to_csv('./fixed_data.csv')

np_fix_data = fix_data.values
print(np_fix_data.shape)

np_fix_data = np_fix_data.reshape(1,-1)
print((np_fix_data.shape)[1])

L = (np_fix_data.shape)[1] # 1229
N = 100
x = np.empty((N, L), 'float64')
np_fix_data = np_fix_data.astype('float64')


"""
plt.figure(figsize=(30,10))
plt.title('Predict future values for time sequences\n(Dashlines are predicted values)', fontsize=30)
plt.xlabel('x', fontsize=20)
plt.ylabel('y', fontsize=20)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)

temp = x[0]
plt.plot(np.arange(L) , temp[:L], 'r', linewidth = 2.0)
plt.savefig('original_nomalization.pdf')
plt.close()
"""
     
x[:] = np_fix_data

torch.save(x, open('traindata.pt', 'wb'))

