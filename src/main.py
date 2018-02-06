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


if __name__ == '__main__':
    # set random seed to 0
    np.random.seed(0)
    torch.manual_seed(0)
    # load data and make training set
    data = torch.load('traindata.pt')   # [100, 1229]
    print(data.shape) # (10, 365)
    print(data) 
    max_num=np.max(data[0])
    print("==================================================================")
    print("max : ", max_num)
    print("==================================================================")
    data[:]= data[0]/max_num   # Nomalization [0, 1]
    print(data) 
    input = Variable(torch.from_numpy(data[3:, :-1]), requires_grad=False)      #[97 , 1228]
    target = Variable(torch.from_numpy(data[3:, 1:]), requires_grad=False)      #[97 ,1228]
    test_input = Variable(torch.from_numpy(data[:3, :-1]), requires_grad=False)  #[3, 1228]
    test_target = Variable(torch.from_numpy(data[:3, 1:]), requires_grad=False)  #[3 ,1228]
    # build the model
    seq = SequenceModel.Sequence()
    seq.double()
    criterion = nn.MSELoss()
    # criterion = nn.SmoothL1Loss()
    # use LBFGS as optimizer since we can load the whole data to train
    optimizer = optim.LBFGS(seq.parameters(), lr=0.8)
    # optimizer = optim.LBFGS(seq.parameters(), lr=0.002)
    #begin to train
    for i in range(15):
        print('STEP: ', i)
        def closure():
            optimizer.zero_grad()
            out = seq(input) #[5 ,335]
            print("out", out.size())
            loss = criterion(out, target) #[5 ,335] , #[5 ,335]
            print('loss:', loss.data.numpy()[0])
            loss.backward()
            return loss
        optimizer.step(closure)
        # begin to predict
        future = 30
        print("before call seq ----------------------------------------------------------------------------------------------")
        pred = seq(test_input, future = future)  #pred: [3, 1593] # test_input : [3 , 1228] , future =1000
        print("pred size ", pred.size())
        loss = criterion(pred[:, :-future], test_target) #pred[:, :-future]: [2 x 335], # test_target : [97 , 1228]
        y = pred.data.numpy()  # [3, 1593]
        # draw the result
        plt.figure(figsize=(30,10))
        plt.title('Predict future values for time sequences (Naver) from 2013-1-1 ~ 2017-12-31 \n(Dashlines are predicted values)', fontsize=30)
        plt.xlabel('Date', fontsize=20)
        plt.ylabel('Korean won', fontsize=20)
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        
        def draw(yi, color):
            print(np.arange(input.size(1)).shape)
            print(yi[:input.size(1)].shape)     
            plt.plot( np.arange(input.size(1)) , yi[:input.size(1)], color, linewidth = 2.0)   # (1228,) (1228,)
            plt.plot( np.arange(input.size(1), input.size(1) + future), yi[input.size(1):], 'g' + ':', linewidth = 2.0)
        
        draw(max_num*y[0], 'r')    # [1x 1999]
        #draw(y[1], 'g')
        # draw(y[2], 'b')
        plt.savefig('predict%d.pdf'%i)
        plt.close()