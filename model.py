import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import math
import numpy as np


class LipNet(torch.nn.Module):
    def __init__(self, attention_size=1, dropout_rate=0.5):
        super(LipNet, self).__init__()

        self.dropout_rate = dropout_rate
        self.attention_size = attention_size

        self.conv1 = nn.Conv3d(3, 32, (3, 5, 5), (1, 2, 2), (1, 2, 2))
        #self.norm1 = nn.BatchNorm3d(32)
        self.pool1 = nn.MaxPool3d((1, 2, 2), (1, 2, 2))

        self.conv2 = nn.Conv3d(32, 64, (3, 5, 5), (1, 1, 1), (1, 2, 2))
        #self.norm2 = nn.BatchNorm3d(64)
        self.pool2 = nn.MaxPool3d((1, 2, 2), (1, 2, 2))

        self.conv3 = nn.Conv3d(64, 96, (3, 3, 3), (1, 1, 1), (1, 1, 1))
        #self.norm3 = nn.BatchNorm3d(96)
        self.pool3 = nn.MaxPool3d((1, 2, 2), (1, 2, 2))
        '''
        if self.use_cuda:
            self.w_omega = Variable(torch.zeros(256 * 2, self.attention_size).cuda())
            self.u_omega = Variable(torch.zeros(self.attention_size).cuda())
        else:
            self.w_omega = Variable(torch.zeros(256 * 2, self.attention_size))
            self.u_omega = Variable(torch.zeros(self.attention_size))
        '''
        # [1,96,10,7,14]
        self.lstm1 = nn.LSTM(96*7*14, 256,num_layers=1, bidirectional=True)
        self.lstm2 = nn.LSTM(256*2, 256,num_layers=1,bidirectional=True)

        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(self.dropout_rate)
        self.dropout3d = nn.Dropout3d(self.dropout_rate)
        self.FC = nn.Linear(256*2, 10+1)
        self._init()

    def _init(self):

        init.kaiming_normal_(self.conv1.weight, nonlinearity='relu')
        init.constant_(self.conv1.bias, 0)

        init.kaiming_normal_(self.conv2.weight, nonlinearity='relu')
        init.constant_(self.conv2.bias, 0)

        init.kaiming_normal_(self.conv3.weight, nonlinearity='relu')
        init.constant_(self.conv3.bias, 0)

        init.kaiming_normal_(self.FC.weight, nonlinearity='sigmoid')
        init.constant_(self.FC.bias, 0)

        for m in (self.lstm1, self.lstm2):
            stdv = math.sqrt(2 / (96 * 3 * 6 + 256))
            for i in range(0, 256 * 3, 256):
                init.uniform_(m.weight_ih_l0[i: i + 256],
                            -math.sqrt(3) * stdv, math.sqrt(3) * stdv)
                init.orthogonal_(m.weight_hh_l0[i: i + 256])
                init.constant_(m.bias_ih_l0[i: i + 256],0)
                init.uniform_(m.weight_ih_l0_reverse[i: i + 256],
                            -math.sqrt(3) * stdv, math.sqrt(3) * stdv)
                init.orthogonal_(m.weight_hh_l0_reverse[i: i + 256])
                init.constant_(m.bias_ih_l0_reverse[i: i + 256], 0)
        

    def attention_net(self, lstm_output): 
        # print(lstm_output.size()) = (squence_length, batch_size, hidden_size*layer_size)
        sequence_length = lstm_output.size(0)
        output_reshape = torch.Tensor.reshape(lstm_output, [-1, 256*2])
        # print(output_reshape.size()) = (squence_length * batch_size, hidden_size*layer_size)
        # torch.mm，向量乘积（（n*M）*（M*P）,得到(n,p)）
        attn_tanh = torch.tanh(torch.mm(output_reshape, self.w_omega))
        # print(attn_tanh.size()) = (squence_length * batch_size, attention_size)

        attn_hidden_layer = torch.mm(
            attn_tanh, torch.Tensor.reshape(self.u_omega, [-1, 1]))
        # print(attn_hidden_layer.size()) = (squence_length * batch_size, 1)

        exps = torch.Tensor.reshape(
            torch.exp(attn_hidden_layer), [-1, self.sequence_length])
        # print(exps.size()) = (batch_size, squence_length)

        alphas = exps / torch.Tensor.reshape(torch.sum(exps, 1), [-1, 1])
        # print(alphas.size()) = (batch_size, squence_length)

        alphas_reshape = torch.Tensor.reshape(
            alphas, [-1, self.sequence_length, 1])
        # print(alphas_reshape.size()) = (batch_size, squence_length, 1)

        state = lstm_output.permute(1, 0, 2)
        # print(state.size()) = (batch_size, squence_length, hidden_size*layer_size)

        attn_output = torch.sum(state * alphas_reshape, 1)
        # print(attn_output.size()) = (batch_size, hidden_size*layer_size)

        return attn_output

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.dropout3d(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.relu(x)
        x = self.dropout3d(x)
        x = self.pool2(x)

        x = self.conv3(x)
        x = self.relu(x)
        x = self.dropout3d(x)
        x = self.pool3(x)

        # (B, C, T, H, W)->(T, B, C, H, W)->(T, B, C*H*W)
        x = x.permute(2, 0, 1, 3, 4).contiguous()
        x = x.view(x.size(0), x.size(1), -1)

        self.lstm1.flatten_parameters()
        self.lstm2.flatten_parameters()
        x, h = self.lstm1(x)
        x = self.dropout(x)
        x, h = self.lstm2(x)
        x = self.dropout(x)
        #attn_output = self.attention_net(x)

        x = self.FC(x)
        x = x.permute(1, 0, 2).contiguous()

        return x
