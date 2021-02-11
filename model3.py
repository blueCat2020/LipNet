import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import math
import numpy as np
"""
model2: encoder + attn + decoder
we use Bi-gru as our encoder, gru as decoder, Luong attention(concat method) as our attention
It refers to paper "Effective Approaches to Attention-based Neural Machine Translation"
"""
# coding = utf-8
# author = 'xy'

"""
model2: encoder + attn + decoder
we use Bi-gru as our encoder, gru as decoder, Luong attention(concat method) as our attention
It refers to paper "Effective Approaches to Attention-based Neural Machine Translation"
"""


import torch
from torch import nn
from torch.nn import functional as f
import test_helper
import numpy as np

class Encoder(nn.Module):
    def __init__(self,hidden_size,num_layers=1, dropout=0.2):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.drop_out = dropout
        

        self.conv1 = nn.Conv3d(3, 32, (3, 5, 5), (1, 2, 2), (1, 2, 2))
        self.norm1 = nn.BatchNorm3d(32)
        self.pool1 = nn.MaxPool3d((1, 2, 2), (1, 2, 2))

        self.conv2 = nn.Conv3d(32, 64, (3, 5, 5), (1, 1, 1), (1, 2, 2))
        self.norm2 = nn.BatchNorm3d(64)
        self.pool2 = nn.MaxPool3d((1, 2, 2), (1, 2, 2))

        self.conv3 = nn.Conv3d(64, 96, (3, 3, 3), (1, 1, 1), (1, 1, 1))
        self.norm3 = nn.BatchNorm3d(96)
        self.pool3 = nn.MaxPool3d((1, 2, 2), (1, 2, 2))
        # [1,96,10,7,14]
        self.rnn = nn.LSTM(96*7*14, hidden_size,num_layers=num_layers, dropout=dropout, bidirectional=True)

    def forward(self, x, src_len):
        """
        :param x: tensor, cuda, (B, C, T, H, W)
        :param src_len: tensor, (batch_size)
        :return: outputs(seq_len, batch_size, hidden_size*2), h_t(num_layers, batch_size, hidden_size*2)
        """
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu(x)
        x = self.pool1(x)
        
        x = self.conv2(x)
        x = self.norm2(x)
        x = self.relu(x)     
        x = self.pool2(x)
        
        x = self.conv3(x)
        x = self.norm3(x)
        x = self.relu(x)
        x = self.pool3(x)

        # (B, C, T, H, W)->(T, B, C, H, W)->(T, B, C*H*W)
        x = x.permute(2, 0, 1, 3, 4).contiguous()
        x = x.view(x.size(0), x.size(1), -1)

        self.rnn.flatten_parameters()
        x = nn.utils.rnn.pack_padded_sequence(x, src_len)
        outputs, h_t = self.rnn(x, None)  
        outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs)
        h_t = torch.cat((h_t[0::2], h_t[1::2]), dim=2)
        return outputs, h_t


class Attn(nn.Module):
    def __init__(self, hidden_size):
        super(Attn, self).__init__()

        self.hidden_size = hidden_size

        self.fc1 = nn.Linear(hidden_size*2, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.v = nn.Linear(hidden_size, 1)

    def forward(self, outputs, src_len, ss):
        """
        :param outputs: h tensor, (src_seq_len, batch_size, hidden_size*2)
        :param src_len: tensor, (batch_size)
        :param ss: s tensor, (tgt_seq_len, batch_size, hidden_size)
        :return: content tensor, (batch_size, tgtq_se_len, hidden_size*2)
        """

        src_seq_len = outputs.size(0)
        tgt_seq_len = ss.size(0)
        batch_size = outputs.size(1)

        h = outputs.view(-1, self.hidden_size*2)
        wh = self.fc1(h).view(src_seq_len, batch_size, self.hidden_size)
        wh = wh.transpose(0, 1)
        wh = wh.unsqueeze(1)
        wh = wh.expand(batch_size, tgt_seq_len, src_seq_len, self.hidden_size)

        s = ss.view(-1, self.hidden_size)
        ws = self.fc2(s).view(tgt_seq_len, batch_size, self.hidden_size)
        ws = ws.transpose(0, 1)
        ws = ws.unsqueeze(2)
        ws = ws.expand(batch_size, tgt_seq_len, src_seq_len, self.hidden_size)

        hs = f.tanh(wh + ws)
        hs = hs.view(-1, self.hidden_size)
        hs = self.v(hs).view(batch_size, tgt_seq_len, src_seq_len)

        # mask
        mask = []
        for i in src_len:
            i = i.item()
            mask.append([0]*i + [1]*(src_seq_len-i))
        mask = torch.ByteTensor(mask)
        mask = mask.unsqueeze(1)
        mask = mask.expand(batch_size, tgt_seq_len, src_seq_len).cuda()
        hs.masked_fill_(mask, -float('inf'))

        hs = f.softmax(hs, dim=2)
        hs = torch.bmm(hs, outputs.transpose(0, 1))
        return hs


class Decoder(nn.Module):
    def __init__(self, hidden_size,num_layers=1, dropout=0.2):
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout

        self.rnn = nn.LSTM(hidden_size*2, hidden_size,num_layers=num_layers,bidirectional=False,dropout=0.2)

        self.attn = Attn(hidden_size)
        self.FC = nn.Linear(hidden_size,10+1)

    def forward(self, tgt, state, outputs, src_len):
        """
        :param tgt: index, (tgt_seq_len, batch_size,hidden_size*2)
        :param state: s_t, (num_layers, batch_size, hidden_size)
        :param outputs: h, (src_seq_len, batch_size, hidden_size*2)
        :param src_len: tensor, (batch_size)
        :return: results(tgt_seq_len, batch_size, vocab_size), state(num_layers, batch_size, hidden_size)
        """
       

        # teacher_forcing mode, also for testing mode
        if flag:
            ss, state = self.rnn(tgt, state)
            content = self.attn(outputs, src_len, ss).transpose(0, 1)  # (tgt_seq_len, batch_size, hidden_size*2)
          

        return result, state

class LipNet(torch.nn.Module):
    def __init__(self, dropout_rate=0.2):
        super(LipNet, self).__init__()

        self.dropout_rate = dropout_rate
        
      
        # [1,96,10,7,14]
        
        self.attn = Attn(hidden_size)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(self.dropout_rate)
        self.FC = nn.Linear(256*2, 10+1)
        self._init()

    def _init(self):

        init.kaiming_normal_(self.conv1.weight, nonlinearity='relu')
        init.constant_(self.conv1.bias, 0.0001)

        init.kaiming_normal_(self.conv2.weight, nonlinearity='relu')
        init.constant_(self.conv2.bias, 0.001)

        init.kaiming_normal_(self.conv3.weight, nonlinearity='relu')
        init.constant_(self.conv3.bias, 0.01)

        init.kaiming_normal_(self.FC.weight, nonlinearity='relu')
        init.constant_(self.FC.bias, 0.1)

        for m in (self.lstm1, self.lstm2):
            stdv = math.sqrt(2 / (96 * 7 * 14 + 256))
            for i in range(0, 256 * 3, 256):
                init.uniform_(m.weight_ih_l0[i: i + 256],
                            -math.sqrt(3) * stdv, math.sqrt(3) * stdv)
                init.orthogonal_(m.weight_hh_l0[i: i + 256])
                init.constant_(m.bias_ih_l0[i: i + 256],0)
                init.uniform_(m.weight_ih_l0_reverse[i: i + 256],
                            -math.sqrt(3) * stdv, math.sqrt(3) * stdv)
                init.orthogonal_(m.weight_hh_l0_reverse[i: i + 256])
                init.constant_(m.bias_ih_l0_reverse[i: i + 256], 0)
        

    
    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu(x)
        x = self.pool1(x)
        
        x = self.conv2(x)
        x = self.norm2(x)
        x = self.relu(x)     
        x = self.pool2(x)
        
        x = self.conv3(x)
        x = self.norm3(x)
        x = self.relu(x)
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
