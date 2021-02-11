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
    def __init__(self, input_size, hidden_size, embedding, num_layers=1, dropout=0.2):
        super(Encoder, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.embedding = embedding
        self.num_layers = num_layers
        self.drop_out = dropout

        self.rnn = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            bidirectional=True
        )

    def forward(self, src, src_len):
        """
        :param src: tensor, cuda, (seq_len, batch_size)
        :param src_len: tensor, (batch_size)
        :return: outputs(seq_len, batch_size, hidden_size*2), h_t(num_layers, batch_size, hidden_size*2)
        """

        src = self.embedding(src)
        src = nn.utils.rnn.pack_padded_sequence(src, src_len)
        outputs, h_t = self.rnn(src, None)
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
    def __init__(self, input_size, hidden_size, embedding, num_layers=1, dropout=0.2):
        super(Decoder, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.embedding = embedding
        self.num_layers = num_layers
        self.dropout = dropout

        self.rnn = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            bidirectional=False
        )

        self.attn = Attn(hidden_size)
        self.fc = nn.Linear(hidden_size*2, hidden_size)
        self.fh = nn.Linear(hidden_size, hidden_size)
        self.ws = nn.Linear(hidden_size, embedding.num_embeddings)

    def forward(self, tgt, state, outputs, src_len, teacher_forcing):
        """
        :param tgt: index, (tgt_seq_len, batch_size)
        :param state: s_t, (num_layers, batch_size, hidden_size)
        :param outputs: h, (src_seq_len, batch_size, hidden_size*2)
        :param src_len: tensor, (batch_size)
        :param teacher_forcing:one train skill,加速训练
        :return: results(tgt_seq_len, batch_size, vocab_size), state(num_layers, batch_size, hidden_size)
        """
        flag = np.random.random() < teacher_forcing

        # teacher_forcing mode, also for testing mode
        if flag:
            embedded = self.embedding(tgt)#三维
            ss, state = self.rnn(embedded, state)
            content = self.attn(outputs, src_len, ss).transpose(0, 1)  # (tgt_seq_len, batch_size, hidden_size*2)
            content = content.contiguous().view(-1, self.hidden_size*2)
            ss = ss.view(-1, self.hidden_size)
            result = f.tanh(self.fc(content) + self.fh(ss))
            result = self.ws(result).view(tgt.size(0), tgt.size(1), -1)

        # generation mode
        else:
            result = []
            embedded = self.embedding(tgt[0: 1])
            for i in range(tgt.size(0)):
                ss, state = self.rnn(embedded, state)
                content = self.attn(outputs, src_len, ss).transpose(0, 1)
                content = content.view(-1, self.hidden_size*2)
                ss = ss.view(-1, self.hidden_size)
                r = f.tanh(self.fc(content) + self.fh(ss))
                r = self.ws(r).view(tgt.size(1), -1)
                result.append(r)

                _, topi = torch.topk(r, k=1, dim=1)
                embedded = self.embedding(topi.transpose(0, 1))
            result = torch.stack(result)

        return result, state


class Seq2seq(nn.Module):
    def __init__(self, input_size, hidden_size, embedding, num_layers=1, dropout=0.2):
        super(Seq2seq, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.embedding = embedding
        self.num_layers = num_layers
        self.dropout = dropout

        self.encoder = Encoder(
            input_size=input_size,
            hidden_size=hidden_size,
            embedding=embedding,
            num_layers=num_layers,
            dropout=dropout
        )

        self.decoder = Decoder(
            input_size=input_size,
            hidden_size=hidden_size,
            embedding=embedding,
            num_layers=num_layers,
            dropout=dropout
        )

        self.fc = nn.Linear(hidden_size*2, hidden_size)

    def forward(self, src, src_len, tgt, teacher_forcing):
        """
        :param src: index, (src_seq_len, batch_size)
        :param src_len: tensor, (batch_size)
        :param tgt: index, (tgt_seq_len, batch_size)
        :param teacher_forcing:
        :return: outputs(tgt_seq_len, batch_size, vocab_size), state(num_layers, batch_size, hidden_size)
        """
        # encode
        outputs, h_t = self.encoder(src, src_len)
        state = h_t.view(-1, self.hidden_size * 2)
        state = f.tanh(self.fc(state)).view(self.num_layers, -1, self.hidden_size)

        # decode
        result, state = self.decoder(tgt, state, outputs, src_len, teacher_forcing)

        return result, state

    def gen(self, index, num_beams, max_len):
        """
        test mode
        :param index: a sample about src, tensor
        :param num_beams:
        :param max_len: max length of result
        :return: result, list
        """
        src = index.unsqueeze(1)#增维操作
        src_len = torch.LongTensor([src.size(0)])

        # encode
        outputs, h_t = self.encoder(src, src_len)
        state = h_t.view(-1, self.hidden_size * 2)
        state = f.tanh(self.fc(state)).view(self.num_layers, -1, self.hidden_size)

        # decoder
        result = test_helper.beam_search(self.decoder, num_beams, max_len, state, outputs, src_len)
        if result[-1] == 2:
            return result[1: -1]
        else:
            return result[1:]