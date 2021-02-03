import torch
import torch.nn as nn
import torch.nn.functional as F
torch.manual_seed(1)  

class BiLSTM_ATT(nn.Module):
    def __init__(self,input_size,output_size,config,pre_embedding):
        super(BiLSTM_ATT,self).__init__()
        self.batch = config['BATCH']

        self.input_size = input_size
        self.embedding_dim = config['EMBEDDING_DIM']
        
        self.hidden_dim = config['HIDDEN_DIM']
        self.tag_size = output_size 
        
        self.pos_size = config['POS_SIZE']
        self.pos_dim = config['POS_DIM'] 
        
        self.pretrained = config['pretrained']

        if self.pretrained:
            self.word_embeds = nn.Embedding.from_pretrained(torch.FloatTensor(pre_embedding),freeze=False)
        else:
            self.word_embeds = nn.Embedding(self.input_size,self.embedding_dim)

        self.pos1_embeds = nn.Embedding(self.pos_size,self.pos_dim) 
        self.pos2_embeds = nn.Embedding(self.pos_size,self.pos_dim) 
        self.dense = nn.Linear(self.hidden_dim,self.tag_size,bias=True)
        self.relation_embeds = nn.Embedding(self.tag_size,self.hidden_dim)
        self.lstm = nn.LSTM(input_size=self.embedding_dim+self.pos_dim*2,hidden_size=self.hidden_dim//2,num_layers=1, bidirectional=True)
        self.hidden2tag = nn.Linear(self.hidden_dim,self.tag_size)

        self.dropout_emb = nn.Dropout(p=0.5)
        self.dropout_lstm = nn.Dropout(p=0.5)
        self.dropout_att = nn.Dropout(p=0.5)
        
        self.hidden = self.init_hidden()
        self.att_weight = nn.Parameter(torch.randn(self.batch,1,self.hidden_dim))
        self.relation_bias = nn.Parameter(torch.randn(self.batch,self.tag_size,1))
        
    def init_hidden(self):
        return torch.randn(2, self.batch, self.hidden_dim // 2)
        
    def init_hidden_lstm(self):
        return (torch.randn(2, self.batch, self.hidden_dim // 2),
                torch.randn(2, self.batch, self.hidden_dim // 2))
  
    def attention(self,H):
        M = torch.tanh(H) # 非线性变换 size:(batch_size,hidden_dim,seq_len)
        a = F.softmax(torch.bmm(self.att_weight,M),dim=2) # a.Size : (batch_size,1,seq_len)
        a = torch.transpose(a,1,2) # (batch_size,seq_len,1)
        return torch.bmm(H,a) # (batch_size,hidden_dim,1)
    def forward(self,sentence,pos1,pos2):
        self.hidden = self.init_hidden_lstm()
        embeds = torch.cat((self.word_embeds(sentence),self.pos1_embeds(pos1),self.pos2_embeds(pos2)),dim=2)
        embeds = torch.transpose(embeds,0,1)
        lstm_out, self.hidden = self.lstm(embeds, self.hidden)
 
        lstm_out = lstm_out.permute(1,2,0)
        lstm_out = self.dropout_lstm(lstm_out)

        att_out = torch.tanh(self.attention(lstm_out ))
        relation = torch.tensor([i for i in range(self.tag_size)], dtype=torch.long).repeat(self.batch, 1)
        relation = self.relation_embeds(relation)
        out = torch.add(torch.bmm(relation, att_out), self.relation_bias)
        out = F.softmax(out,dim=1)
        return out.view(self.batch,-1) 