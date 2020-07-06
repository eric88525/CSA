#!/usr/bin/env python
# coding: utf-8

# # CSA

# In[1]:


import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.autograd import Variable
from torchtext import data
from torchtext.vocab import GloVe
import numpy as np


class customizedModule(nn.Module):
    def __init(self):
        super(customizedModule,self).__init()
    def customizedLinear(self,in_dim,out_dim,activation=None,dropout=False):
        c1 = nn.Sequential(nn.Linear(in_dim,out_dim))
        nn.init.xavier_uniform_(c1[0].weight)
        nn.init.constant_(c1[0].bias,0)
        
        if activation is not None:
            c1.add_module(str(len(c1)),activation)
        if dropout:
            c1.add_module(str(len(c1)),nn.Dropout(p=self.args.dropout))  
        return c1

class CrossAttention(customizedModule):
    def __init__(self,dx,dq,mode):
        super(CrossAttention,self).__init__()
        self.w1 = self.customizedLinear(dx,dx)
        self.w2 = self.customizedLinear(dq,dx)   
        self.w1[0].bias.requires_grad = False
        self.w2[0].bias.requires_grad = False
        
        # bias for add attention
        self.wt = self.customizedLinear(dx,1)
        self.wt[0].bias.requires_grad = False
        self.bsa = nn.Parameter(torch.zeros(dx))  
        # 'mul' or 'add'
        self.mode = mode  
        self.debug = False
    def forward(self,x,q):
        if self.mode is 'mul':     
            # W(1)x W(2)c
            wx = self.w1(x)
            wq = self.w2(q)
            wq = wq.unsqueeze(-2)         
            # <x,q>
            p = wx*wq  
            # p = [a0,a1,a2...]
            p = torch.sum(p,dim=-1,keepdim=True)    
            # softmax along row       
            p = F.softmax(p,dim=-2)  
            #p = torch.reshape(p,(p.size(0),-1))
            return p
        
        elif self.mode is 'add':   
            wx = self.w1(x)
            wq = self.w2(q) 
            wq = wq.unsqueeze(-2)
            p = self.wt(wx+wq+self.bsa)
            p = F.softmax(p,dim = -2)
            return p
        else:
            raise NotImplementedError('CrossAttention error:<mul or add>')

class PositionwiseFeedForward(customizedModule):
    ''' A two-feed-forward-layer module '''

    def __init__(self, d_in, d_hid, dropout=0.1):
        super().__init__()
        self.w_1 = self.customizedLinear(d_in, d_hid) # position-wise
        self.w_2 = self.customizedLinear(d_hid, d_in) # position-wise
        self.layer_norm = nn.LayerNorm(d_in, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        x = self.w_2(F.relu(self.w_1(x)))
        x = self.dropout(x)
        x += residual
        x = self.layer_norm(x)
        return x
    
class csa(customizedModule):
    def __init__(self,args,dx,dq):
        super(csa,self).__init__()
        self.args = args
        self.dx = dx
        self.dq = dq  
        if self.args.csa_mode is 'mul':
            self.crossAttention = CrossAttention(dx,dq,'mul')
        elif self.args.csa_mode is 'add':
            self.crossAttention = CrossAttention(dx,dq,'add')
        else:
            raise NotImplementedError('CSA->CrossAttention error')
        
        self.Wsa1 = self.customizedLinear(dx,dx)
        self.Wsa2 = self.customizedLinear(dx,dx)
        self.Wsa1[0].bias.requires_grad = False
        self.Wsa2[0].bias.requires_grad = False
        self.wsat = self.customizedLinear(dx,1)
        self.bsa1 = nn.Parameter(torch.zeros(dx))  
        self.bsa2 = nn.Parameter(torch.zeros(dx)) 
        
        self.debug = False
        self.PFN = PositionwiseFeedForward(dx,dx)
    def forward(self,x,c):
        # x(batch,seq_len,word_dim) c(batch,word_dim)
        seq_len = x.size(-2)
        p = self.crossAttention(x,c)     
        h = x*p       
        # p = (seq_len*seq_len): the attention of xi to xj
        hi = self.Wsa1(h)
        hj = self.Wsa2(h)
        hi = hi.unsqueeze(-2)
        hj = hj.unsqueeze(-3)
        
        #fcsa(xi,xj|c)
        fcsa = hi+hj+self.bsa1
        fcsa = self.wsat(fcsa)
        fcsa = torch.sigmoid(fcsa)
        fcsa = fcsa.squeeze()
        
        # mask 對角
        M = Variable(torch.eye(seq_len)).to(self.args.gpu).detach()
        M[M==1]= float('-inf')
        fcsa = fcsa+M  
        fcsa = F.softmax(fcsa,dim=-1)          
        fcsa = fcsa.unsqueeze(-1)
        ui = fcsa*x.unsqueeze(1) 
        ui = torch.sum(ui,1)
        ui = self.PFN(ui)
        return  ui


# # Passage Encoder

# In[2]:




class customizedModule(nn.Module):
    def __init__(self):
        super(customizedModule, self).__init__()

    # linear transformation (w/ initialization) + activation + dropout
    def customizedLinear(self, in_dim, out_dim, activation=None, dropout=False):
        cl = nn.Sequential(nn.Linear(in_dim, out_dim))
        nn.init.xavier_uniform_(cl[0].weight)
        nn.init.constant_(cl[0].bias, 0)

        if activation is not None:
            cl.add_module(str(len(cl)), activation)
        if dropout:
            cl.add_module(str(len(cl)), nn.Dropout(p=self.args.dropout))

        return cl




class BiBloSAN(customizedModule):
    def __init__(self, args):
        super(BiBloSAN, self).__init__()

        self.args = args

        self.mBloSA_fw = mBloSA(self.args, 'fw')
        self.mBloSA_bw = mBloSA(self.args, 'bw')

        # two untied fully connected layers
        self.fc_fw = self.customizedLinear(self.args.word_dim, self.args.word_dim, activation=nn.ReLU())
        self.fc_bw = self.customizedLinear(self.args.word_dim, self.args.word_dim, activation=nn.ReLU())

        self.s2tSA = s2tSA(self.args, self.args.word_dim * 2)

    def forward(self, x):
        input_fw = self.fc_fw(x)
        input_bw = self.fc_bw(x)

        # (batch, seq_len, word_dim)
        u_fw = self.mBloSA_fw(input_fw)
        u_bw = self.mBloSA_bw(input_bw)    
        u_bi = torch.cat([u_fw, u_bw], dim=2)
        #print('ufw: {} ubw: {} u_bi: {}'.format(u_fw.shape,u_bw.shape,u_bi.shape))
        # (batch, seq_len, word_dim * 2) -> (batch, word_dim * 2)
        #u_bi = self.s2tSA(torch.cat([u_fw, u_bw], dim=2))
        return u_bi


class mBloSA(customizedModule):
    def __init__(self, args, mask):
        super(mBloSA, self).__init__()

        self.args = args
        self.mask = mask

        # init submodules
        self.s2tSA = s2tSA(self.args, self.args.word_dim)
        self.init_mSA()
        self.init_mBloSA()

    def init_mSA(self):
        self.m_W1 = self.customizedLinear(self.args.word_dim, self.args.word_dim)
        self.m_W2 = self.customizedLinear(self.args.word_dim, self.args.word_dim)
        self.m_b = nn.Parameter(torch.zeros(self.args.word_dim))

        self.m_W1[0].bias.requires_grad = False
        self.m_W2[0].bias.requires_grad = False

        self.c = nn.Parameter(torch.Tensor([self.args.c]), requires_grad=False)

    def init_mBloSA(self):
        self.g_W1 = self.customizedLinear(self.args.word_dim, self.args.word_dim)
        self.g_W2 = self.customizedLinear(self.args.word_dim, self.args.word_dim)
        self.g_b = nn.Parameter(torch.zeros(self.args.word_dim))

        self.g_W1[0].bias.requires_grad = False
        self.g_W2[0].bias.requires_grad = False

        self.f_W1 = self.customizedLinear(self.args.word_dim * 3, self.args.word_dim, activation=nn.ReLU())
        self.f_W2 = self.customizedLinear(self.args.word_dim * 3, self.args.word_dim)

    def mSA(self, x):
        """
        masked self-attention module
        :param x: (batch, (block_num), seq_len, word_dim)
        :return: s: (batch, (block_num), seq_len, word_dim)
        """
        seq_len = x.size(-2)

        # (batch, (block_num), seq_len, 1, word_dim)
        x_i = self.m_W1(x).unsqueeze(-2)
        # (batch, (block_num), 1, seq_len, word_dim)
        x_j = self.m_W2(x).unsqueeze(-3)

        # build fw or bw masking
        # (seq_len, seq_len)
        M = Variable(torch.ones((seq_len, seq_len))).to(self.args.gpu).triu().detach()
        M[M == 1] = float('-inf')

        # CASE 1 - x: (batch, seq_len, word_dim)
        # (1, seq_len, seq_len, 1)
        M = M.contiguous().view(1, M.size(0), M.size(1), 1)
        # (batch, 1, seq_len, word_dim)
        # padding to deal with nan
        pad = torch.zeros(x.size(0), 1, x.size(-2), x.size(-1))
        pad = Variable(pad).to(self.args.gpu).detach()

        # CASE 2 - x: (batch, block_num, seq_len, word_dim)
        if len(x.size()) == 4:
            M = M.unsqueeze(1)
            pad = torch.stack([pad] * x.size(1), dim=1)

        # (batch, (block_num), seq_len, seq_len, word_dim)
        f = self.c * torch.tanh((x_i + x_j + self.m_b) / self.c)

        # fw or bw masking
        if f.size(-2) > 1:
            if self.mask == 'fw':
                M = M.transpose(-2, -3)
                f = F.softmax((f + M).narrow(-3, 0, f.size(-3) - 1), dim=-2)
                f = torch.cat([f, pad], dim=-3)
            elif self.mask == 'bw':
                f = F.softmax((f + M).narrow(-3, 1, f.size(-3) - 1), dim=-2)
                f = torch.cat([pad, f], dim=-3)
            else:
                raise NotImplementedError('only fw or bw mask is allowed!')
        else:
            f = pad

        # (batch, (block_num), seq_len, word_dim)
        s = torch.sum(f * x.unsqueeze(-2), dim=-2)
        return s

    def forward(self, x):
        """
        masked block self-attention module
        :param x: (batch, block,seq_len, word_dim)
        :param M: (seq_len, seq_len)
        :return: (batch, seq_len, word_dim)
        """
        r = x.size(-2)
        n = x.size(1)
        # (batch, block_num(m), seq_len(r), word_dim)
        h = self.mSA(x)
        # (batch, block_num(m), word_dim)
        v = self.s2tSA(h)

        # --- Inter-block self-attention ---
        # (batch, m, word_dim)
        o = self.mSA(v)
        # (batch, m, word_dim)
        G = torch.sigmoid(self.g_W1(o) + self.g_W2(v) + self.g_b)
        # (batch, m, word_dim)
        e = G * o + (1 - G) * v

        # --- Context fusion ---
        # (batch, n, word_dim)
        E = torch.cat([torch.stack([e.select(1, i)] * r, dim=1) for i in range(e.size(1))], dim=1).narrow(1, 0, n)
        x = x.view(x.size(0), -1, x.size(-1)).narrow(1, 0, n)
        h = h.view(h.size(0), -1, h.size(-1)).narrow(1, 0, n)

        # (batch, n, word_dim * 3) -> (batch, n, word_dim)
        fusion = self.f_W1(torch.cat([x, h, E], dim=2))
        G = torch.sigmoid(self.f_W2(torch.cat([x, h, E], dim=2)))
        # (batch, n, word_dim)
        u = G * fusion + (1 - G) * x

        return u


class s2tSA(customizedModule):
    def __init__(self, args, hidden_size):
        super(s2tSA, self).__init__()

        self.args = args
        self.s2t_W1 = self.customizedLinear(hidden_size, hidden_size, activation=nn.ReLU())
        self.s2t_W = self.customizedLinear(hidden_size, hidden_size)

    def forward(self, x):
        """
        source2token self-attention module
        :param x: (batch, (block_num), seq_len, hidden_size)
        :return: s: (batch, (block_num), hidden_size)
        """

        # (batch, (block_num), seq_len, word_dim)
        f = self.s2t_W1(x)
        f = F.softmax(self.s2t_W(f), dim=-2)
        # (batch, (block_num), word_dim)
        s = torch.sum(f * x, dim=-2)
        return s

class psEncoder(customizedModule):
    def __init__(self, args):
        super(psEncoder, self).__init__()
        self.args = args
        self.args.c = self.args.mSA_scalar
        self.BiBloSAN = BiBloSAN(self.args)
        self.l = self.customizedLinear(args.word_dim*2,args.word_dim,activation=nn.Sigmoid())
    def forward(self, p):
        # p (batch,block,seq_len,word_dim)
        s = self.BiBloSAN(p)
        s = self.l(s)
        return s


# # Query Encoder

# In[3]:


class Q_S2T(customizedModule):
    def __init__(self, hidden_size):
        super(Q_S2T, self).__init__()

        self.s2t_W1 = self.customizedLinear(hidden_size, hidden_size, activation=nn.ReLU())
        self.s2t_W = self.customizedLinear(hidden_size, hidden_size)

    def forward(self, x):
        """
        source2token self-attention module
        :param x: (batch, seq_len, hidden_size)
        :return: s: (batch, hidden_size)
        """

        # (batch, (block_num), seq_len, word_dim)
        f = self.s2t_W1(x)
        f = F.softmax(self.s2t_W(f), dim=-2)
        # (batch, (block_num), word_dim)
        s = torch.sum(f * x, dim=-2)
        return s    


class qEncoder(nn.Module):
    def __init__(self,word_dim,n_head,n_hid,dropout,nlayers):
        super(qEncoder, self).__init__()
        from torch.nn import TransformerEncoder, TransformerEncoderLayer
        encoder_layers = TransformerEncoderLayer(word_dim, n_head, n_hid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.s2t = Q_S2T(word_dim)
        #self.l = nn.Linear(300,1)
    def forward(self,x):
        #(batch,sequence,worddim)
        x = self.transformer_encoder(x,None)
        x = self.s2t(x)
        return  x


# In[4]:


class CSATransformer(customizedModule):
    def __init__(self, args, data):
        super(CSATransformer, self).__init__()

        self.args = args
        self.data = data
       
        self.word_emb = nn.Embedding(len(data.PASSAGE.vocab.vectors), len(data.PASSAGE.vocab.vectors[0]))
        
        # initialize word embedding with GloVe
        self.word_emb.weight.data.copy_(data.PASSAGE.vocab.vectors)
        
        # fine-tune the word embedding
        self.word_emb.weight.requires_grad = True
        
        # index for <sep>
        self.sep_index = data.PASSAGE.vocab.stoi['<sep>']
                
            
        #(self,word_dim,n_head,n_hid,dropout,nlayers):    
        # model list
        self.p_encoder = psEncoder(args)
        self.q_encoder = qEncoder(300,4,300,0.1,2)
        self.csa = csa(args,args.word_dim, args.word_dim)
        self.decoder =  nn.Sequential(nn.Linear(args.word_dim, 1),nn.Sigmoid())
        
    def batch_init(self,batch):
        # transpose batch data to [batchsize*passage_index]
        batch.Question = batch.Question.transpose(0,1)
        batch.Answer = batch.Answer.transpose(0,1)
        batch.Passage = batch.Passage.transpose(0,1)       
        
        return batch
    
    # input a [passage_len] tensor represent a passage
    # padding to it's passage max length
    def to_block(self,passage,mlen):
        #print('we going to block!',mlen)
        t_list = []
        nt = passage.to('cpu').numpy()
        sep_index = np.where(nt == self.sep_index)[0]
        pre_index = 0
        for i,s in enumerate(sep_index):      
            slen = s - pre_index -1    
            pad_len = mlen - slen
            if pad_len < 0:
                print('slen<0! = ',sep_index,pre_index,s)
            pad = Variable(torch.zeros(pad_len).long()).to(self.args.gpu).detach()
            #print('pad is:',pad,pad.shape)
            # tensor of sentence
            if i is 0:
                s_t = passage.narrow(0,0,slen)
            else:
                s_t = passage.narrow(0,pre_index+1,slen)
            #print('p',s_t.shape,'pad',pad.shape)    
            t_list.append(torch.cat([s_t,pad]))
            pre_index = s    
        blocks = torch.stack(t_list,dim = 0)  
        #print('block:',blocks.shape)
        return blocks
    
    #return the max sentence length in a passage
    #p is a [1*passagelength] tensor
    def maxPassageSL(self,passage):
        p_numpy = passage.to('cpu').numpy()
        sep_index = np.where(p_numpy == self.sep_index)[0]
        pre_index = 0
        mlen = 0 
        for s in sep_index:         
            senlen = s - pre_index
            if senlen > mlen:
                mlen = senlen
            pre_index = s 
        return mlen
    
    def forward(self, batch):
          
        batch = self.batch_init(batch)
        pred = []
        for i in range(0,batch.batch_size):    
            p = batch.Passage[i] # p (passage_len)
            p = self.to_block(p,self.maxPassageSL(p)).unsqueeze(0) # p (batch(1),block,passage_index)
            p = self.word_emb(p) # p (batch(1),block,passage_len,word_dim)
            q = batch.Question[i] # a tensor [passage_length]
            q = self.word_emb(q).unsqueeze(0) # q (question_length,word_dim)
            #print('P:{} Q {}'.format(p.shape,q.shape))
            q = self.q_encoder(q)
            p = self.p_encoder(p) 
            c = self.csa(p,q)
            res = self.decoder(c).squeeze()
            #print('After encode: P {} Q {} RES {}'.format(p.shape,q.shape,res.shape))
            #print(res)
            pred.append(res)
        return pred

