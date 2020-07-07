#!/usr/bin/env python
# coding: utf-8

# In[1]:


import spacy
import csv
import pyprind
from torchtext import data
from torchtext.vocab import GloVe
import numpy as np
import torch
from torchtext.data import Iterator, BucketIterator
#from transformers import BertTokenizer
#from spacy.symbols import ORTH

class dataProcesser():
    def __init__(self,src,des,n):
        self.src = src
        self.des = des
        self.n = n
        self.dics = {}
        self.sepToken = '<sep> '
        self.endToken = '<sep>'
        CONTEXT = data.Field()
        ANSWER  = data.Field()
        QUESTION = data.Field()
       
        # define col: {[source data col name]:[your data col name],Field}
        fields = {'context':('Context',CONTEXT),'question':('Question',QUESTION),'supporting_facts':('Answer',ANSWER)}
        dataset = data.TabularDataset(path = src,format='json',fields=fields)
        dataset = dataset.examples[0]
        
        self.dics = []
        #len(dataset.Context)
        for i in range (0,len(dataset.Context)):
            s_id = 0
            ts = {}
            for title,sentence in dataset.Context[i]:
                # title:str sentence:list     
                ts[title] = [sentence,s_id]
                s_id = s_id + len(sentence)
            self.dics.append(ts)
        self.go(dataset)
    
    def getAnswer(self,ans,idx):
        # idx: data index
        label = ''
        res = ''
        dic = self.dics[idx]
        for title, sent_id in ans:
            if title in dic:
                if sent_id < len(dic[title][0]):
                    res += dic[title][0][sent_id] + self.sepToken
                    #print('write data {} {}'.format(dic[title][1],sent_id))
                    label += (str(dic[title][1]+sent_id) + ',')
        label = label[:-1]
        return [res,label]
    
    def getContext(self,text2DimList):
        res = ''
        for paragragh in text2DimList:
            res += self.sepToken.join(paragragh[1]) + self.sepToken
        return  res  

    def go(self,dataset):
        if self.n is -1:
            self.n = len(dataset.Context)
            print('data length is',self.n)
        pbar = pyprind.ProgBar(self.n)
        with open(self.des,'w',encoding="utf-8",newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['passage','question', 'answer','label'])
            for i in range (0,self.n):
                c =  self.getContext(dataset.Context[i]).lower()
                q =  dataset.Question[i].lower()
                a,l =  self.getAnswer(dataset.Answer[i],i)
                writer.writerow([c,q,a,l])
                pbar.update()
        print('write down')  
        
        


# In[2]:


class getHotpotData():
    def __init__(self,args,trainPath,devPath):
               
        # args 
        self.args = args
        self.trainpath= trainPath
        self.devpath= devPath
        self.batch_size = args.batch_size
        # Tokenizer
        #self.spacy_Tokenizer = spacy.load('en_core_web_sm') 
        #self.bert_Tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
        
        # Data field
        self.ANSWER  = data.Field(tokenize = self.t_tokenizer,lower=True)
        self.QUESTION = data.Field(tokenize = self.t_tokenizer)
        self.PASSAGE = data.Field(tokenize = self.t_tokenizer)
        self.LABEL = data.Field(tokenize = self.label_tokenizer)
        
        fields = {'passage':('Passage', self.PASSAGE),'question':('Question', self.QUESTION)
                  ,'answer':('Answer',  self.ANSWER ),'label':('Label',self.LABEL)}  
        
        self.train = data.TabularDataset(path = self.trainpath,format='csv',fields=fields) 
        self.dev = data.TabularDataset(path = self.devpath,format='csv',fields=fields)
        
        self.PASSAGE.build_vocab(self.train,self.dev, vectors=GloVe(name='6B', dim=300))  
        self.QUESTION.build_vocab(self.train) 
        self.ANSWER.build_vocab(self.train)
        self.LABEL.build_vocab(self.train,self.dev)
        
        # sep index
        self.sep_index = self.PASSAGE.vocab.stoi['<sep>']
        
        # iter
        self.train_iter = data.BucketIterator(dataset=self.train, 
                                              batch_size=args.batch_size,                                        
                                              shuffle=False,
                                              sort_within_batch=False, 
                                              repeat=False,device=args.gpu)
        # sort_key=lambda x: len(x.Question) 
          #,sort_key=lambda x: len(x.Question) 
        self.dev_iter = data.BucketIterator(dataset=self.dev, batch_size=args.batch_size,
                                            shuffle=False,
                                            sort_within_batch=False,
                                            repeat=False,device=args.gpu)
        

        # caculate block size
        #self.calculate_block_size(args.batch_size)   
    """          
    def bert_tokenizer(self,text):
        return  self.bert_Tokenizer.tokenize(text)
    def spacy_tokenizer(self,text):
        return [str(token) for token in self.spacy_Tokenizer(text)]
    """
    def label_tokenizer(self,text):
        return [i for i in text.split(',')]
        
    
    def t_tokenizer(self,text):
        speciallToken = ['(',')',',','?','!',';',',','"','-','.<sep>']
        for t in speciallToken:
            if t is '.<sep>':
                text = text.replace(t,' . <sep>')
            else:
                text = text.replace(t,' '+t+' ')
        text = text.replace('  ',' ')
        text = text.replace('  ',' ')
        return text.split(' ')
  
    def spilter(self,x,tk):
        res = []
        s = 0
        sqlen = len(x)
        for i,t in enumerate(x): 
            if t == tk:
                res.append(x[s:i])
                s = i+1   
        if s is 0:       
            res.append(x[s:-1])
        return res  




