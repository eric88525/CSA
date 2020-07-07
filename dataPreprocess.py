#!/usr/bin/env python
# coding: utf-8

# In[1]:


from mydata import dataProcesser

src = './dataset/train.json'
des = './dataset/train.csv'
dataProcesser(src,des,-1)

src = './dataset/dev.json'
des = './dataset/dev.csv'

dataProcesser(src,des,-1)

