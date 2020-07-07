#!/usr/bin/env python
# coding: utf-8

# In[5]:


import argparse
import copy
import os
import torch
from torch import nn, optim
from torch.utils.tensorboard import SummaryWriter
from time import gmtime, strftime

from models import CSATransformer
from mydata import getHotpotData


def test(model, data, mode='test'):
    if mode == 'dev':
        iterator = iter(data.dev_iter)
    else:
        iterator = iter(data.test_iter)
    criterion = nn.BCELoss()    
    model.eval()
    loss, size = 0, 0
    for i, batch in enumerate(iterator):
        pred = model(batch) 
        label = fixlabel( batch, data.LABEL.vocab) 
        for b in range(0,len(pred)):
            x = pred[b]
            y = torch.narrow(label[b],0,0,x.size(0))
            one_loss = criterion(x,y) 
            loss += one_loss
            size += 1
    loss/= size
    return loss

def fixlabel(batch,dit):
    # probality distribute 
    b = batch.Label.transpose(0,1) # batchsize*tensor
    #print(b)
    result = []
    for index_tensor in b:
        p = [0]*100
        p_numpy = index_tensor.to('cpu').numpy()
        for idx in p_numpy:    
            act_idx = dit.itos[idx]
            if act_idx=='<pad>':
                break
            p[int(act_idx)] = 1
        result.append(torch.Tensor(p))
    return result


# In[8]:


def train(args, data):
    model = CSATransformer(args, data).to(args.gpu)
   
    parameters = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = optim.Adadelta(parameters, lr=args.learning_rate, weight_decay=args.weight_decay)
    criterion = nn.BCELoss()

    writer = SummaryWriter(log_dir='./runs/' + args.model_time.replace(':','_'))

    model.train()
    loss, last_epoch = 0, -1
    #max_dev_acc, max_test_acc = 0, 0
    min_dev_loss = 100
    iterator = data.train_iter
    
    for i, batch in enumerate(iterator):
       
        present_epoch = int(iterator.epoch)
        if present_epoch == args.epoch:
            break
        if present_epoch > last_epoch:
            print('epoch:', present_epoch + 1)
        last_epoch = present_epoch
        # pred and label [tensor,tensor,tensor...]
        pred = model(batch) 
        label = fixlabel( batch, data.LABEL.vocab)
        
        # train step
        optimizer.zero_grad()
        
        loss = 0
        for b in range(0,args.batch_size):
            x = pred[b]
            y = torch.narrow(label[b],0,0,x.size(0))
            #print(y)
            batch_loss = criterion(x,y) 
            loss += batch_loss
            batch_loss.backward()  
        loss/= args.batch_size 
        print('Batch {} Loss is {}'.format(i+1,loss))
        optimizer.step()
        
        if (i + 1) % args.print_freq == 0:
            dev_loss = test(model, data, mode='dev')
            print('Dev loss is {}'.format(dev_loss))
            #test_loss, test_acc = test(model, data)
            c = (i + 1) // args.print_freq
            writer.add_scalar('loss/train', loss, c)
            writer.add_scalar('loss/dev', dev_loss, c)
            if dev_loss < min_dev_loss:
                min_dev_loss = dev_loss
                best_model = copy.deepcopy(model)
            model.train()
    writer.close()    
    print(f'Min dev loss: {min_dev_loss:.3f}')
    return best_model

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', default=8, type=int)
    parser.add_argument('--block-size', default=-1, type=int)
    parser.add_argument('--data-type', default='hotpot')
    parser.add_argument('--dropout', default=0.1, type=float)
    parser.add_argument('--epoch', default=20, type=int)
    parser.add_argument('--learning-rate', default=0.001, type=float)
    parser.add_argument('--mSA-scalar', default=5.0, type=float)
    parser.add_argument('--print-freq', default=1000, type=int)
    parser.add_argument('--weight-decay', default=5e-5, type=float)
    parser.add_argument('--word-dim', default=300, type=int)
    parser.add_argument('--csa-mode',default='mul',type = str)
    parser.add_argument('--gpu', default=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'), type=int)
    
    args = parser.parse_args([])

    print('loading Hotpot data...')
    trainpath = './small_train_sep_1000.csv'
    devpath = './small_dev_sep_1000.csv'
    data = getHotpotData(args,trainpath,devpath)
    
    setattr(args, 'model_time', strftime('%H:%M:%S', gmtime()))
    
    print('training start!')
    best_model = train(args, data)

    if not os.path.exists('saved_models'):
        os.makedirs('saved_models')
        
    modeltime = args.model_time.replace(':','_')     
    torch.save(best_model.state_dict(), f'saved_models/CSA_{modeltime}.pt')

    print('training finished!')


if __name__ == '__main__':
    main()

