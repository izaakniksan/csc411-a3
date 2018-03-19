import csv
import nltk
import os
import random
import collections
import pickle
from math import *
import torch 
from numpy import *
import matplotlib.pyplot as plt
import part4 as p4
from sklearn import tree

def H(Y):
    '''
    This function calculates the entropy of random variable Y with two possible
    classes. Y is an array of length #samples, with each element being 0 or 1.
    0 corresponds to real news, 1 corresponds to fake news.
    '''
    totalcount=len(Y)

    count0=(Y == 0).sum()
    count1=(Y == 1).sum()
    
    p_0=count0/totalcount
    p_1=count1/totalcount
    
    entropy = -p_0*math.log(p_0,2) - p_1*math.log(p_1,2)
    return entropy

#def mutual_information(Y,x,H):
#    '''
#    This function computes the mutual information of x (#samples,1) which
#    is a chosen feature, and Y. Y is defined the same way as it was for H(Y)
#    '''
#    totalcount=len(Y)
#    
#    HYX=x.sum() #number of times wordi appears over all samples
#    HYX= -HYX*((Y == 0).sum()/totalcount)
#    return 0
#    return H(Y)-HYX

if __name__ == "__main__":
    print('*** Part 8 running ***')
    
    with open('real_train.pickle', 'rb') as handle:
        real_train = pickle.load(handle)  
    with open('real_val.pickle', 'rb') as handle:
        real_val = pickle.load(handle)   
    with open('real_test.pickle', 'rb') as handle:
        real_test = pickle.load(handle)
    with open('fake_train.pickle', 'rb') as handle:
        fake_train = pickle.load(handle)
    with open('fake_val.pickle', 'rb') as handle:
        fake_val = pickle.load(handle)
    with open('fake_test.pickle', 'rb') as handle:
        fake_test = pickle.load(handle)
    with open('counts.pickle', 'rb') as handle:
        counts = pickle.load(handle)
    with open('real_train_lines.pickle', 'rb') as handle:
        real_train_lines = pickle.load(handle)
    with open('real_val_lines.pickle', 'rb') as handle:
        real_val_lines = pickle.load(handle)
    with open('real_test_lines.pickle', 'rb') as handle:
        real_test_lines = pickle.load(handle)
    with open('fake_train_lines.pickle', 'rb') as handle:
        fake_train_lines = pickle.load(handle)
    with open('fake_val_lines.pickle', 'rb') as handle:
        fake_val_lines = pickle.load(handle)
    with open('fake_test_lines.pickle', 'rb') as handle:
        fake_test_lines = pickle.load(handle)
        
        
#--------------------IMPORT TREE FROM P7-------------------------
    with open('optimal_tree.pickle', 'rb') as handle:
        clf = pickle.load(handle)
        
#--------------------CREATE SETS,INPUTS,OUTPUTS-------------------------
    print('creating network inputs')
    #First create trainingset
    trainingset=append(real_train_lines,fake_train_lines)

    #Make arrays of words for trainingset
    trainingset_words=[]
    for i in range(0,len(trainingset)):
        trainingset_words.append(p4.clean_headline(trainingset[i],trainingset))

    #Create input vector x_train

    x_train=p4.create_v(trainingset_words,all_words)
    x_train=x_train.T #since create_v was initially meant for p4
    
    #Create output vector y_train
    y_train=zeros((len(trainingset_words)))
    for i in range(0,len(real_train_lines)):
        y_train[i]=0 #real
    for i in range(len(real_train_lines),len(trainingset_words)):
        y_train[i]=1 #fake
    
    #Create validation set
    validationset=append(real_val_lines,fake_val_lines)
    
    #Make arrays of words for validation set
    validationset_words=[]
    for i in range(0,len(validationset)):
        validationset_words.append(p4.clean_headline(validationset[i],trainingset))
        
    #Create x_val
    
    x_val=p4.create_v(validationset_words,all_words)
    x_val=x_val.T
    
    #Create output vector y_val
    y_val=zeros((len(validationset_words)))
    for i in range(0,len(real_val_lines)):
        y_val[i]=0 #real
    for i in range(len(real_val_lines),len(validationset_words)):
        y_val[i]=1 #fake
    print('entropy of y_train is:',H(y_train))
    
#--------------------8. a)-------------------------
    #Compute mutual information of the top word 'the'
    wordi_index=all_words.index('the')
    xi=x_train[:,wordi_index]
    totalcount=len(y_train)

    #H(Y)-H(Y|wordi) : (assume cond ind)
    mi_i=H(y_train)-((xi==1).sum()/totalcount)*H(y_train)

#--------------------8. b)-------------------------
    wordj_index=all_words.index('trumps')
    xj=x_train[:,wordj_index]
    
    mi_j=H(y_train)-((xj==1).sum()/totalcount)*H(y_train)
    
    print('mutual information for "the" is ',mi_i)
    print('mutual information for "trumps" is ',mi_j)
    print('*** Part 8 Finished ***\n')
    print('Project 3 Completed!')