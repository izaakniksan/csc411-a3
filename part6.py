import csv
import nltk
import os
import random
import collections
import pickle
from math import *
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
import torch 
from numpy import *
from torch.autograd import Variable
import matplotlib.pyplot as plt

if __name__ == "__main__":
    print('*** Part 6 running ***')
    print('importing workspace')
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
     
    print('importing optimal_w from part4.py')  
    with open('part_4_w.pickle', 'rb') as handle:
        w = pickle.load(handle)  
    
    all_words=[]
    [all_words.append(word) for word in real_train.keys()]
    #PART 6 a)
    #create theta array:
    theta=zeros((len(w)))
    
    #fill in theta with the values derived in part5. This theta corresponds to
    #the thetas for the REAL HEADLINE inequality
    theta=w[:,0]-w[:,1]
    theta=theta[1:]
    
    #find indices of largest 10 elements
    print('the following correspond to stopwords included: \n')
    
    ind = argpartition(theta, -10)[-10:]
    ind = ind[argsort(theta[ind])]
    ind=flip(ind,0)
    print('\nthe 10 largest values of theta are: ',theta[ind],'\n') 
    print('the words corresponding to the 10 largest values are: ',[all_words[i] for i in ind],'\n')
    
    #find indices of smallest 10 elements
    ind = argpartition(theta, 10)[:10]
    ind = ind[argsort(theta[ind])]
    print('the 10 most negative values of theta are: ',theta[ind],'\n')
    print('the words corresponding to the 10 most negative values are: ',[all_words[i] for i in ind],'\n')
    
    
    #PART 6 b)
    words_nostop=[]
    stopword_ind=[]
    i=0
    
    for word in all_words:
        if word not in ENGLISH_STOP_WORDS:
            words_nostop.append(word)
        else:
            stopword_ind.append(i)
        i=i+1
        
    theta=delete(theta,stopword_ind)
    
    #find indices of largest 10 elements
    print('the following correspond to stopwords excluded: \n')
    
    ind = argpartition(theta, -10)[-10:]
    ind = ind[argsort(theta[ind])]
    ind=flip(ind,0)
    print('\nthe 10 largest values of theta are: ',theta[ind],'\n') 
    print('the words corresponding to the 10 largest values are: ',[words_nostop[i] for i in ind],'\n')
    
    #find indices of smallest 10 elements
    ind = argpartition(theta, 10)[:10]
    ind = ind[argsort(theta[ind])]
    print('the 10 most negative values of theta are: ',theta[ind],'\n')
    print('the words corresponding to the 10 most negative values are: ',[words_nostop[i] for i in ind],'\n')
    print('*** part 6 finished ***\n')
        