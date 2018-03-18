import csv
import nltk
import os
import random
import collections
import pickle
from math import *
'''
This file implements naive bayes to determine whether an article is real or 
fake. 
'''
def naive_bayes_istrue(headline,real,fake,real_count,fake_count,m,p_hat):
    '''
    Inputs:
        headline:    New headline which will be classified as real or fake. It 
                     is assumed that the headline is already cleanly processed.
        real:        Dictionary with keys as words which appear in real  
                     headlines and values as the number of times the words 
                     appear.
        fake:        Same type of dictionary but for fake headlines.
        real_count:  Number of real headlines.
        fake_count:  Number of fake headlines.
        m:           Number of virtual examples added to the sets.
        p_hat:       Prior probability. 
        
    Output:
        Boolean value, with true being true and false being fake
    '''
    
    line=headline
    line=line.rstrip('\n')
    temp=line.split(' ') # words contains all the words in the headline
    
    #remove any duplicated words in the headline:
    words=[]
    [words.append(item) for item in temp if item not in words]

    #remove any words in the headline which are not found in the training set:
    [words.remove(word) for word in words if word not in real]
    
    preal=real_count/(real_count+fake_count) #p(real)
    pfake=fake_count/(real_count+fake_count) #p(fake)
    real_prob=0 #Probability that headline is real
    fake_prob=0 #Probability that headline is fake
    
    #First calculate the probability that headline is real
    for key in real:
        if key in words:
            real_prob+=log((real[key] + m*p_hat)/(real_count+m))
        else:
            real_prob+=log(1-(real[key] + m*p_hat)/(real_count+m))
    real_prob+=log(preal)
    
    #Now calculate the probability that headline is fake
    for key in fake:
        if key in words:
            fake_prob+=log((fake[key]+m*p_hat)/(fake_count+m))
        else:
            fake_prob+=log(1-(fake[key]+m*p_hat)/(fake_count+m))
    
    fake_prob+=log(pfake)

    #At this point, we have been summing the log probabilities. We could 
    #exponentiate then divide by the normalization factor to get the actual
    #probabilites, but instead we just return the larger of the two values 
    #since we only care about relative sizes.
    
    if fake_prob>real_prob:
        return False
    else:
        return True

if __name__ == "__main__":
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
            
    #First, tune the hyperparameters:
    performance=0
    optimal_m=0
    optimal_p_hat=0
    max_performance=0
    for m in range (1,10):
        for j in range (5,100,5):
            print(performance,max_performance,m,j)
            p_hat=j/100
            performance=0
            for i in range (0,len(real_val_lines)):
                result=naive_bayes_istrue(real_val_lines[i], real_train,fake_train,counts['real_train'], counts['fake_train'],m,p_hat)
                if result==True:
                    performance+=1
            for i in range (0,len(fake_val_lines)):
                result=naive_bayes_istrue(fake_val_lines[i], real_train,fake_train,counts['real_train'], counts['fake_train'],m,p_hat)
                if result==False:
                    performance+=1
            if performance>max_performance:
                max_performance=performance
                optimal_m=m
                optimal_p_hat=p_hat
    
    #Next, determine the performance on the training and test sets:
    #optimal_m=1
    #optimal_p_hat=0.35
    print('program running')
    m=1
    p_hat=0.35
    performance=0
    for i in range (0,len(real_train_lines)):
        result=naive_bayes_istrue(real_train_lines[i], real_train,fake_train,counts['real_train'], counts['fake_train'],m,p_hat)
        if result==True:
            performance+=1
    for i in range(0,len(fake_train_lines)):
        result=naive_bayes_istrue(fake_train_lines[i], real_train,fake_train,counts['real_train'], counts['fake_train'],m,p_hat)
        if result==False:
            performance+=1
    train_performance=performance/(len(real_train_lines)+len(fake_train_lines))
    
    performance=0
    for i in range (0,len(real_test_lines)):
        result=naive_bayes_istrue(real_test_lines[i], real_train,fake_train,counts['real_train'], counts['fake_train'],m,p_hat)
        if result==True:
            performance+=1
    for i in range(0,len(fake_test_lines)):
        result=naive_bayes_istrue(fake_test_lines[i], real_train,fake_train,counts['real_train'], counts['fake_train'],m,p_hat)
        if result==False:
            performance+=1
    test_performance=performance/(len(real_test_lines)+len(fake_test_lines))
    

    
    
    
    