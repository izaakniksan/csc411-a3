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
    words=line.split(' ') # words contains all the words in the headline
    
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
            temp=(1-(fake[key]+m*p_hat)/(fake_count+m))
            fake_prob+=log(1-(fake[key]+m*p_hat)/(fake_count+m))
    
    fake_prob+=log(pfake)

    #At this point, we have been summing the log probabilities. We could 
    #exponentiate then divide by the normalization factor to get the actual
    #probabilites, but instead we just want to take the magnitudes (since the
    #sum of logs of small decimals will be a negative number) and return
    #the larger of the two values.
    
    real_prob=abs(real_prob)
    fake_prob=abs(fake_prob)
    
    if fake_prob>real_prob:
        return False
    else:
        return True

if __name__ == "__main__":
    #Open the saved dictionaries:
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
    m=1
    p_hat=0.5
    print(naive_bayes_istrue(real_test_lines[1], real_train,fake_train,counts['real_train'], counts['fake_train'],m,p_hat))
    