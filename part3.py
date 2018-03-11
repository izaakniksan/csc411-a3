import csv
import nltk
import os
import random
import collections
import pickle
from math import *
import part2 as p2
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

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
        
    #part 3 a:
    
    ten_frequent_real = sorted(real_train, key=real_train.get, reverse=True)[:10]
    print('The most common words in real emails are:',ten_frequent_real,'\n')
    
    ten_infrequent_real = sorted(real_train, key=real_train.get, reverse=True)[-10:]
    print('The least common words in real emails are:',ten_infrequent_real,'\n')
    
    ten_frequent_fake = sorted(fake_train, key=fake_train.get, reverse=True)[:10]
    print('The most common words in fake emails are:',ten_frequent_fake,'\n')
    
    ten_infrequent_fake = sorted(fake_train, key=fake_train.get, reverse=True)[-10:]
    print('The least common words in fake emails are:',ten_infrequent_fake,'\n')
    
    #part 3 b:
    real_nostop=real_train.copy()
    fake_nostop=fake_train.copy()

    #Remove stopwords:
    for key in real_train:
        if key in ENGLISH_STOP_WORDS:
            real_nostop.pop(key, None)
    
    for key in fake_train:
        if key in ENGLISH_STOP_WORDS:
            fake_nostop.pop(key, None)
    
    ten_frequent_real_nostop = sorted(real_nostop, key=real_nostop.get,\
                                      reverse=True)[:10]
    print('Disregarding stopwords, the most common words in real emails are:',\
          ten_frequent_real_nostop,'\n')
    
    ten_frequent_fake_nostop=sorted(fake_nostop, key=fake_nostop.get,\
                                      reverse=True)[:10]
    print('Disregarding stopwords, the most common words in fake emails are:',\
          ten_frequent_fake_nostop,'\n')