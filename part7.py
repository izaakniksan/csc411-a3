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


#Note: credit to graphviz for plotting the decision trees. The code for plotting
#them is commented out since it is not installed in the cs teaching lab.

if __name__ == "__main__":
    print('*** Part 7 running ***')
    
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

    alpha=0.001
    _lambda=0.8
    all_words=[]
    [all_words.append(word) for word in real_train.keys()]
    
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
        
#--------------------CREATE AND TRAIN DECISION TREE------------------------
    clf = tree.DecisionTreeClassifier()
    clf = clf.fit(x_train, y_train)
    
    depths=[1,2,5,10,20,40,75,120,200,500]
    '''
    import graphviz
    val_prediction=clf.predict(x_val)
    dot_data = tree.export_graphviz(clf, out_file=None) 
    graph = graphviz.Source(dot_data) 
    graph.render("p7b_2layers")
    '''
    
    print('*** Part 7 finished ***')