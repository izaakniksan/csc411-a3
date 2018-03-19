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
from mpl_toolkits.mplot3d import Axes3D


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
    #defaults:
    clf = tree.DecisionTreeClassifier()
    clf = clf.fit(x_train, y_train)
    
    #training:
    depths=[5,10,25,50,100,150,300,500,750,1000] #values for max_depth
    max_feat_perc=[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0] # % values for max_features
    train_performance=zeros((len(depths),len(max_feat_perc)))
    val_performance=train_performance.copy()
    index1=0
    index2=0
    print('optimization about to start')
    for depth in depths:
        index2=0
        for percent in max_feat_perc:
                #create the tree for these parameters:
                clf = tree.DecisionTreeClassifier(max_depth=depth,max_features=percent)
                clf = clf.fit(x_train, y_train)
                
                #performance on training set:
                predict=clf.predict(x_train)
                for i in range(0,len(x_train)):
                    if y_train[i]==predict[i]:
                        train_performance[index1][index2]=train_performance[index1][index2]+1
                        
                #performance on validation set:
                predict=clf.predict(x_val)
                for i in range(0,len(x_val)):
                    if y_val[i]==predict[i]:
                        val_performance[index1][index2]=val_performance[index1][index2]+1
                index2=index2+1
        index1=index1+1
        print(index1*10,'% finished optimizing')
    
    
    max_train_perf=unravel_index(argmax(train_performance, axis=None), train_performance.shape)
    print('\nOptimal Training Set Parameters:')
    print('max_depth: ',depths[max_train_perf[0]],'percentage: ',max_feat_perc[max_train_perf[1]]*100)
    
    max_val_perf=unravel_index(argmax(val_performance, axis=None), val_performance.shape)
    print('Optimal Validation Set Parameters:')
    print('max_depth: ',depths[max_val_perf[0]],'percentage: ',max_feat_perc[max_val_perf[1]]*100)
    
    #Now plot performance for training and validation sets for varying max_depths
    #and a constant number of max features (chosen to be the optimal one for the
    #VALIDATION SET)
    train_norm=len(x_train[:,0])
    val_norm=len(x_val[:,0])
    plt.plot(depths,train_performance[:,max_val_perf[1]]/train_norm, label='Training')
    plt.plot(depths,val_performance[:,max_val_perf[1]]/val_norm,label='Validation')
    plt.ylabel('% Correctly Classified')
    plt.xlabel('max_depth')
    plt.title('Decision Tree Performance for Varying max_depth')
    plt.legend()
    
    #note: optimization seems to vary from run to run
    depth=150 
    feat=0.6 
    
    #---------------------part 7 b)---------------------------------------
    
    '''
    import graphviz
    val_prediction=clf.predict(x_val)
    dot_data = tree.export_graphviz(clf, out_file=None) 
    graph = graphviz.Source(dot_data) 
    graph.render("p7b_2layers")
    '''
    
    print('*** Part 7 finished ***')