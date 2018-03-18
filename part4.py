import csv
import nltk
import os
import random
import collections
import pickle
from math import *
import part2 as p2
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
import torch 
from numpy import *
from torch.autograd import Variable
import matplotlib.pyplot as plt

MAX_ITERS = 10000

def create_v(set_words,all_words):
    '''
    Create input and expected output vectors for logistic regression. 
    set_words is array of cleaned headlines (which are separated into words)
    count=m
    len(all_words)=n
    all_words is all the words that appear in the trainingset
    '''
    #all_words=real_train.keys()
    
    #creating input vector and output vectors
    #v is input: nxm
 
    count=len(set_words)
    #create input:
    v=zeros((len(all_words),count)) 
    i=0
    for headline in set_words:
        for j in range(0,len(all_words)):
            if all_words[j] in headline:
                v[j][i]=1
            else:
                v[j][i]=0
        i=i+1
    return v

def clean_headline(headline,_set):
    #takes in a headline and removed duplicated words and any words not found 
    #in the set. Outputs a list of these words.
    line=headline
    line=line.rstrip('\n')
    temp=line.split(' ') 
    
    #remove any duplicated words in the headline:
    h_words=[] # h_words contains all the words in the headline
    [h_words.append(item) for item in temp if item not in h_words]

    #remove any words in the headline which are not found in the training set:
    [h_words.remove(word) for word in h_words if word not in _set]
    return h_words

def softmax(x,y,w):
    #compute o:
    temp_o = dot(w.T, x) #o
    #then compute p:
    p= exp(temp_o)/tile(sum(exp(temp_o),0), (len(temp_o),1)) #softmax
    return p

def logcost(x,y,w,_lambda):
    #Given x = input, y=expected output, and w, compute the log cost by first
    #computing o, then passing it through softmax, then outputting the reg cost
    
    #compute o:
    temp_o = dot(w.T, x) #o
    #then compute p:
    temp_p = exp(temp_o)/tile(sum(exp(temp_o),0), (len(temp_o),1)) #softmax 
    c = -(y*log(temp_p)) + _lambda*linalg.norm(w) 
    return sum(c) #scalar

def vector_grad(x,y,w,_lambda):
    """
    Usin the reg logcost function and the weights, return a 2D matrix of the 
    gradient, where the izth element is di C/di wiz. This matrix is nxj. 
    Since we sum over all images for the cost function, we sum over all
    gradient matrices for the final output matrix.
    """
    #compute o:
    temp_o = dot(w.T, x)
    #then compute p:
    temp_p = exp(temp_o)/tile(sum(exp(temp_o),0), (len(temp_o),1)) #softmax
    return dot(x,(temp_p - y).T) + 2*_lambda*w
    
def grad_descent_curves(vector_grad,logcost, softmax,v, y, v_val,y_val, init_w, alpha,_lambda):
    """Gradient Descent function that also returns arrays to plot performance. """
    firsttime=1
    performance_array=zeros((1,1))
    val_performance_array=zeros((1,1))
    iteration_array=zeros((1,1))
    EPS = 1e-5  # EPS = 10**(-5)
    prev_w = init_w - 10 * EPS
    w = init_w.copy()
    iter = 0
    while linalg.norm(w - prev_w) > EPS and iter < MAX_ITERS:
        prev_w = w.copy()
        deriv = vector_grad(v,y,w,_lambda)
        w -= alpha * deriv
        if iter % 50 == 0:
            if firsttime==1:
                #compute o:
                temp_o = dot(w.T, v) #o
                #then compute p:
                temp_p = exp(temp_o)/tile(sum(exp(temp_o),0), (len(temp_o),1)) #softmax
                temp_perf=0
                for k in range(0,len(y_train[0,:])):
                    if argmax(y_train[:,k])==argmax(temp_p[:,k]):
                        temp_perf=temp_perf+1
                temp_perf=100*temp_perf/len(y_train[0,:])
                performance_array[0]=temp_perf
               

                temp_o = dot(w.T, v_val) #o
                temp_p = exp(temp_o)/tile(sum(exp(temp_o),0), (len(temp_o),1)) #softmax
                temp_perf_val=0
                for k in range(0,len(y_val[0,:])):
                    if argmax(y_val[:,k])==argmax(temp_p[:,k]):
                        temp_perf_val=temp_perf_val+1
                temp_perf_val=100*temp_perf_val/len(y_val[0,:])
                val_performance_array[0]=temp_perf_val
                
                
                iteration_array[0]=iter
                firsttime=0
            else:
                #compute o:
                temp_o = dot(w.T, v) #o
                #then compute p:
                temp_p = exp(temp_o)/tile(sum(exp(temp_o),0), (len(temp_o),1)) #softmax
                temp_perf=0
                for k in range(0,len(y_train[0,:])):
                    if argmax(y_train[:,k])==argmax(temp_p[:,k]):
                        temp_perf=temp_perf+1
                temp_perf=100*temp_perf/len(y_train[0,:])
                performance_array=append(performance_array,temp_perf)
                
                temp_o = dot(w.T, v_val) #o
                temp_p = exp(temp_o)/tile(sum(exp(temp_o),0), (len(temp_o),1)) #softmax
                temp_perf_val=0
                for k in range(0,len(y_val[0,:])):
                    if argmax(y_val[:,k])==argmax(temp_p[:,k]):
                        temp_perf_val=temp_perf_val+1
                temp_perf_val=100*temp_perf_val/len(y_val[0,:])
                val_performance_array=append(val_performance_array,temp_perf_val)
                
                iteration_array=append(iteration_array,iter)
            print('Iters:  ', iter,'Training %: ',temp_perf, "Val %",temp_perf_val)
        iter += 1
    return w,iteration_array,performance_array, val_performance_array

if __name__ == "__main__":
    print('*** Part 4 running ***')
    
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
    #First create trainingset
    trainingset=append(real_train_lines,fake_train_lines)

    #Make arrays of words for trainingset
    trainingset_words=[]
    for i in range(0,len(trainingset)):
        trainingset_words.append(clean_headline(trainingset[i],trainingset))

    #Create input vector v: nxm=len(all_words) x count

    v_train=create_v(trainingset_words,all_words)
    v_train = vstack((ones((1, v_train.shape[1])), v_train))
    
    #Create output vector y
    #y is output: jxm=2xm = #possible outputs (real or fake) x #examples 
    y_train=zeros((len(trainingset_words),2))
    for i in range(0,len(real_train_lines)):
        y_train[i][0]=1 #real
    for i in range(len(real_train_lines),len(trainingset_words)):
        y_train[i][1]=1 #fake

    y_train=y_train.T #used because I did loop indices wrong
    
    #Create validation set
    validationset=append(real_val_lines,fake_val_lines)
    
    #Make arrays of words for validation set
    validationset_words=[]
    for i in range(0,len(validationset)):
        validationset_words.append(clean_headline(validationset[i],trainingset))
        
    #Create v_val
    
    v_val=create_v(validationset_words,all_words)
    v_val = vstack((ones((1, v_val.shape[1])), v_val))
    
    #Create output vector y_val
    y_val=zeros((len(validationset_words),2))
    for i in range(0,len(real_val_lines)):
        y_val[i][0]=1 #real
    for i in range(len(real_val_lines),len(validationset_words)):
        y_val[i][1]=1 #fake
        
    y_val=y_val.T
    
    #Now create testset
    testset=append(real_test_lines,fake_test_lines)
    
    #Make arrays of words for testset
    testset_words=[]
    for i in range(0,len(testset)):
        testset_words.append(clean_headline(testset[i],trainingset))
    #note: the testset uses the trainingset here since we need to make sure
    #all words in the testset can be fed into the model
        
    #Create output vector y for testset
    y_test=zeros((len(testset_words),2))
    for i in range(0,len(real_test_lines)):
        y_test[i][0]=1 #real
    for i in range(len(real_test_lines),len(testset_words)):
        y_test[i][1]=1 #fake
    
    y_test=y_test.T
    
#-------------RUN GRADIENT DESCENT----------------
    
#   w is nxj=nx2
    init_w=random.rand(len(all_words)+1,2)
    optimal_w,iteration_array,performance_array, val_performance_array = \
        grad_descent_curves(vector_grad,logcost, softmax, v_train, y_train, \
                            v_val,y_val,init_w, alpha,_lambda)
    
#----------------------PLOT-----------------------
    
    plt.plot(iteration_array,performance_array, label='Training')
    plt.plot(iteration_array,val_performance_array,label='Validation')
    plt.ylabel('% Correctly Classified')
    plt.xlabel('Iterations')
    plt.title('Gradient Descent Learning Curve')
    plt.legend()
            
    print('*** part 4 finished ***\n')
        
        