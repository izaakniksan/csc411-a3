import csv
import nltk
import os
import random
import collections
import pickle

'''
This file splits the datasets up into training, validation, and test sets and
counts the number of times each word appears, recording them in dictionaries.
Then, the dictionaries are dumped to pickle files.
'''
fake_file = open('clean_fake.txt')
real_file = open('clean_real.txt')

real_lines=[]
fake_lines=[]

for line in real_file:
    line=line.rstrip('\n')
    real_lines.append(line)
for line in fake_file:
    line=line.rstrip('\n')
    fake_lines.append(line)

    
#Randomly shuffle all the lines
random.shuffle(real_lines)
random.shuffle(fake_lines)

#Put the lines into three sets: test 70%, val 15%, test 15%
real_train_lines=real_lines[0:int(len(real_lines)*0.7)]
real_val_lines=real_lines[int(len(real_lines)*0.7):int(len(real_lines)*0.85)]
real_test_lines=real_lines[int(len(real_lines)*0.85):]

fake_train_lines=fake_lines[0:int(len(fake_lines)*0.7)]
fake_val_lines=fake_lines[int(len(fake_lines)*0.7):int(len(fake_lines)*0.85)]
fake_test_lines=fake_lines[int(len(fake_lines)*0.85):]

#Save the headline counts in pickle files
counts={'real_train':len(real_train_lines), 'real_val':len(real_val_lines),\
        'real_test':len(real_test_lines), 'fake_train':len(fake_train_lines), \
        'fake_val':len(fake_val_lines), 'fake_test':len(fake_test_lines)}
with open('counts.pickle', 'wb') as handle:
    pickle.dump(counts, handle, protocol=pickle.HIGHEST_PROTOCOL)

#Create dictionaries with the number of times each word appears
words=[]
for line in real_train_lines:
    temp=line.split(' ')
    unique=[]
    [unique.append(item) for item in temp if item not in unique]
    words.extend(unique)
real_train=collections.Counter(words)

words=[]
for line in real_val_lines:
    temp=line.split(' ')
    unique=[]
    [unique.append(item) for item in temp if item not in unique]
    words.extend(unique)
real_val=collections.Counter(words)

words=[]
for line in real_test_lines:
    temp=line.split(' ')
    unique=[]
    [unique.append(item) for item in temp if item not in unique]
    words.extend(unique)
real_test=collections.Counter(words)

words=[]
for line in fake_train_lines:
    temp=line.split(' ')
    unique=[]
    [unique.append(item) for item in temp if item not in unique]
    words.extend(unique)
fake_train=collections.Counter(words)

words=[]
for line in fake_val_lines:
    temp=line.split(' ')
    unique=[]
    [unique.append(item) for item in temp if item not in unique]
    words.extend(unique)
fake_val=collections.Counter(words)

words=[]
for line in fake_test_lines:
    temp=line.split(' ')
    unique=[]
    [unique.append(item) for item in temp if item not in unique]
    words.extend(unique)
fake_test=collections.Counter(words)

with open('real_train.pickle', 'wb') as handle:
    pickle.dump(real_train, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
with open('real_val.pickle', 'wb') as handle:
    pickle.dump(real_val, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
with open('real_test.pickle', 'wb') as handle:
    pickle.dump(real_test, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
with open('fake_train.pickle', 'wb') as handle:
    pickle.dump(fake_train, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
with open('fake_val.pickle', 'wb') as handle:
    pickle.dump(fake_val, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
with open('fake_test.pickle', 'wb') as handle:
    pickle.dump(fake_test, handle, protocol=pickle.HIGHEST_PROTOCOL)