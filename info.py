from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from collections import Counter
from num2words import num2words

import nltk
import os
import string
import numpy as np
import copy
import pandas as pd
import pickle
import re
import math
import operator
from nltk import tokenize
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
import os
import nltk
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem.porter import PorterStemmer
import re
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from matplotlib import pyplot as plt
from pandas import DataFrame
import tkinter as tk
from tkinter import filedialog as fd
from tkinter import messagebox
from tkinter import Text
import matplotlib.pyplot as plt
from tkinter import Scrollbar
from tkinter import *
from scalable_support_vector_clustering import ScalableSupportVectorClustering


def tokenize(text):
    tokens = nltk.word_tokenize(text)
    stems = []
    for item in tokens:
        if len(item)>3:
            stems.append(PorterStemmer().stem(item))
    return stems


def convert_lower_case(data):
    return np.char.lower(data)

def remove_stop_words(data):
    stop_words = stopwords.words('english')
    words = word_tokenize(str(data))
    new_text = ""
    for w in words:
        if w not in stop_words and len(w) > 1:
            new_text = new_text + " " + w
    return new_text

def remove_punctuation(data):
    symbols = "!\"#$%&()*+-./:;<=>?@[\]^_`{|}~\n"
    for i in range(len(symbols)):
        data = np.char.replace(data, symbols[i], ' ')
        data = np.char.replace(data, "  ", " ")
    data = np.char.replace(data, ',', '')
    return data


def remove_apostrophe(data):
    return np.char.replace(data, "'", "")

def stemming(data):
    stemmer= PorterStemmer()
    
    tokens = word_tokenize(str(data))
    new_text = ""
    for w in tokens:
        new_text = new_text + " " + stemmer.stem(w)
    return new_text

def convert_numbers(data):
    tokens = word_tokenize(str(data))
    new_text = ""
    for w in tokens:
        try:
            w = num2words(int(w))
        except:
            a = 0
        new_text = new_text + " " + w
    new_text = np.char.replace(new_text, "-", " ")
    return new_text

def preprocess(data):
    data = convert_lower_case(data)
    data = remove_punctuation(data) #remove comma seperately
    data = remove_apostrophe(data)
    data = remove_stop_words(data)
    data = convert_numbers(data)
    data = stemming(data)
    data = remove_punctuation(data)
    data = convert_numbers(data)
    data = stemming(data) #needed again as we need to stem the words
    data = remove_punctuation(data) #needed again as num2word is giving few hypens and commas fourty-one
    data = remove_stop_words(data) #needed again as num2word is giving stop words 101 - one hundred and one
    return data

def buildKB():
    global tfidf1,tfs1,processed_title
    print('started building the knowledge base')
    directory='./knowbase'
    processed_text = []
    processed_title = []
    for filename in os.listdir(directory):
        if filename.endswith(".txt"):
            fullname=os.path.join(directory, filename)
            print('Processing file ',fullname)
            f = open(fullname,"r")
            text1 = f.read()
            prepr=preprocess(text1)
            processed_text.append(prepr)
            processed_title.append(filename)

    tfidf1 = TfidfVectorizer()
    
    tfs1 = tfidf1.fit_transform(processed_text)
    print('!!!!!!!Feature vector is writen to a csv file');
    dfReviews = DataFrame(tfs1.A, columns=tfidf1.get_feature_names())
    dfReviews = dfReviews.astype(float)
    dfReviews.to_csv("fv.csv")
    print('knowledge base is built')
    

def getInfo():
    global e1,tfidf1,tfs1,processed_title,rlabel,listbox
    query=e1.get()
    print('Trying to look up for the query:',query);
    prepr=preprocess(query)
    print('After preprocessing ',prepr)
    qs=[]
    qs.append(prepr)
    
    #tfs2 = tfidf1.transform(qs)
    #cosim = cosine_similarity(tfs2,tfs1)
    #col=cosim.shape[1]
    #key_value ={}
    #nummatch=0
    #for j in range(col):
    #    if cosim[0][j]!=0:
    #        key_value[processed_title[j]]=cosim[0][j]
    #        nummatch=nummatch+1

    ssvc = ScalableSupportVectorClustering()
    key_value=ssvc.getMatchCluster(tfidf1,tfs1,processed_title,qs)
    nummatch=len(key_value)
    if nummatch>0:
        sorted_d = dict(sorted(key_value.items(), key=operator.itemgetter(1),reverse=True))
    
        print(sorted_d)

        sth= " Matching results in order of highest relevance "

        rlabel.config(text=sth)
        k=1
        for i in sorted_d:
            listbox.insert(k,i)
            k=k+1
    else:
        sth= " No Matching results pls update the knowledge base "
        rlabel.config(text=sth)

def fitnessfunction(X,Y):
    X=X.lower()
    Y=Y.lower()
    X_list = word_tokenize(X)  
    Y_list = word_tokenize(Y) 
      
    # sw contains the list of stopwords 
    sw = stopwords.words('english')  
    l1 =[];l2 =[] 
      
    # remove stop words from string 
    X_set = {w for w in X_list if not w in sw}  
    Y_set = {w for w in Y_list if not w in sw} 
      
    # form a set containing keywords of both strings  
    rvector = X_set.union(Y_set)  
    for w in rvector: 
        if w in X_set: l1.append(1) # create a vector 
        else: l1.append(0) 
        if w in Y_set: l2.append(1) 
        else: l2.append(0) 
    c = 0
      
    # cosine formula  
    for i in range(len(rvector)): 
            c+= l1[i]*l2[i] 
    cosine = c / float((sum(l1)*sum(l2))**0.5)
    return cosine

def viewDetail():
    global listbox,e1,answerlabel
    query=e1.get()
    value=str((listbox.get(ACTIVE)))
    print('Selected item is ',value)
    directory='./knowbase'
    fullname=os.path.join(directory, value)
    print('Processing file ',fullname)
    f = open(fullname,"r")
    text1 = f.read()
    t1tok=sent_tokenize(text1)
    numsen=len(t1tok)
    print('total sentences ',numsen)
    bestparticle=0;
    maxfit=0
    for i in range(numsen):
        fit=fitnessfunction(query,t1tok[i])
        if fit>maxfit:
            maxfit=fit
            bestparticle=i

    answerlabel.config(text=t1tok[bestparticle])
    
if __name__ == '__main__':
    global e1,rlabel,listbox,answerlabel
    buildKB()
    parent = tk.Tk()
    parent.title("Information retrieval")
    frame = tk.Frame(parent)
    frame.pack()

    w = tk.Label(frame, text="ML based information retrieval",
                     fg = "red",
                     font = "Times")
    w.pack()

    
    w=tk.Label(frame, 
             text="Query")
    w.pack()
    e1 = tk.Entry(frame)
    e1.pack()

    text_disp= tk.Button(frame, 
                       text="GET INFO", 
                       command=getInfo
                       )
    text_disp.pack()

    rlabel = tk.Label(frame, fg="green")
    rlabel.pack()

    listbox = Listbox(frame)
    listbox.pack()


    text_disp= tk.Button(frame, 
                       text="VIEW DETAIL", 
                       command=viewDetail
                       )
    text_disp.pack()

    answerlabel = tk.Label(frame, fg="blue")
    answerlabel.pack()
    
    
    parent.mainloop()
    
    





    

