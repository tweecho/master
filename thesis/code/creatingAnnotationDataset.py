# -*- coding: utf-8 -*-
"""
Created on Sat May  4 14:22:52 2019

@author: Arlene
1 annotation for doccano
2 json annotation from doccano
3 Flair to text classification
4 logistic regression baseline model
"""

import os
import csv
import random

######################################################
######################################################
##这是用来生成用于annotation的csv的代码
######################################################
######################################################  
def readFile(filename):
    with open(filename,'r',encoding = "utf-8") as f:
        content = f.read()
        content = content.split()
        content = ' '.join(content)
         
        return content
     
#file_dir = 'F://ThesisProject/data//tryAnnotation//'
file_dir = 'F://ThesisProject//data//ann//annotation_txt//'

####################################
#####this is for NER
#annotation_csv = 'F://ThesisProject/data//tryAnnotation//tryAnnotation.tsv'
##write text needs to be annotated into a csv file (for doccano annotation)
#all_txt = [] #this is for NER
#for i in random.sample(os.listdir(file_dir),50):
#    txt =  readFile(file_dir + i)
#    
#    all_txt.append([txt]) 
#    with open(annotation_csv, 'w', encoding = 'utf-8') as f: #encoding='ascii', errors='ignore',
#        csv_writer = csv.writer(f, delimiter='\t', lineterminator='\n')
#        csv_writer.writerows(all_txt)
###################################
####this is for text classification
annotation_csv = 'F://ThesisProject/data//tryAnnotation//tryAnnotation.csv' #this is for text classification
all_txt = [['text', 'label']] 
for i in random.sample(os.listdir(file_dir),19):
    txt =  readFile(file_dir + i)        
    all_txt.append([txt, '1']) 
    with open(annotation_csv, 'w', encoding = 'utf-8') as f: #encoding='ascii', errors='ignore',
        csv_writer = csv.writer(f, delimiter=',', lineterminator='\n')
        csv_writer.writerows(all_txt)



#######################################################
#######################################################
###这是Flair用来生成用于text classiffication的代码
###
###可以adjust hyperparameters
###可以把F-score显示出来
#######################################################
####################################################### 
import pandas as pd

#path = 'F://ThesisProject//data//tryAnnotation1//'
#annotated_csv = 'F://ThesisProject//data//tryAnnotation1//annotation_bernhardtry100.csv' #all_non_5552   all_at_least_one_4280
#all_tit9831 = 'F://ThesisProject//data//tryAnnotation1//all_titAbs_9831_1.csv'
#train = 'F://ThesisProject//data//tryAnnotation1//train.csv'
#test = 'F://ThesisProject//data//tryAnnotation1//test.csv'
#dev = 'F://ThesisProject//data//tryAnnotation1//dev.csv'
##
#data = pd.read_csv(annotated_csv, encoding = 'utf-8')
#data = data[['label', 'text']]
#data['label'] = '__label__' + data['label'].astype(str)
    
#data.iloc[0:int(len(data)*0.8)].to_csv(train, sep=',', index = False, header = False)
#data.iloc[int(len(data)*0.8):int(len(data)*0.9)].to_csv(test, sep=',', index = False, header = False)
#data.iloc[int(len(data)*0.9):].to_csv(dev, sep=',', index = False, header = False)
#    
#from flair.data_fetcher import NLPTaskDataFetcher
#from flair.embeddings import WordEmbeddings, FlairEmbeddings, DocumentLSTMEmbeddings
#from flair.models import TextClassifier
#from flair.trainers import ModelTrainer
#from flair.training_utils import EvaluationMetric
#from pathlib import Path
#
#corpus = NLPTaskDataFetcher.load_classification_corpus(Path(path), test_file='test.csv', dev_file='dev.csv', train_file='train.csv')
#word_embeddings = [WordEmbeddings('glove'), FlairEmbeddings('news-forward-fast'), FlairEmbeddings('news-backward-fast')]
#document_embeddings = DocumentLSTMEmbeddings(word_embeddings, hidden_size=512, reproject_words=True, reproject_words_dimension=256)
#classifier = TextClassifier(document_embeddings, label_dictionary=corpus.make_label_dictionary(), multi_label=False)
#trainer = ModelTrainer(classifier, corpus)
#trainer.train(path, max_epochs=10)
##trainer.train(path, evaluation_metric=EvaluationMetric.MICRO_F1_SCORE, learning_rate=0.1, mini_batch_size=32,anneal_factor=0.5 ,max_epochs=20)
#
##use the trained model to predict the class of new sentences  (use F1-score to see the performance)  
#from flair.models import TextClassifier
#from flair.data import Sentence
#classifier = TextClassifier.load_from_file(path + 'best-model.pt')
#sentence = Sentence("This policy is going to describe",use_tokenizer = True)
#classifier.predict(sentence)
#print(sentence.labels)  
#
#predict_all = []
#for i in range(0,len(data)):
#    sentence = Sentence(data.iloc[i][1])
#    classifier.predict(sentence)
#    predict = '__label__' + str(sentence.labels[0]).split(',')[0]
#    predict_all.append(predict)
#
#from sklearn.metrics import precision_recall_fscore_support, accuracy_score
#import numpy as np
#y_true = np.array(data['label'].values.tolist())
#y_pred = np.array(predict_all)
#precision_recall_fscore_support(y_true, y_pred, average='macro')
#accuracy_score(y_true, y_pred)
#
##from flair.visual.training_curves import Plotter
##plotter = Plotter()
##plotter.plot_training_curves('F://ThesisProject//data//tryAnnotation1//loss.tsv')
##plotter.plot_weights('F://ThesisProject//data//tryAnnotation1//weights.txt')
#





###############################################################################
##########for predicting the 57 articles which has the ftp triplets
#the results shows that all the labels are __label__11 which means they all 
#belong to other group, which is not the truth
#def readFile(filename):
#    with open(filename,'r',encoding = "utf-8") as f:
#        content = f.read()
#        content = content.split()
#        content = ' '.join(content)
#         
#        return content
#
#path = 'F://ThesisProject//data//tryAnnotation1//'
#abstract_path = 'F://ThesisProject//data//biorefinery_txt_abstract_has_ftp//abstracts//'
##use the trained model to predict the class of new sentences  (use F1-score to see the performance)  
#from flair.models import TextClassifier
#from flair.data import Sentence
#classifier = TextClassifier.load_from_file(path + 'best-model.pt')
#
#predict_all = []
#for i in os.listdir(abstract_path):
#    abstract = readFile(abstract_path + i)
#    sentence = Sentence(abstract,use_tokenizer = True)
#    classifier.predict(sentence)
#    predict = '__label__' + str(sentence.labels[0]).split(',')[0]
#    predict_all.append(predict) 

###############################################################################
#abstract_path = 'F://ThesisProject//data//biorefinery_txt_abstract_has_ftp//abstracts//'
##save all 57 abstracts and its title to tsv
#all_57_abstract = []
#for i in os.listdir(abstract_path):
#    abstract = readFile(abstract_path + i)
#    all_57_abstract.append([i, abstract])
#
#abstract_path_tsv = 'F://ThesisProject//data//biorefinery_txt_abstract_has_ftp//abstracts.tsv'
#with open(abstract_path_tsv, 'w', encoding='utf-8', newline='') as f:
#    tsv_writer = csv.writer(f, delimiter='\t', lineterminator='\n')
#    tsv_writer.writerows(all_57_abstract)    










########################################################
########################################################
####这是logistic regression baseline model用来生成用于text classiffication的代码
####每次运行的f score都不一样
###https://heartbeat.fritz.ai/using-transfer-learning-and-pre-trained-language-models-to-classify-spam-549fc0f56c20
########################################################
########################################################
#load the required libraries
#import pandas as pd
#import numpy as np
#from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer
#from sklearn.linear_model.logistic import LogisticRegression
#from sklearn.model_selection import train_test_split, cross_val_score,LeaveOneOut
#from sklearn.metrics import accuracy_score, f1_score
#from sklearn.metrics import precision_recall_fscore_support
#from sklearn import svm
#
#annotated_csv = 'F://ThesisProject//data//tryAnnotation1//annotation_bernhard.csv' #all_non_5552  annotation_bernhard
#test_csv = 'F://ThesisProject//data//tryAnnotation1//all_non_5552.csv'
#
##read the data
#data = pd.read_csv(annotated_csv, encoding = 'utf-8')#.sample(frac=1)   
#data = data[['label', 'text']]
#
##df = pd.read_csv('SMSSpamCollection.txt', delimiter='\t',header=None)
##df.rename(columns = {0:'label',1: 'text'}, inplace = True)
##Input and output variables
#X = data['text']
#y = data['label']
#
#measurement_scores = [['accuracy', 'precision','recall', 'fscore']]
##for i in range(10):
#seed = 5
#test_size = 0.2
##split dataset into train and test sets
#X_train_raw, X_test_raw, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=seed)
##Convert to a matrix of TF-IDF features
#vectorizer = TfidfVectorizer()
#X_train = vectorizer.fit_transform(X_train_raw)
#X_test = vectorizer.transform(X_test_raw)
#
##Model training
#classifier = LogisticRegression(max_iter=100)
#classifier.fit(X_train, y_train)
##prediction
#predictions = classifier.predict(X_test)
##model evaluation
#score = accuracy_score(y_test, predictions)
##f_score = f1_score(y_test, predictions, average='micro')
##print("The accuracy score (Logistic Regression) is:", score)
##print("The F score-Micro (Logistic Regression) is:", f_score)
#
#z = precision_recall_fscore_support(y_test, predictions, average='macro')
#scores = [score, z[0], z[1], z[2]]
#
#print(scores)
 





#############n-fold cross-validation
#annotated_csv = 'F://ThesisProject//data//tryAnnotation1//annotation_bernhard.csv' #all_non_5552  annotation_bernhard
#test_csv = 'F://ThesisProject//data//tryAnnotation1//all_non_5552.csv'
#
##read the data
#data = pd.read_csv(annotated_csv, encoding = 'utf-8')#.sample(frac=1)   
#data = data[['label', 'text']]
#
##df = pd.read_csv('SMSSpamCollection.txt', delimiter='\t',header=None)
##df.rename(columns = {0:'label',1: 'text'}, inplace = True)
##Input and output variables
#X = data['text']
#y = data['label']
#
#measurement_scores = [['accuracy', 'precision','recall', 'fscore']]
##for i in range(10):
##seed = 5
##test_size = 0.2
###split dataset into train and test sets
##X_train_raw, X_test_raw, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=seed)
###Convert to a matrix of TF-IDF features
#vectorizer = CountVectorizer() #TfidfVectorizer()
##X_train = vectorizer.fit_transform(X_train_raw)
##X_test = vectorizer.transform(X_test_raw)
##
#X_vec = vectorizer.fit_transform(X)
#
##Model training
#classifier = LogisticRegression(max_iter=100) #svm.SVC(kernel='linear')
##classifier.fit(X_train, y_train)
#
#from sklearn import metrics
#import sklearn
#predicted = sklearn.model_selection.cross_val_predict(classifier, X_vec, y, cv=5)
#metrics.accuracy_score(y, predicted) 
#
#accuracy = cross_val_score(classifier, X_vec, y, cv=5,scoring='accuracy')
#print (accuracy)
#print (cross_val_score(classifier, X_vec, y, cv=5,scoring='accuracy').mean())
#
#from nltk import ConfusionMatrix 
#print (ConfusionMatrix(list(y), list(predicted)))
##print (ConfusionMatrix(list(y), list(yexpert)))
#
#y_true = np.array(list(y))
#y_pred = np.array(list(predicted))
#precision_recall_fscore_support(y_true, y_pred, average='micro')
#accuracy_score(y_true, y_pred)





################Leave one out cross validation
#annotated_csv = 'F://ThesisProject//data//tryAnnotation1//annotation_bernhard.csv' #all_non_5552  annotation_bernhard
#test_csv = 'F://ThesisProject//data//tryAnnotation1//all_non_5552.csv'
#
#all_tit9831 = 'F://ThesisProject//data//tryAnnotation1//all_titAbs_9831_1.csv'
#
##data for classifying 
#data_all9831 = pd.read_csv(all_tit9831, encoding="ISO-8859-1")#.sample(frac=1)   
#data_all9831 = data_all9831[['label', 'text']]
#
##read the data
#data = pd.read_csv(annotated_csv, encoding = 'utf-8')#.sample(frac=1)   
#data = data[['label', 'text']]
#
##df = pd.read_csv('SMSSpamCollection.txt', delimiter='\t',header=None)
##df.rename(columns = {0:'label',1: 'text'}, inplace = True)
##Input and output variables
#X = data['text']
#y = data['label']
#
#measurement_scores = [['accuracy', 'precision','recall', 'fscore']]
##for i in range(10):
#seed = 5
#test_size = 0.2
##split dataset into train and test sets
##X_train_raw, X_test_raw, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=seed)
##Convert to a matrix of TF-IDF features
#vectorizer = CountVectorizer() #TfidfVectorizer()
##X_train = vectorizer.fit_transform(X_train_raw)
##X_test = vectorizer.transform(X_test_raw)
#
#X_vec = vectorizer.fit_transform(X)
#
#loo_c = LeaveOneOut()
#loo = loo_c.split(X_vec)
#
#ytests = []
#ypreds = []
#
#for train_index, test_index in loo:
#    X_train, X_test = X_vec[train_index], X_vec[test_index]
#    y_train, y_test = y[train_index], y[test_index]
#    
#    #Model training
#    classifier = svm.SVC(kernel='linear')     #LogisticRegression(max_iter=100)
#    classifier.fit(X_train, y_train)
#
#    y_pred = classifier.predict(X_test)
#    
#    ytests += list(y_test)
#    ypreds += list(y_pred)
#
#from nltk import ConfusionMatrix 
#print(ConfusionMatrix(list(ytests), list(ypreds)))
#
#z = precision_recall_fscore_support(ytests, ypreds, average='micro')
#
#y_true = np.array(list(ytests))
#y_pred = np.array(list(ypreds))
#precision_recall_fscore_support(y_true, y_pred, average='micro')
#accuracy_score(y_true, y_pred)
#
#import matplotlib.pyplot as plt
#from sklearn.metrics import confusion_matrix
#from sklearn.utils.multiclass import unique_labels
#
#
#
#
#
#df_confusion = pd.crosstab(y_true, y_pred)
#
#plot_confusion_matrix(df_confusion)


#annotated_csv = 'F://ThesisProject//data//tryAnnotation1//annotation_bernhard.csv' #all_non_5552  annotation_bernhard
#test_csv = 'F://ThesisProject//data//tryAnnotation1//all_non_5552.csv'
#
#all_tit9831 = 'F://ThesisProject//data//tryAnnotation1//all_titAbs_9831_1.csv'
#
##data for classifying 
#data_all9831 = pd.read_csv(all_tit9831, encoding="ISO-8859-1")#.sample(frac=1)   
#data_all9831 = data_all9831[['label', 'text']]
#
##read the data
#data = pd.read_csv(annotated_csv, encoding = 'utf-8')#.sample(frac=1)   
#data = data[['label', 'text']]
#
#X_all9831 = data_all9831['text']
#y_all9831 = data_all9831['label']
#
##df = pd.read_csv('SMSSpamCollection.txt', delimiter='\t',header=None)
##df.rename(columns = {0:'label',1: 'text'}, inplace = True)
##Input and output variables
#X = data['text']
#y = data['label']
#
#measurement_scores = [['accuracy', 'precision','recall', 'fscore']]
##for i in range(10):
#seed = 5
#test_size = 0.2
##split dataset into train and test sets
##X_train_raw, X_test_raw, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=seed)
##Convert to a matrix of TF-IDF features
#vectorizer = CountVectorizer() #TfidfVectorizer()
##X_train = vectorizer.fit_transform(X_train_raw)
##X_test = vectorizer.transform(X_test_raw)
#
#X_vec = vectorizer.fit_transform(X_all9831)
#
#
#loo_c = LeaveOneOut()
#loo = loo_c.split(X_vec[0:122])
#
#ytests = []
#ypreds = []
#
#for train_index, test_index in loo:
#    X_train, X_test = X_vec[train_index], X_vec[test_index]
#    y_train, y_test = y[train_index], y[test_index]
#    
#    #Model training
#    classifier = svm.SVC(kernel='linear')  #svm.SVC(kernel='linear')     #LogisticRegression(max_iter=100)
#    classifier.fit(X_train, y_train)
#
#    y_pred = classifier.predict(X_test)
#    
#    ytests += list(y_test)
#    ypreds += list(y_pred)
#
#from nltk import ConfusionMatrix 
#print(ConfusionMatrix(list(ytests), list(ypreds)))
#
#z = precision_recall_fscore_support(ytests, ypreds, average='micro')
#
#y_true = np.array(list(ytests))
#y_pred = np.array(list(ypreds))
#precision_recall_fscore_support(y_true, y_pred, average='micro')
#accuracy_score(y_true, y_pred)
#
#
#y_pred_all9831 = classifier.predict(X_vec[122:])
#
#count = 0
#for i in y_pred_all9831:
#    if i == 1:
#        count += 1
#
#
#
#count = 0
#for i in y_pred_all9831:
#    if i == 1:
#        count += 1
#        
#        
#        
#count_least1 = 0
#for i in y_pred_all9831[0:4281]:
#    if i == 1:
#        count_least1 += 1
#
#
#count_none = 0
#for i in y_pred_all9831[4281:]:
#    if i == 1:
#        count_none += 1
##




















###########ULMFiT classification   
###############################################################################ULMFiT classification
#import fastai
#from fastai import * 
#from fastai.text import * 
#import pandas as pd
#import numpy as np
#from functools import partial
##from fastai.utils.show_install import *
##show_install()
#import io
#import os
#import nltk
#nltk.download('stopwords')
#
#from nltk.corpus import stopwords 
#stop_words = stopwords.words('english')
#
#
#annotated_csv = 'F://ThesisProject//data//tryAnnotation1//annotation_bernhard.csv' #all_non_5552  annotation_bernhard
#test_csv = 'F://ThesisProject//data//tryAnnotation1//all_non_5552.csv'
#all_tit9831 = 'F://ThesisProject//data//tryAnnotation1//all_titAbs_9831_1.csv'
#
##data for classifying 
#data_all9831 = pd.read_csv(all_tit9831, encoding="ISO-8859-1")#.sample(frac=1)   
#data_all9831 = data_all9831[['label', 'text']]
#X_all9831 = data_all9831['text']
#y_all9831 = data_all9831['label']
#
#data = pd.read_csv(annotated_csv, encoding = 'utf-8')#.sample(frac=1)   
#data = data[['label', 'text']]
#
#data['text'] = data['text'].str.replace("[^a-zA-Z]", " ")
#
#tokenized_doc = data['text'].apply(lambda x: x.split())
#tokenized_doc = tokenized_doc.apply(lambda x: [item for item in x if item not in stop_words])
#
#detokenized_doc = [] 
#for i in range(len(data)): 
#    t = ' '.join(tokenized_doc[i]) 
#    detokenized_doc.append(t) 
#
#data['text'] = detokenized_doc
#
#
#from sklearn.model_selection import train_test_split
#
## split data into training and validation set
#df_trn, df_val = train_test_split(data, stratify = data['label'], test_size = 0.2, random_state = 12)
#
#defaults.cpus=1
#
## Language model data
#data_lm = TextLMDataBunch.from_df(train_df = df_trn, valid_df = df_val, path = all_tit9831, num_workers=0)
#
#learn = language_model_learner(data_lm, AWD_LSTM, drop_mult=0.5)
#learn.fit_one_cycle(1, 1e-2)
#
#learn.unfreeze()
#learn.fit_one_cycle(1, 1e-3)
#
#learn.save_encoder('ft_enc')
#
#
#
## Classifier model data
#data_clas = TextClasDataBunch.from_df(path = all_tit9831, train_df = df_trn, valid_df = df_val, vocab=data_lm.train_ds.vocab, bs=32,num_workers=0)
#
#learnC = text_classifier_learner(data_clas,AWD_LSTM, drop_mult=0.5, metrics=[accuracy,
#               Precision(average='macro'),
#               Recall(average='macro'),
#               FBeta(average='macro')])
#learnC.load_encoder('ft_enc')
#
#learnC.fit_one_cycle(1, 1e-2)
#
#learnC.freeze_to(-2)
#learnC.fit_one_cycle(1, slice(5e-3/2., 5e-3))
#
#learnC.unfreeze()
#learnC.fit_one_cycle(1, slice(2e-3/100, 2e-3))
#
## get predictions
#preds, targets = learnC.get_preds()
#
#predictions = np.argmax(preds, axis = 1)
#pd.crosstab(predictions, targets)
#
#data_clas.show_batch()
#
#y_true = np.array(list(targets))
#y_pred = np.array(list(predictions))
#precision_recall_fscore_support(y_true, y_pred, average='macro')
#accuracy_score(y_true, y_pred)


















