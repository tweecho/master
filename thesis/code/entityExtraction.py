# -*- coding: utf-8 -*-
"""
Created on Wed Feb 27 18:06:44 2019
@author: Arlene
"""
import nltk
import re
from nltk.corpus import stopwords #153
#from spacy.en.language_data import STOP_WORDS #307
from sklearn.feature_extraction.stop_words import ENGLISH_STOP_WORDS #318
from nltk.tokenize import word_tokenize, sent_tokenize
import math
import os 
from nltk.stem import WordNetLemmatizer
wordnet_lemmatizer = WordNetLemmatizer()
import os
import spotlight
import time
from nltk.stem import WordNetLemmatizer
from nltk.chunk import conlltags2tree, tree2conlltags
import pandas as pd #for data handling
import nltk
import random
import shutil
import csv
from collections import Counter
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from sklearn.decomposition import PCA
import gensim
from gensim.models import Word2Vec
import gensim.models.keyedvectors as word2vec
import re  # For preprocessing
from collections import defaultdict  # For word frequency
import spacy  # For preprocessing
import gensim.downloader as api
from numpy  import array
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from gensim.test.utils import datapath

stop_words = set(stopwords.words('english'))
tokenizer = RegexpTokenizer(r'\w+')


def returnAllWords(text):
    stop_words = set(stopwords.words('english')) 
    match = re.findall('[A-Za-z][A-Za-z0-9]*(?:-[A-Za-z0-9]+)*', text) #('[0-9\w]+', text)
    #starting and ending with a letter, accepting only letters, numbers and -
    
    new_match = [x for x in match if not x.isdigit()]
    new_match = [x.lower() for x in new_match]
    
    filtered_match = [x for x in new_match if not x in stop_words]
    filtered_match = [x for x in filtered_match if len(x) > 1]
    filtered_match = [wordnet_lemmatizer.lemmatize(x, pos="v") for x in filtered_match]
    
    return filtered_match

def computeTF(wordDict, bow): #https://github.com/mayank408/TFIDF/blob/master/TFIDF.ipynb
    tfDict = {}
    bowCount = len(bow)
    for word, count in wordDict.items():
        tfDict[word] = count/float(bowCount)
    return tfDict  

def computeIDF(docList):
    idfDict = {}
    N = len(docList)
    
    idfDict = dict.fromkeys(docList[0].keys(), 0)
    for doc in docList:
        for word, val in doc.items():
            if val > 0:
                idfDict[word] += 1
    
    for word, val in idfDict.items():
        idfDict[word] = math.log10(N / float(val))
        
    return idfDict

def computeTFIDF(tfBow, idfs):
    tfidf = {}
    for word, val in tfBow.items():
        tfidf[word] = val*idfs[word]
    return tfidf 



def readFile(filename):
    with open(filename,'r',encoding = "utf-8") as f:
        content = f.read()
        content = content.split()
        content = ' '.join(content)
         
        return content

#seperate abstract into a list of sentences
def absToSentence(abstract):
    sentences = []
    stop_words = set(stopwords.words('english'))
    tokenizer = RegexpTokenizer(r'\w+')
    
    abstract_senDict = nltk.sent_tokenize(abstract)
    
    for sent in abstract_senDict:
        #abstract_word = nltk.word_tokenize(sent)
        abstract_word = tokenizer.tokenize(sent)
        abstract_word_filtered = []
        for i in abstract_word:
            if i not in stop_words and i.isnumeric() == False:
                abstract_word_filtered.append(i)
        abstract_word = ' '.join( word.lower() for word in abstract_word_filtered)
        sentences.append(abstract_word)
      
    return sentences

def generate_ngrams(words_list, n):
    ngrams_list = []
    ngram = []
    
    for num in range(0, len(words_list) - n + 1):
        ngram = words_list[num:num+n]
        ngrams_list.append(ngram)
        
    return ngrams_list 


def existedEntityStrToList(entity_list):
    entity_lists = []

    for i in entity_list:
        i = nltk.word_tokenize(i)
        entity_lists.append(i)
    
    return entity_lists

#match all existed entities in list to sentences in articles(entities that co-occurs in the sentence)
def matchEntity(sentence, entity_list):
    entity_recognized = []
    
    entity = []
    sentence_unigram_str = sentence.split()
    sentence_unigram = [[i] for i in sentence_unigram_str]
    sentence_bigram = generate_ngrams(sentence_unigram_str,2)
    sentence_trigram = generate_ngrams(sentence_unigram_str,3)
    for i in entity_list:
        if i in sentence_unigram:
            i = ' '.join(m for m in i)
            entity.append(i)
        elif i in sentence_bigram:
            i = ' '.join(j for j in i)
            entity.append(i)
        elif i in sentence_trigram:
            i = ' '.join(k for k in i)
            entity.append(i)
        else:
            pass
    if entity != [] and len(entity) >= 2:
        entity_recognized.append(entity)
    
    return entity_recognized

def generatCatogoryEntityList(annotated_entity_csv_path): #csv file contains (category entity) for each row

    ann_entity_pd = pd.read_csv(annotated_entity_csv_path, header = None)
    ann_entity = ann_entity_pd[0].values.tolist()
    ann_entity_2 = ann_entity_pd[1].values.tolist()
    ann_entity_map = list(list(i) for i in zip(ann_entity, ann_entity_2))

    #return ['product', 'biogas'],['feedstock', 'green crops'], etc.
    return ann_entity_map

def getCategoryEntitylist(ann_entity_map): # list of lists like ['feedstock', 'green crops']
    technology = []
    product = []
    feedstock = []
    for i in range(len(ann_entity_map)): 
        if ann_entity_map[i][0] == 'technology':
            technology.append(('t', ann_entity_map[i][1]))
        elif ann_entity_map[i][0] == 'product':
            product.append(('p', ann_entity_map[i][1]))
        else:
            feedstock.append(('f', ann_entity_map[i][1]))
    
    df_t = pd.DataFrame(columns=['technology'])
    df_t['technology'] = [technology[i][1] for i in range(len(technology))]
    df_p = pd.DataFrame(columns=['product'])
    df_p['product'] = [product[i][1] for i in range(len(product))]
    df_f = pd.DataFrame(columns=['feedstock'])
    df_f['feedstock'] = [feedstock[i][1] for i in range(len(feedstock))]
    
    df_all = pd.concat([df_f, df_t,df_p], ignore_index=True, axis=1)
    df_all.columns = ['feedstock', 'technology', 'product']
    
    technology_dic = {}
    for x, y in technology:
        technology_dic.setdefault(x, []).append(y) #{'t',[...]}
    product_dic = {}
    for x, y in product:
        product_dic.setdefault(x, []).append(y) #{'p',[...]}
    feedstock_dic = {}
    for x, y in feedstock:
        feedstock_dic.setdefault(x, []).append(y) #{'f',[...]}
        
    #df_f_list = df_f['feedstock'].tolist() #['','',...]
    #df_t_list = df_t['technology'].tolist() 
    #df_p_list = df_p['product'].tolist() 

    f_entity_list = existedEntityStrToList(feedstock_dic['f'])  #[[''],['',''],['']...]
    t_entity_list = existedEntityStrToList(technology_dic['t'])
    p_entity_list = existedEntityStrToList(product_dic['p']) 
    
    return f_entity_list, t_entity_list, p_entity_list

def flatten(list): #x = [[1,2],[3],[4]]  
  for i in list:
    for j in i:
      yield j  #list(flatten(x)) = [1,2,3,4] or tuple(flatten(x)) = (1,2,3,4)
      
      

HOST = "https://api.dbpedia-spotlight.org/en/annotate?"

def getAnnotations(text, confidence):#, support):
    try:
        annotations = spotlight.annotate(HOST, text, confidence)#, support)
    except spotlight.SpotlightException:
        return None
    return annotations

def getDBpediaResources(text):
    resources = []
    annotations = getAnnotations(text)
    for a in annotations:
        resources.append(a['URI'])
    return resources

def readCsv(file_path):
    ann_entity_pd = pd.read_csv(file_path, header = None)
    ann_entity = ann_entity_pd[0].values.tolist()
    
    return ann_entity

def compare(list_a, list_b):
    matches = []
    
    for i in list_a:
        if i in list_b and i not in matches:
            matches.append(i)
            
    return len(matches)

def getDBpediaAnnotation(file_dir, confidence): #get the entities annotated by DBpedia

    DB_entities_list = []
    for i in os.listdir(file_dir):
        print(file_dir + i)
        try:
            txt = readFile(file_dir + i)
            DB_annotation = getAnnotations(txt, confidence) #a list of dictionaries

            DB_entities = []
            for i in DB_annotation:
                if DB_annotation is not None:
                    for j in DB_annotation:
                        entity = j['surfaceForm'].lower()
                        if entity not in DB_entities and entity not in stop_words and entity.isnumeric() == False:
                            DB_entities.append(entity)
            DB_entities_list.append(DB_entities)
            time.sleep(5)
        except:
            return DB_entities_list

#get embeddings for all entities
def entitiesToVecsDic(entity_list, model):
    entity_vec = {}  
    
    for entity in entity_list:
        try:
            if len(entity) == 2:
                key = ' '.join(entity)
                entity_vec[key] = (model[entity[0]] + model[entity[1]]) / 2
            elif len(entity) == 3:
                key = ' '.join(entity)
                entity_vec[key] = (model[entity[0]] + model[entity[1]] + model[entity[2]]) / 3
            elif len(entity) == 1:
                key = entity[0]
                entity_vec[key] = model[entity[0]]
            else:
                pass
        except:
            pass 
    
    return entity_vec

#extract all arrays from dict to form the array contains all entities
def dictVecToArray(vec_dic):
    vec_list = []
    
    for i in vec_dic.keys():
        vec_list.append(vec_dic[i])

    vec_array = np.array(vec_list)

    return vec_array
    
#get similarity for 2 dicts of entity vectors    
def similarityMatrixForEntities(dict1, dict2):
    dict1_len = len(dict1.keys()) 
    dict2_len = len(dict2.keys()) 
    
    similarity_matrix = np.zeros(shape=(dict1_len, dict2_len)) 
    dict1_entity_vec_list =  [i for i in dict1.keys()]
    dict2_entity_vec_list =  [i for i in dict2.keys()]
    for i in range(dict1_len):
        for j in range(dict2_len):
            similarity_matrix[i][j] = cosine_similarity(dict1[dict1_entity_vec_list[i]].reshape(1, -1), dict2[dict2_entity_vec_list[j]].reshape(1, -1))
         
    return similarity_matrix


###############################################################################
annotated_entity_csv_path_origin = 'F://ThesisProject//data//annotatedEntities.csv'
annotated_entity_csv_path_refined = 'F://ThesisProject//data//annotatedEntities_refinedList.csv'

file_dir_tit_abs_db = 'F://ThesisProject//data//biorefinery_txt_titleAbstract_ann_DB//'
file_dir_ann = 'F://ThesisProject//data//ann//annotation_txt//'
file_dir_abs = 'F://ThesisProject//data//biorefinery_txt_abstract//'
file_dir_tit = 'F://ThesisProject//data//biorefinery_txt_title//'
file_dir_tit_abs = 'F://ThesisProject//data//biorefinery_txt_titleAbstract//'
file_dir_tit_abs_content = 'F://ThesisProject//data//biorefinery_txt_titleAbstractBody//'

file_dir = file_dir_tit_abs
csv_path = annotated_entity_csv_path_refined

ann_entity_map = generatCatogoryEntityList(csv_path)
f_entity_list, t_entity_list, p_entity_list = getCategoryEntitylist(ann_entity_map)

entity_list_fp = f_entity_list + p_entity_list
entity_list_t = t_entity_list
entity_list_f = f_entity_list
entity_list_p = p_entity_list
entity_list_ftp = f_entity_list + t_entity_list + p_entity_list

ann_entity = []
for i in entity_list_ftp:
    if len(i) > 1:
        x = ' '.join([j for j in i])
        ann_entity.append(x)
    else:
        ann_entity.append(i[0])

###############################################################################get all tokens for the data set
#all_sentences = []
#for i in os.listdir(file_dir):
#    abstract = readFile(file_dir + i)
#    sentences = absToSentence(abstract)
#    all_sentences.append(sentences) 
#
######get all sentences
#all_sentences = list(flatten(all_sentences))
#
######get all tokens
#all_tokens_unique = []
#all_tokens_filtered_unique = []
#all_tokens = []
#all_tokens_filtered = []
#stop_words = set(stopwords.words('english'))
#tokenizer = RegexpTokenizer(r'\w+')
#for i in all_sentences:
#    tokens_filtered = []
#    #tokens = nltk.word_tokenize(i)  #doesn't remove punctuation
#    tokens = tokenizer.tokenize(i)  #removes punctuation
#    for w in tokens:
#        if w not in stop_words and w.isnumeric() == False:
#            tokens_filtered.append(w)
#    
#    all_tokens.append(tokens)
#    all_tokens_filtered.append(tokens_filtered)
#  
#for i in list(flatten(all_tokens)): #find unique tokens
#    if i not in all_tokens_unique:
#        all_tokens_unique.append(i)
#    
#for i in list(flatten(all_tokens_filtered)): #find unique filtered tokens
#    if i not in all_tokens_filtered_unique:
#        all_tokens_filtered_unique.append(i)
    

###############################################################################tf-idf to get important terms for docs     
#file_path = 'F://ThesisProject//data//ann//annotation_txt//'
##file_path = 'F://ThesisProject//data//biorefinery_txt_abstract//'
#bow_texts = []
#for i in os.listdir(file_path):
#    text = readFile(file_path + i)
#    bow_text = returnAllWords(text)
#    bow_texts.append(bow_text)
#    bow_texts_flat = list(flatten(bow_texts))
#    
#for i in bow_texts_flat:
#    wordSet = {k:None for k in bow_texts_flat}
#
#word_dicts = []
#for i in range(len(bow_texts)) :
#    word_dict = dict.fromkeys(wordSet, 0) 
#    for word in bow_texts[i]:
#        word_dict[word] += 1
#    word_dicts.append(word_dict)
#    
#
#pd_word_dicts = pd.DataFrame([x for x in word_dicts])
#
#
#tfidf_bow = []
#idfs = computeIDF(word_dicts)
#
#for i in range(len(bow_texts)):
#    tf_bow = computeTF(word_dicts[i], bow_texts[i])
#    tfidf_bow.append(computeTFIDF(tf_bow, idfs))
#
#pd_tfidf_bow = pd.DataFrame([x for x in tfidf_bow])
#
#all_top_tfidf = []
#for i in range(len(bow_texts)):
#    top5_tfidf = pd_tfidf_bow.loc[i].nlargest(5).index.tolist()
#    for j in top5_tfidf:
#        if j not in all_top_tfidf:
#            all_top_tfidf.append(j)
#
#all_top_word_fre = []
#for i in range(len(bow_texts)):
#    top5_word = pd_word_dicts.loc[i].nlargest(5).index.tolist()
#    for j in top5_word:
#        if j not in all_top_word_fre:
#            all_top_word_fre.append(j)
#


###############################################################################get better parameter from DBpedia
#######compare 2 list to see the performance of the DBpedia extraction
#######find the best parameter "confidence" for using DBpedia to extract entities
##file_dir = 'F://ThesisProject//data//ann//annotation_txt//' #for test and experiment
#file_dir = 'F://ThesisProject//data//biorefinery_txt_abstract//'
#DB_entity_csv = 'F://ThesisProject//data//DB_entity_csv.csv'
#confidence = [0.4] #[0.3, 0.4, 0.5]
#scores = []
#
#for i in confidence:
#    score = []
#    
#    DB_entities = getDBpediaEntities(file_dir, i)
##    num_same_gold_DB_rec = compare(ann_entity, DB_entities)
##    num_DB_total_rec_entity = len(DB_entities)
##    alpha = num_gold_standard_entity/num_DB_total_rec_entity
##    score.append(alpha)
##    beta = num_same_gold_DB_rec/num_DB_total_rec_entity
##    score.append(beta)
##    gamma = num_same_gold_DB_rec/num_gold_standard_entity
##    score.append(gamma)
##    scores.append(score)
##    time.sleep(10)
    

############################################################################### DBpedia to get entities
#DB_entities = getDBpediaAnnotation(file_dir, 0.4)
#DB_entities_list = list(flatten(DB_entities))
#DB_entities_list_uniqu = []
#for i in DB_entities_list:
#    if i not in DB_entities_list_uniqu:
#        DB_entities_list_uniqu.append(i)
#
##df0 = pd.DataFrame(DB_entities_list_uniqu) array is better
#
######第一步   file_dir 改成 file_dir_tit_abs_db再跑
#a_all = np.load('F://ThesisProject//dbpedia_entities//a_all.npy')
#a_all_unique = np.load('F://ThesisProject//dbpedia_entities//a_all_unique.npy')
#
######第二步
#a76 = array(DB_entities_list_uniqu)
#a_all = np.concatenate((a75,a_all), axis=0)
##a_all_unique = np.unique(a_all, axis=0)



###############################################################################Noun-phrase chunker to get entities   
def extractNounPhrases(txt):
    #Define your grammar using regular expressions
    grammar = ('''NP: {<NN.*|JJ.*>*<NN.*>} # NP''') #<DT>?
    #grammar = ('''NP: {<JJ>*<NN.*>} # NP''') #<DT>?
    #grammar = """NP: {<DT>?<JJ>*<NN.*>+}
    #       RELATION: {<V.*>}
    #                 {<DT>?<JJ>*<NN.*>+}
    #       ENTITY: {<NN.*>}"""
    chunkParser = nltk.RegexpParser(grammar)
    
    sentences = nltk.sent_tokenize(txt)
    sentences = [[i.lower() for i in nltk.word_tokenize(sent)] for sent in sentences]
    sentences = [nltk.pos_tag(sent) for sent in sentences]
        
    noun_phrases_list = [[' '.join(leaf[0] for leaf in tree.leaves()) 
                          for tree in chunkParser.parse(sent).subtrees() 
                          if tree.label()=='NP'] 
                          for sent in sentences]  
    
    unique_noun_phrases_list = []
    for noun_phrases in noun_phrases_list:
        for j in noun_phrases:
            if j not in unique_noun_phrases_list:
                unique_noun_phrases_list.append(j)
    
    #for Nouns and Verbs            
    trees = []
    all_tagged = []
    nouns = []
    verbs = []
    for sentence in sentences:
       tree = chunkParser.parse(sentence)  
       trees.append(tree)
       
       iob_tagged = tree2conlltags(tree)
       all_tagged.append(iob_tagged)
       #pprint(iob_tagged)
       #filter out all those entities which contains stopwords to filter the terms to be candidate
       for i in range(len(iob_tagged)):
           if iob_tagged[i][1] == 'NN' or iob_tagged[i][1] == 'NNS':
               nouns.append(iob_tagged[i][0])
           elif iob_tagged[i][1] == 'VB' or iob_tagged[i][1] == 'VBD' or iob_tagged[i][1] == 'VBG' or iob_tagged[i][1] == 'VBN' or iob_tagged[i][1] == 'VBP' or iob_tagged[i][1] == 'VBZ':
               verbs.append(iob_tagged[i][0])
           else:
               pass
    
    return nouns, verbs, unique_noun_phrases_list
   
  
all_nouns = []    
all_verbs = []
all_noun_phrases = []     
for i in os.listdir(file_dir):
    txt = readFile(file_dir + i)
    print(file_dir + i)
    nouns, verbs, noun_phrases = extractNounPhrases(txt)  
    for j in noun_phrases:
        if j not in all_noun_phrases:
            all_noun_phrases.append(j) 
    for k in nouns:
        if k not in all_nouns:
            all_nouns.append(k)
    for l in verbs:
        if l not in all_verbs:
            all_verbs.append(l)
#num_same = compare(ann_entity, all_noun_phrases)   #115 same to ann_entity
#same = compare(all_noun_phrases, all_nouns)    #376 same element to all_noun_phrases  
#same1 = compare(ann_entity, all_nouns)          #85  same element to ann_entity

all_noun_phrases_array = array(all_noun_phrases)
all_noun_arry = array(all_nouns)
all_verb_array = array(all_verbs)

all_noun_verb_ = np.concatenate((all_noun_verb,all_noun_phrases_array), axis=0)
#a_all_unique = np.unique(a_all, axis=0)



#all_entities = []
#for i in all_noun_phrases:
#    if i not in all_entities:
#        all_entities.append(i)
#        
#for i in all_nouns:
#    if i not in all_entities:
#        all_entities.append(i)
#
#all_different_tfidf = []
#for i in all_top_tfidf:
#    if i not in all_entities:
#        all_entities.append(i)
#    else:
#        all_different_tfidf.append(i)











###############################################################################use own trained wordembedding to select terms
#model_save_path = 'F:/ThesisProject/wordEmbedding/model_trained_own_dataset.model'
#model_save_path_SG = 'F:/ThesisProject/wordEmbedding/model_trained_own_dataset_SG.model'
#model_trained_own_dataset =  Word2Vec.load(model_save_path_SG) 
##
model_google = word2vec.KeyedVectors.load_word2vec_format(datapath("GoogleNews-vectors-negative300.bin"), binary=True)
#model_glove = word2vec.KeyedVectors.load_word2vec_format(datapath("glove-wiki-gigaword-300.txt"), binary=False)
#model_glove100 = word2vec.KeyedVectors.load_word2vec_format(datapath("glove-wiki-gigaword-100.txt"), binary=False)	
#model_fasttext = word2vec.KeyedVectors.load_word2vec_format(datapath("fasttext-wiki-news-subwords-300"), binary=False)
#
model = model_google
similarity_threshold = 0.7

np_all_noun = np.load('F://ThesisProject//dbpedia_entities//np_all_noun.npy')
np_all_verb = np.load('F://ThesisProject//dbpedia_entities//np_all_verb.npy')
np_all_noun_phrases = np.load('F://ThesisProject//dbpedia_entities//np_all_noun_phrases.npy')
np_all_noun_verb_phrases = np.load('F://ThesisProject//dbpedia_entities//np_all_noun_verb_phrases.npy')

np_db_unique = np.load('F://ThesisProject//dbpedia_entities//np_db_unique.npy')
db_all_unique = np.load('F://ThesisProject//dbpedia_entities//db_all_unique.npy')


fp_entity_vec_dic = entitiesToVecsDic(entity_list_fp, model)
t_entity_vec_dic = entitiesToVecsDic(entity_list_t, model)
p_entity_vec_dic = entitiesToVecsDic(entity_list_p, model)
f_entity_vec_dic = entitiesToVecsDic(entity_list_f, model)

similarity_matrix_fp_t = similarityMatrixForEntities(fp_entity_vec_dic, t_entity_vec_dic)    # .mean() .min() .max() .var() .std()
similarity_matrix_f_f = similarityMatrixForEntities(f_entity_vec_dic, f_entity_vec_dic) 
similarity_matrix_p_p = similarityMatrixForEntities(f_entity_vec_dic, t_entity_vec_dic) 
similarity_matrix_t_t = similarityMatrixForEntities(t_entity_vec_dic, t_entity_vec_dic) 
similarity_matrix_f_p = similarityMatrixForEntities(f_entity_vec_dic, p_entity_vec_dic) 


similarity_matrix_f_f.mean() 
similarity_matrix_p_p.mean()
similarity_matrix_t_t.mean()
similarity_matrix_fp_t.mean()
similarity_matrix_f_p.mean()



fp_entity_vec_array = dictVecToArray(fp_entity_vec_dic)
t_entity_vec_array = dictVecToArray(t_entity_vec_dic)
f_entity_vec_array = dictVecToArray(f_entity_vec_dic)
p_entity_vec_array = dictVecToArray(p_entity_vec_dic)

fp_entity_vec_mean = fp_entity_vec_array.mean(axis=0)
t_entity_vec_mean = t_entity_vec_array.mean(axis=0)
f_entity_vec_mean = f_entity_vec_array.mean(axis=0)
p_entity_vec_mean = p_entity_vec_array.mean(axis=0)


np_all_noun_list = np_all_noun.tolist()
np_all_noun_list = [nltk.word_tokenize(i) for i in np_all_noun_list]
np_all_noun_vec = entitiesToVecsDic(np_all_noun_list, model)

np_all_noun_verb_phrases_list = np_all_noun_verb_phrases.tolist()
np_all_noun_verb_phrases_list = [nltk.word_tokenize(i) for i in np_all_noun_verb_phrases_list]
np_all_noun_verb_phrases_vec = entitiesToVecsDic(np_all_noun_verb_phrases_list, model)

db_all_unique_list = db_all_unique.tolist()
db_all_unique_list = [nltk.word_tokenize(i) for i in db_all_unique_list]
db_all_unique_vec = entitiesToVecsDic(db_all_unique_list, model)

np_db_unique_list = np_db_unique.tolist()
np_db_unique_list = [nltk.word_tokenize(i) for i in np_db_unique_list]
db_all_unique_vec = entitiesToVecsDic(np_db_unique_list, model)

#db_detected = []    ####to see entities only detected by DB not NP chunking
#for i in np_db_unique_list:
#    if i not in np_all_noun_verb_phrases_list:
#        db_detected.append(i)
        

expanded_fp_np = []
expanded_f_np = []
expanded_p_np = []
expanded_t_np = []
for i in np_all_noun_verb_phrases_vec.keys():
    cosine_similarity_fp = 0
    cosine_similarity_t = 0
    
    if cosine_similarity(fp_entity_vec_mean.reshape(1, -1), np_all_noun_verb_phrases_vec[i].reshape(1, -1)) > similarity_threshold and i not in expanded_fp_np:
        expanded_fp_np.append(i)
        cosine_similarity_fp = cosine_similarity(fp_entity_vec_mean.reshape(1, -1), np_all_noun_verb_phrases_vec[i].reshape(1, -1)) 
        similarity_fp = cosine_similarity_fp
        if cosine_similarity_fp > similarity_fp:
            similarity_fp = cosine_similarity_fp
    elif cosine_similarity(t_entity_vec_mean.reshape(1, -1), np_all_noun_verb_phrases_vec[i].reshape(1, -1)) > similarity_threshold and i not in expanded_t_np:
        expanded_t_np.append(i)
        cosine_similarity_t = cosine_similarity(t_entity_vec_mean.reshape(1, -1), np_all_noun_verb_phrases_vec[i].reshape(1, -1))
        similarity_t = cosine_similarity_t
        if cosine_similarity_t > similarity_t:
            similarity_t = cosine_similarity_t
    else:
        pass


for i in np_all_noun_verb_phrases_vec.keys():
    cosine_similarity_f = 0
    cosine_similarity_p = 0
    
    if cosine_similarity(f_entity_vec_mean.reshape(1, -1), np_all_noun_verb_phrases_vec[i].reshape(1, -1)) > similarity_threshold and i not in expanded_f_np and cosine_similarity(f_entity_vec_mean.reshape(1, -1), np_all_noun_verb_phrases_vec[i].reshape(1, -1)) > cosine_similarity(p_entity_vec_mean.reshape(1, -1), np_all_noun_verb_phrases_vec[i].reshape(1, -1)):
        expanded_fp_np.append(i)
        cosine_similarity_f = cosine_similarity(f_entity_vec_mean.reshape(1, -1), np_all_noun_verb_phrases_vec[i].reshape(1, -1)) 
        similarity_f = cosine_similarity_f
        if cosine_similarity_f > similarity_f:
            similarity_f = cosine_similarity_f
    elif cosine_similarity(p_entity_vec_mean.reshape(1, -1), np_all_noun_verb_phrases_vec[i].reshape(1, -1)) > similarity_threshold and i not in expanded_p_np and cosine_similarity(p_entity_vec_mean.reshape(1, -1), np_all_noun_verb_phrases_vec[i].reshape(1, -1)) > cosine_similarity(f_entity_vec_mean.reshape(1, -1), np_all_noun_verb_phrases_vec[i].reshape(1, -1)) :
        expanded_p_np.append(i)
        cosine_similarity_p = cosine_similarity(p_entity_vec_mean.reshape(1, -1), np_all_noun_verb_phrases_vec[i].reshape(1, -1))
        similarity_p = cosine_similarity_p
        if cosine_similarity_p > similarity_p:
            similarity_p = cosine_similarity_p
    else:
        pass


#f_100_expanded_t_np_goo = array(random.sample(expanded_t_np, 100))
#f_100_expanded_fp_np_goo = array(random.sample(expanded_fp_np, 100))
#
#expanded_t_np_gool = f_100_expanded_t_np_goo.tolist()
#expanded_fp_np_gool = f_100_expanded_fp_np_goo.tolist()

#import random
#expanded_fp_np_fastTextl = expanded_fp_np_fastText.tolist()
#expanded_fp_np_glo_100l = expanded_fp_np_glo_100.tolist()
#expanded_fp_np_glo_300l = expanded_fp_np_glo_300.tolist()
#expanded_fp_np_gool = expanded_fp_np_goo.tolist()
#expanded_fp_np_ownl = expanded_fp_np_own.tolist()
#expanded_t_np_fastTextl = expanded_t_np_fastText.tolist()
#expanded_t_np_glo_100l = expanded_t_np_glo_100.tolist()
#expanded_t_np_glo_300l = expanded_t_np_glo_300.tolist()
#expanded_t_np_gool = expanded_t_np_goo.tolist()
#expanded_t_np_ownl = expanded_t_np_own.tolist()
#
#f_50_expanded_fp_np_fastText = array(random.sample(expanded_fp_np_fastTextl, 50))
#f_50_expanded_fp_np_glo100 = array(random.sample(expanded_fp_np_glo_100l, 50))
#f_50_expanded_fp_np_glo300 = array(random.sample(expanded_fp_np_glo_300l, 50))
#f_50_expanded_fp_np_goo = array(random.sample(expanded_fp_np_gool, 50))
#f_50_expanded_fp_np_own = array(random.sample(expanded_fp_np_ownl, 50))
#f_50_expanded_t_np_fastText = array(random.sample(expanded_t_np_fastTextl, 50))
#f_50_expanded_t_np_glo100 = array(random.sample(expanded_t_np_glo_100l, 50))
#f_50_expanded_t_np_glo300 = array(random.sample(expanded_t_np_glo_300l, 50))
#f_50_expanded_t_np_goo = array(random.sample(expanded_t_np_gool, 50))
#f_50_expanded_t_np_own = array(random.sample(expanded_t_np_ownl, 50))





###############################################################################pretrained model:word2vec-google-news-300
###Load pretrained model (since intermediate data is not included, the model cannot be refined with additional data)
####model_google_path = 'F:/ThesisProject/wordEmbedding/GoogleNews_vectors_negative300/GoogleNews-vectors-negative300.bin'
####model_google = word2vec.KeyedVectors.load_word2vec_format(model_google_path, binary=True)
#
#model_google = api.load('word2vec-google-news-300')
#        
#cou_in_goo, cou_in_not_goo, inclu_goo, not_inclu_goo = findTokensNotIncludedInPretrained(model_google, all_tokens_unique)
#cou_in_filt_goo, cou_in_not_filt_goo, inclu_filt_goo, not_inclu_filt_goo  = findTokensNotIncludedInPretrained(model_google, all_tokens_filtered_unique)
#
#x_goo = []
#for i in not_inclu_goo:
#    if i in stop_words or i.isnumeric() == True:
#        x.append(i)
#
#cou_in_enti_goo, cou_in_not_enti_goo, inclu_enti_goo, not_inclu_enti_goo = findTokensNotIncludedInPretrained(model_google, entity_list_new)
#
#model_google.similarity('feed','fertilizer')




################################################################################pretrained model:glove-wiki-gigaword-300
##Load pretrained model
####model_glove_path = 'F:/ThesisProject/wordEmbedding/GoogleNews_vectors_negative300/xxxxxxxxx.bin'
####model_glove = word2vec.KeyedVectors.load_word2vec_format(word2vec.KeyedVectors.load_word2vec_format(model_glove, binary=True)
#
#model_glove = api.load('glove-wiki-gigaword-300')
#
###model_glove.similarity('feed','fertilizer')
###model_glove.most_similar("cat")
#
#cou_in_glo, cou_in_not_glo, inclu_glo, not_inclu_glo = findTokensNotIncludedInPretrained(model_glove, all_tokens_unique)
#cou_in_filt_glo, cou_in_not_filt_glo, inclu_filt_glo, not_inclu_filt_glo  = findTokensNotIncludedInPretrained(model_glove, all_tokens_filtered_unique)
#
#x_glo = []
#for i in not_inclu_glo:
#    if i in stop_words or i.isnumeric() == True:
#        x.append(i)
#        
#cou_in_enti, cou_in_not_enti, inclu_enti, not_inclu_enti = findTokensNotIncludedInPretrained(model_glove, entity_list_new)








###############################################################################pretrained model:glove-wiki-gigaword-300
##Load pretrained model
####model_fasttext_path = 'F:/ThesisProject/wordEmbedding/GoogleNews_vectors_negative300/xxxxxxxxx.bin'
####model_fasttext = word2vec.KeyedVectors.load_word2vec_format(word2vec.KeyedVectors.load_word2vec_format(model_fasttext_path, binary=True)
#
#model_fasttext = api.load('fasttext-wiki-news-subwords-300')
#
#model_fasttext.similarity('feed','fertilizer')
#model_fasttext.most_similar("cat")

#cou_in_fas, cou_in_not_fas, inclu_fas, not_inclu_fas = findTokensNotIncludedInPretrained(model_fasttext, all_tokens_unique)
#cou_in_filt_fas, cou_in_not_filt_fas, inclu_filt_fas, not_inclu_filt_fas = findTokensNotIncludedInPretrained(model_fasttext, all_tokens_filtered_unique)
#
#x_fas = []
#for i in not_inclu_fas:
#    if i in stop_words or i.isnumeric() == True:
#        x.append(i)
#
#cou_in_enti_fas, cou_in_not_enti_fas, inclu_enti_fas, not_inclu_enti_fas = findTokensNotIncludedInPretrained(model_fasttext, entity_list_new)

