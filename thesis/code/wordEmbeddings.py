# -*- coding: utf-8 -*-
"""
Created on Sun May 19 13:13:55 2019

@author: Arlene
"""

import pandas as pd #for data handling
import nltk
import os
import random
import shutil
import csv
from collections import Counter
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import gensim
from gensim.models import Word2Vec
import gensim.models.keyedvectors as word2vec
import re  # For preprocessing
from time import time  # To time our operations
from collections import defaultdict  # For word frequency
import spacy  # For preprocessing

import gensim.downloader as api
from gensim.models import Word2Vec

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


def getTriplets(feedstock, technology, product): #f list, t list, p list -- > f_entity_list, t_entity_list, p_entity_list
    triplets = []
    
    for i in range(len(feedstock)):
        for j in range(len(technology)):
            for k in range(len(product)):
                triplet= (feedstock[i], technology[j], product[k])
                triplets.append(triplet)
    
    #return a list tuples like(['green', 'crops'], ['protein', 'extraction'], ['biogas'])   
    return  triplets   #all possible combination of tfp triplets

##all_tripples = getTripples(df_f_list,df_t_list,df_p_list)
##
##tripple_file = 'F://ThesisProject//results//tfp_tripples.tsv'
##with open(tripple_file, 'w', encoding = 'utf-8', newline = '') as f:
##    tsv_writer = csv.writer(f, delimiter = '\t', lineterminator = '\n')
##    for i in all_tripples:
##        tsv_writer.writerow([i[0] + ', '  + i[1] + ', ' + i[2]])

def getTripletsCooccurrenceForArticle(sentences, f_entity_list, t_entity_list, p_entity_list):
    entity_triplets_cooccurrence_an_article = []
    
    for i in range(len(sentences)):
        entity_cooccurrence = {}
        
        f_entity_recognized = matchEntity(sentences[i], f_entity_list)
        t_entity_recognized = matchEntity(sentences[i], t_entity_list)
        p_entity_recognized = matchEntity(sentences[i], p_entity_list)
        
        entity_cooccurrence[i] = f_entity_recognized + t_entity_recognized + p_entity_recognized
        if len(entity_cooccurrence[i]) == 3: 
            entity_triplets_cooccurrence_an_article.append([[i],entity_cooccurrence[i]])  #i is the sentence identifier
            
    return entity_triplets_cooccurrence_an_article  #each sentence has 3 list: f&t&p  


def getPairs(feedstock, product): # f list,p list
    pairs = []
    
    for i in range(len(feedstock)):
        for k in range(len(product)):
            pair = (feedstock[i], product[k])
            pairs.append(pair)
     
    #return a list of tuples like (['green', 'crops'], ['biogas'])    
    return  pairs #all possible combination of fp pairs     


def getPairsCooccurrenceForArticle(sentences, f_entity_list, p_entity_list):
    entity_pairs_cooccurrence_an_article = []
    
    for i in range(len(sentences)):
        entity_cooccurrence = {}
        
        f_entity_recognized = matchEntity(sentences[i], f_entity_list)
        p_entity_recognized = matchEntity(sentences[i], p_entity_list)
        
        entity_cooccurrence[i] = f_entity_recognized + p_entity_recognized
        if len(entity_cooccurrence[i]) == 2: 
            entity_pairs_cooccurrence_an_article.append([[i],entity_cooccurrence[i]])  #i is the sentence identifier
            
    return entity_pairs_cooccurrence_an_article  #each sentence has 2 list: f&p  
 
def getTechOccurrenceForArticle(sentences, t_entity_list):
    entity_tech_occurrence_an_article = []
    
    for i in range(len(sentences)):
        entity_occurrence = {}
        
        t_entity_recognized = matchEntity(sentences[i], t_entity_list)
        
        entity_occurrence[i] = t_entity_recognized 
        if len(entity_occurrence[i]) == 1:
            entity_tech_occurrence_an_article.append([[i],entity_occurrence[i]])  #i is the sentence identifier
            
    return entity_tech_occurrence_an_article  #each sentence has 1 list: t     


def flatten(list): #x = [[1,2],[3],[4]]  
  for i in list:
    for j in i:
      yield j  #list(flatten(x)) = [1,2,3,4] or tuple(flatten(x)) = (1,2,3,4)
 
def getKeysByValue(dict_of_elements, value_to_find):
    
    list_of_keys = list()
    list_of_items = dict_of_elements.items()
    for item  in list_of_items:
        if item[1] == value_to_find:
            list_of_keys.append(item[0])
    return  list_of_keys[0]

def findTokensNotIncludedInPretrained(model, unique_token_list):
    count_inc = 0
    count_not_inc = 0
    not_included = []
    included = []
    
    for i in unique_token_list:   
        try:
            model[i]
            count_inc += 1
            included.append(i)
        except:
            not_included.append(i)
            count_not_inc += 1
            pass
    
    return count_inc, count_not_inc, included, not_included



annotated_entity_csv_path_origin = 'F://ThesisProject//data//annotatedEntities.csv'
annotated_entity_csv_path_refined = 'F://ThesisProject//data//annotatedEntities_refinedList.csv'

file_dir_ann = 'F://ThesisProject//data//ann//annotation_txt//'
file_dir_abs = 'F://ThesisProject//data//biorefinery_txt_abstract//'
file_dir_tit = 'F://ThesisProject//data//biorefinery_txt_title//'
file_dir_tit_abs = 'F://ThesisProject//data//biorefinery_txt_titleAbstract//'
file_dir_tit_abs_content = 'F://ThesisProject//data//biorefinery_txt_titleAbstractBody//'

file_dir = file_dir_tit_abs
csv_path = annotated_entity_csv_path_refined

ann_entity_map = generatCatogoryEntityList(csv_path)
f_entity_list, t_entity_list, p_entity_list = getCategoryEntitylist(ann_entity_map)

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
#
#
#entity_list = f_entity_list + t_entity_list + p_entity_list
#entity_list_new = []
#for i in entity_list:
#    if len(i) > 1:
#        i = '_'.join(j for j in i)
#        entity_list_new.append(i)
#    else:
#        entity_list_new.append(i[0])
#
#entity_list_fp = f_entity_list + p_entity_list
#entity_list_t = t_entity_list




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








###############################################################################train my own word2vec model
"""
#######for title+abstract only, see the occurrences of the terms
import matplotlib.pyplot as plt
from collections import Counter

token_list_to_sort = all_tokens   #all_tokens_filtered, all_tokens

all_tokens_in_one_list = list(flatten(token_list_to_sort))
all_tokens_occurrence = Counter(all_tokens_in_one_list)
all_tokens_occurrence_sorted = all_tokens_occurrence.most_common()

term_occurs_once = 0
for i in all_tokens_occurrence_sorted:
    if i[1] == 3:
        term_occurs_once += 1

x_val = [x[0] for x in all_tokens_occurrence_sorted]
y_val = [x[1] for x in all_tokens_occurrence_sorted]
plt.plot(x_val,y_val)
plt.plot(x_val,y_val,'or')
plt.show()
"""

import logging  # Setting up the loggings to monitor gensim
logging.basicConfig(format="%(levelname)s - %(asctime)s: %(message)s", datefmt= '%H:%M:%S', level=logging.INFO)
stop_words = set(stopwords.words('english'))
tokenizer = RegexpTokenizer(r'\w+')

class MySentences(object):
    def __init__(self, dirname):
        self.dirname = dirname
 
    def __iter__(self):
        for fname in os.listdir(self.dirname):
            abstract = readFile(self.dirname + fname)
            sentences = absToSentence(abstract)
            for line in sentences:
                yield line.split()
 
    
##file_dir_ann      file_dir_tit_abs_content
#   
                
                
import multiprocessing
from gensim.corpora.wikicorpus import WikiCorpus
from gensim.models.word2vec import Word2Vec     
from gensim.models.fasttext import FastText          
                
wiki_data_path_1 = 'F:/ThesisProject/wordEmbedding/enwiki-20190220-pages-articles-multistream1.xml.bz2'  
wiki_data_path_2 = 'F:/ThesisProject/wordEmbedding/enwiki-20190220-pages-articles-multistream2.xml.bz2'  
#wiki_data_path_3 = 'F:/ThesisProject/wordEmbedding/enwiki-20190220-pages-articles-multistream3.xml.bz2'
#wiki_data_path_4 = 'F:/ThesisProject/wordEmbedding/enwiki-20190220-pages-articles-multistream4.xml.bz2'
#wiki_data_path_5 = 'F:/ThesisProject/wordEmbedding/enwiki-20190220-pages-articles-multistream5.xml.bz2'
##
##              
#wiki = WikiCorpus(wiki_data_path_1, 
#                  lemmatize=False, dictionary={})               
#                
#wiki2 = WikiCorpus(wiki_data_path_2, 
#                  lemmatize=False, dictionary={})   
#
##wiki3 = WikiCorpus(wiki_data_path_3, 
##                  lemmatize=False, dictionary={})   
##
##wiki4 = WikiCorpus(wiki_data_path_4, 
##                  lemmatize=False, dictionary={})   
##
##wiki5 = WikiCorpus(wiki_data_path_5, 
##                  lemmatize=False, dictionary={})   
#
#sentences = list(wiki.get_texts()) 
#sentences2 = list(wiki2.get_texts()) 
##sentences3 = list(wiki3.get_texts()) 
##sentences4 = list(wiki4.get_texts()) 
##sentences5 = list(wiki5.get_texts())  
#
#sentences_all = sentences + sentences2 #+ sentences3 + sentences4 + sentences5        
#                
#                
sentences_own_dataset = MySentences(file_dir_tit_abs_content) # a memory-friendly iterator
#model_trained_own_dataset = gensim.models.Word2Vec([i for i in sentences_own_dataset] + sentences_all , size = 300, window = 10, min_count = 1, workers = 10, iter = 10, sg = 0)  #training my own model
#model_trained_own_dataset = gensim.models.FastText([i for i in sentences_own_dataset] + sentences_all , size = 300, window = 10, min_count = 1, workers = 10, iter = 10, sg = 0)  #training my own model
model_trained_own_dataset = gensim.models.Word2Vec([i for i in sentences_own_dataset] + sentences_all , size = 300, window = 2, min_count = 1, workers = 10, iter = 10, sg = 0)  #training my own model

model_save_path = 'F:/ThesisProject/wordEmbedding/model_trained_own_dataset_window2_wiki_1_2.model'
model_trained_own_dataset.save(model_save_path)



#model.wv.save_word2vec_format('model.bin')
#
#model_trained_own_dataset = Word2Vec.load(model_save_path) 
#
#model_trained_own_dataset.similarity('feed','fertilizer')
#model_trained_own_dataset.most_similar("cat")
#
#cou_in_own, cou_in_not_own, inclu_own, not_inclu_own = findTokensNotIncludedInPretrained(model_trained_own_dataset, all_tokens_unique)
#cou_in_filt_own, cou_in_not_filt_own, inclu_filt_own, not_inclu_filt_own = findTokensNotIncludedInPretrained(model_trained_own_dataset, all_tokens_filtered_unique)
#
#x_own = []
#for i in not_inclu_own:
#    if i in stop_words or i.isnumeric() == True:
#        x_own.append(i)
#
#cou_in_enti_own, cou_in_not_enti_own, inclu_enti_own, not_inclu_enti_own = findTokensNotIncludedInPretrained(model_trained_own_dataset, entity_list_new)
#
#
#len(model_trained_own_dataset.wv.vectors)
#words = list(model_trained_own_dataset.wv.vocab)


##https://radimrehurek.com/gensim/models/deprecated/word2vec.html
#from gensim.models import Phrases
#bigram_transformer = gensim.models.Phrases(sentences)
#model = Word2Vec(bigram_transformer[sentences], size=100, ...)



##Visualize Word Embedding
#X = model_trained_own_dataset[model_trained_own_dataset.wv.vocab]
#pca = PCA(n_components=2) 
#result = pca.fit_transform(X)
#plt.scatter(result[:, 0], result[:, 1])
#
#words = list(model_trained_own_dataset.wv.vocab)
#for i, word in enumerate(words[:5]):
#	plt.annotate(word, xy=(result[i, 0], result[i, 1]))
    
#w1 = ['biorefinery'] 
#model.wv.most_similar(positive = w1, topn = 6)










###############################################################################train the model with wiki,corpus
               
                
                


              
                
                