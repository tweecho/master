# -*- coding: utf-8 -*-
"""
Created on Wed May 29 16:55:38 2019

@author: Arlene
"""

import numpy as np
import gensim.models.keyedvectors as word2vec
from gensim.test.utils import datapath
import pandas as pd
import os
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from sklearn.metrics.pairwise import cosine_similarity
from itertools import  compress
import itertools
from gensim.models import Word2Vec


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

def generatCatogoryEntityList(annotated_entity_csv_path): #csv file contains (category entity) for each row

    ann_entity_pd = pd.read_csv(annotated_entity_csv_path, header = None)
    ann_entity = ann_entity_pd[0].values.tolist()
    ann_entity_2 = ann_entity_pd[1].values.tolist()
    ann_entity_map = list(list(i) for i in zip(ann_entity, ann_entity_2))

    #return ['product', 'biogas'],['feedstock', 'green crops'], etc.
    return ann_entity_map

def existedEntityStrToList(entity_list):
    entity_lists = []

    for i in entity_list:
        i = nltk.word_tokenize(i)
        entity_lists.append(i)
    
    return entity_lists

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

def flatten(list): #x = [[1,2],[3],[4]]  
  for i in list:
    for j in i:
      yield j  #list(flatten(x)) = [1,2,3,4] or tuple(flatten(x)) = (1,2,3,4)


def removeSubsets(entity_list_in_a_sent):
    supersets = list(map(lambda a: list(filter(lambda x: len(a) < len(x) and set(a).issubset(x), entity_list_in_a_sent)),entity_list_in_a_sent))
    new_list = list(compress(entity_list_in_a_sent, list(map(lambda x: 0 if x else 1, supersets))))

    entity_list_return = []
    for i in new_list:
        i = ' '.join(m for m in i)
        entity_list_return.append(i)
        
    return entity_list_return

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

def getOneOccurrenceForArticle(sentences, entity_list):
    entity_tech_occurrence_an_article = []
    
    for i in range(len(sentences)):
        entity_occurrence = {}
        
        entity_recognized = matchEntity(sentences[i], entity_list)
        entity_occurrence[i] = entity_recognized

        if len(entity_occurrence[i]) == 1:
            entity_occurrence = existedEntityStrToList(entity_occurrence[i][0])
            entity_occurrence= removeSubsets(entity_occurrence)
            entity_tech_occurrence_an_article.append([[i],entity_occurrence])  #i is the sentence identifier
        else:
            pass
                  
    return entity_tech_occurrence_an_article  #each sentence has 1 entity list  

def extractNounPhrases(txt):
    #Define your grammar using regular expressions
    grammar = ('''NP: {<NN.*|JJ.*>*<NN.*>} # NP''') #<DT>?
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

    return noun_phrases_list


def articleNPs(txt): #return a dict(abstract) of dicts(sentences), a dict(sentence) contains NPs in a sentence
    noun_phrases_article = {}
    noun_phrases = extractNounPhrases(txt)
    
    for i in range(len(noun_phrases)):
        noun_phrases_article[i] = noun_phrases[i]

    return noun_phrases_article


def abstractEntityMatchList(an_abstract_dict, ftp_list):
    an_abstract_dict_filtered = {}   
    
    for i in an_abstract_dict.keys():
        sent_entity = existedEntityStrToList(an_abstract_dict[i])
        sent_entity = [i for i in sent_entity if i in ftp_list]
        s = sorted(sent_entity)
        sent_entity = list(s for s,_ in itertools.groupby(s))
        #print(sent_entity)
        an_abstract_dict_filtered[i] = sent_entity
        
        an_abstract_dict_filtered_cleaned = dict((k, v) for k, v in an_abstract_dict_filtered.items() if v)
        
    return an_abstract_dict_filtered_cleaned

def cosSim(entity_embed, f_entity_vec_mean,t_entity_vec_mean, p_entity_vec_mean):
    
    cos_f = cosine_similarity(entity_embed.reshape(1, -1),f_entity_vec_mean.reshape(1, -1))
    cos_t = cosine_similarity(entity_embed.reshape(1, -1),t_entity_vec_mean.reshape(1, -1))
    cos_p = cosine_similarity(entity_embed.reshape(1, -1),p_entity_vec_mean.reshape(1, -1))
    
    return cos_f,cos_t,cos_p

def getFTPCategory(entity_dic_vec): #input: entity vectors in a sencence (a dict)
    f = []
    t = []
    p = []
    dic = {}
    for i in entity_dic_vec.keys():
        x = cosSim(entity_dic_vec[i],f_entity_vec_mean,t_entity_vec_mean, p_entity_vec_mean)
        if x[0] > x[1] and x[0] > x[2]:
            f.append(i)
        elif x[1] > x[0] and x[1] > x[2]:
            t.append(i)
        elif x[2] > x[0] and x[2] > x[1]:
            p.append(i)
        elif x[0] == x[2] and x[0] > x[1]:
            f.append(i)
            p.append(i)
        else:
            pass
        
    dic["f"] = f 
    dic["t"] = t
    dic["p"] = p

    #{'f': ['green biomass'], 't': ['green biorefinery'], 'p': []}
    return dic

def getFTPDict(file_single_abst):
    vec_dic = {}
    ftp_dic = {}
    for i in file_single_abst.keys():
        vec_dic[i] = entitiesToVecsDic(file_single_abst[i], model)
        vec_dic = dict((k, v) for k, v in vec_dic.items() if v)
    for j in vec_dic.keys():
        ftp_dic[j] = getFTPCategory(vec_dic[j])
        ftp_dic[j] = dict((k, v) for k, v in ftp_dic[j].items() if v) #remove those ftp dict which has no element
        if len(ftp_dic[j].keys()) != 3: #if a sentence doesen't has ftp
            ftp_dic.pop(j)
    
    return ftp_dic


def getFTPDictGSEl(file_single_abst):
    vec_dic = {}
    ftp_dic = {}
    for i in file_single_abst.keys():
        f1 = []
        t1 = []
        p1 = []
        for j in file_single_abst[i]:
            if j in f_entity_list:
                j = ' '.join(j)
                f1.append(j)
            elif j in t_entity_list:
                j = ' '.join(j)
                t1.append(j)
            elif j in p_entity_list:
                j = ' '.join(j)
                p1.append(j)
            else:
                pass   
        vec_dic[i] = {'f':f1 ,'t':t1 ,'p':p1} 
        ftp_dic[i] = dict((k, v) for k, v in vec_dic[i].items() if v) #remove those ftp dict which has no element
        if len(ftp_dic[i].keys()) != 3: #if a sentence doesen't has ftp
            ftp_dic.pop(i)
    
    return ftp_dic    


def getTriplets(feedstock, technology, product): #f list, t list, p list -- > f_entity_list, t_entity_list, p_entity_list
    triplets = []
    
    for i in range(len(feedstock)):
        for j in range(len(technology)):
            for k in range(len(product)):
                triplet= (feedstock[i], technology[j], product[k])
                if triplet not in triplets and (triplet[2], triplet[1], triplet[0]) not in triplets and triplet[2]!=triplet[0]:
                    triplets.append(triplet)
    
    #return a list tuples like(['green', 'crops'], ['protein', 'extraction'], ['biogas'])   
    return  triplets   #all possible combination of tfp triplets


def tupleEachArticle(ftp_dict_for_article): #dict(abstract) of dicts(sentences) or dicts(f list, p list, t list) to dict(abstract) of lists(triple list)
    tuple_list = []
    for i in ftp_dict_for_article.keys():
        for j in ftp_dict_for_article[i].keys():
            for k in ftp_dict_for_article[i][j].keys():
                if k == "f":
                    feedstock = ftp_dict_for_article[i][j][k]
                elif k == 't':
                    technology = ftp_dict_for_article[i][j][k]
                else:
                    product = ftp_dict_for_article[i][j][k] 
                    tuple_list = getTriplets(feedstock, technology, product)
            
            ftp_dict_for_article[i][j] = tuple_list


    random_sample_dict_article = {}
    
    for i in ftp_dict_for_article.keys():
        triple_list = []
        for j in ftp_dict_for_article[i].keys():
            triple_list.append(ftp_dict_for_article[i][j]) 
        random_sample_dict_article[i] =  list(flatten(triple_list))  
        
    return random_sample_dict_article




np_all_noun_verb_phrases = np.load('F://ThesisProject//dbpedia_entities//np_all_noun_verb_phrases_new1.npy')

annotated_entity_csv_path_origin = 'F://ThesisProject//data//annotatedEntities.csv'
annotated_entity_csv_path_refined = 'F://ThesisProject//data//annotatedEntities_refinedList.csv'

#file_dir_tit_abs_db = 'F://ThesisProject//data//biorefinery_txt_titleAbstract_ann_DB//'
file_dir_ann = 'F://ThesisProject//data//ann//annotation_txt//'
file_dir_abs = 'F://ThesisProject//data//biorefinery_txt_abstract//'
file_dir_tit = 'F://ThesisProject//data//biorefinery_txt_title//'
file_dir_tit_abs = 'F://ThesisProject//data//biorefinery_txt_titleAbstract//'
file_dir_tit_abs_content = 'F://ThesisProject//data//biorefinery_txt_titleAbstractBody//'

file_dir = file_dir_tit_abs
csv_path = annotated_entity_csv_path_refined

#model_save_path = 'F:/ThesisProject/wordEmbedding/model_trained_own_dataset.model'
#model_save_path_SG = 'F:/ThesisProject/wordEmbedding/model_trained_own_dataset_SG.model'
#model_save_path_wiki_own = 'F:/ThesisProject/wordEmbedding/model_trained_own_dataset_wiki.model'
#model_save_path_wiki_own1_2 = 'F:/ThesisProject/wordEmbedding/model_trained_own_dataset_wiki_1_2.model'
#model_save_path_wiki_own_fastText = 'F:/ThesisProject/wordEmbedding/model_trained_own_dataset_wiki_fastText_1_2.model'
#model_save_path_wiki_own_window5_wiki_1_2 = 'F:/ThesisProject/wordEmbedding/model_trained_own_dataset_window5_wiki_1_2.model'
#model_save_path_wiki_own_window2_wiki_1_2 = 'F:/ThesisProject/wordEmbedding/model_trained_own_dataset_window2_wiki_1_2.model'
#model_trained_own_dataset =  Word2Vec.load(model_save_path)
#
##model_google = word2vec.KeyedVectors.load_word2vec_format(datapath("GoogleNews-vectors-negative300.bin"), binary=True)
##model_glove = word2vec.KeyedVectors.load_word2vec_format(datapath("glove-wiki-gigaword-300.txt"), binary=False)
##model_glove100 = word2vec.KeyedVectors.load_word2vec_format(datapath("glove-wiki-gigaword-100.txt"), binary=False)	
##model_fasttext = word2vec.KeyedVectors.load_word2vec_format(datapath("fasttext-wiki-news-subwords-300"), binary=False)
#
#model = model_trained_own_dataset
#
ann_entity_map = generatCatogoryEntityList(csv_path)
f_entity_list, t_entity_list, p_entity_list = getCategoryEntitylist(ann_entity_map)
#
##get the average value for f,t,p word lists' word embeddings 
#t_entity_vec_mean = dictVecToArray(entitiesToVecsDic(t_entity_list, model)).mean(axis=0)
#f_entity_vec_mean = dictVecToArray(entitiesToVecsDic(f_entity_list, model)).mean(axis=0)
#p_entity_vec_mean = dictVecToArray(entitiesToVecsDic(p_entity_list, model)).mean(axis=0)
#
#np_all_noun_verb_phrases_list = np_all_noun_verb_phrases.tolist()
#np_all_noun_verb_phrases_list = [nltk.word_tokenize(i) for i in np_all_noun_verb_phrases_list]
#np_all_noun_verb_phrases_vec = entitiesToVecsDic(np_all_noun_verb_phrases_list, model)
#
#
#similarity_threshold = 0.6
##expand the ftp list (make sure each entity is unique)
#expanded_ftp_np = []
#for i in np_all_noun_verb_phrases_vec.keys():
#    
#    if cosine_similarity(f_entity_vec_mean.reshape(1, -1), np_all_noun_verb_phrases_vec[i].reshape(1, -1)) > similarity_threshold and i not in expanded_ftp_np:
#        expanded_ftp_np.append(i)
#    elif cosine_similarity(t_entity_vec_mean.reshape(1, -1), np_all_noun_verb_phrases_vec[i].reshape(1, -1)) > similarity_threshold and i not in expanded_ftp_np:
#        expanded_ftp_np.append(i)
#    elif cosine_similarity(p_entity_vec_mean.reshape(1, -1), np_all_noun_verb_phrases_vec[i].reshape(1, -1)) > similarity_threshold and i not in expanded_ftp_np:
#        expanded_ftp_np.append(i)
#    else:
#        pass
#
#expanded_ftp_np_add = existedEntityStrToList(expanded_ftp_np) + f_entity_list + t_entity_list + p_entity_list
ftp_np = f_entity_list + t_entity_list + p_entity_list
#
#expanded_ftp_np_adds = []
#for i in expanded_ftp_np_add:
#    if i not in expanded_ftp_np_adds:
#        expanded_ftp_np_adds.append(i)










dir_tit = 'F://ThesisProject//data//biorefinery_txt_title//'
dir_ann = 'F://ThesisProject//data//ann//annotation_txt//'
dir_titabs = 'F://ThesisProject//data//biorefinery_txt_titleAbstract//'
file_dir = dir_titabs
file_list = os.listdir(file_dir)
num_abstract = len(file_list)

file_abs_dict = {}
file_abs_dict_filtered = {}
file_abs_dict_ftp = {}
for i in range(num_abstract):
    i = file_dir + file_list[i]
    print(i)
    abstract = readFile(i)
    file_abs_dict[i] = articleNPs(abstract)
    file_abs_dict_filtered[i] = abstractEntityMatchList(file_abs_dict[i],ftp_np)
    file_single_abst = file_abs_dict_filtered[i]
    file_abs_dict_ftp[i] = getFTPDictGSEl(file_single_abst)
    file_abs_dict_ftp_all = dict((k, v) for k, v in file_abs_dict_ftp.items() if v)
    GSEl = tupleEachArticle(file_abs_dict_ftp_all)
#    file_abs_dict_ftp[i] = getFTPDict(file_single_abst)
#    file_abs_dict_ftp_all = dict((k, v) for k, v in file_abs_dict_ftp.items() if v) #exclude those articles which doesn't has ftp occurs together
#
    



   
    
       
#trained_google = tupleEachArticle(file_abs_dict_ftp_all)

#trained_own = tupleEachArticle(file_abs_dict_ftp_all)
#trained_own_wiki = tupleEachArticle(file_abs_dict_ftp_all)
#trained_own_wiki_12 = tupleEachArticle(file_abs_dict_ftp_all)
#trained_goo = tupleEachArticle(file_abs_dict_ftp_all)
#trained_glo100 = tupleEachArticle(file_abs_dict_ftp_all)
#trained_glo300 = tupleEachArticle(file_abs_dict_ftp_all)
#trained_fastText = tupleEachArticle(file_abs_dict_ftp_all)
#trained_fastText_wiki_1 = tupleEachArticle(file_abs_dict_ftp_all)
#trained_window5_wiki_12 = tupleEachArticle(file_abs_dict_ftp_all)
#trained_window2_wiki_12 = tupleEachArticle(file_abs_dict_ftp_all)
#
#
#x1 = trained_fastText.keys()
#x2 = trained_glo100.keys()
#x3 = trained_goo.keys()
#x4 = trained_own_wiki_12.keys()
#x5 = trained_own_wiki_1.keys()
#x6 = trained_own.keys()
#x7 = trained_glo300.keys()
#x8 = trained_fastText_wiki_1_08.keys()
#x9 = trained_window5_wiki_12.keys()
#x10 = trained_window2_wiki_12.keys()
#x_key = []
#for i in x6:
#    if i in x1 and i in x2 and i in x3 and i in x4 and i in x5 and i in x6 and i in x8 and i in x9 and i in x10:
#       x_key.append(i) 
#        
#
#x_all_triple = {}
#for i in x_key:
#    x_all_triple[i] = {'trained_own':trained_own[i], 'trained_own_wiki': trained_own_wiki_1[i],
#                'trained_own_wiki_12': trained_own_wiki_12[i],'trained_goo': trained_goo[i],
#                'trained_glo100': trained_glo100[i], 'trained_glo300': trained_glo300[i], 
#                'trained_fastText': trained_fastText[i], 'trained_fastText_wiki_1_08':trained_fastText_wiki_1_08[i],
#                'trained_window5_wiki_12': trained_window5_wiki_12[i], 'trained_window2_wiki_12':trained_window2_wiki_12[i]}
#
#x_key = []
#for i in x6:
#    if i in x3 and i in x10:
#        x_key.append(i)
#        
#x_all_triple = {}
#for i in x_key:
#    x_all_triple[i] = {'trained_own':trained_own[i], 'trained_goo': trained_goo[i],
#                 'trained_window2_wiki_12':trained_window2_wiki_12[i]}



import random

random_sample_entity_triplets_articles = []
for i in file_abs_dict_ftp_all_NP.keys():
    random_sample_entity_triplets_articles.append(i)

random.seed(6)
random_samples = random.sample(random_sample_entity_triplets_articles,22) 

import shutil
for i in random_samples:
    shutil.copy(i, "F://ThesisProject//data//has_triples_baseline_method1//")

random_sample_dict = {}
for i in random_samples:
    random_sample_dict[i] = file_abs_dict_ftp_all_NP[i]
 
  
     

 
    
    
    
    
    


       