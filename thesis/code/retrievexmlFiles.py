# -*- coding: utf-8 -*-
"""
Created on Mon Feb 11 10:59:18 2019
@author: Arlene

***crawl xml files from Elsevier's API***
***1.generate query
***2.get all DOIs which satisfy the query 
***3.extract DOIs and remove the redundant ones
***4.generate urls used for retrieve documents
***5.use urls to form xml trees
***6.save xml files with xml trees
"""

import os
import re
import time
import requests
from xml.etree import ElementTree as ET 

def generateQuery(searchQuery, year, count, key):
    #customize the query according to your needs
    
    query = "https://api.elsevier.com/content/search/sciencedirect?query=" + searchQuery + "&date=" + year + "&count=" + count + "&start=" + key
            
    return query
  
def getDOIs(apiKey, query):
    #Query through API to get document DOIs
    
    #input MYAPIKEY & query to get the documents' DOIs
    resp = requests.get(query,headers={'Accept' : 'text/xml', 
                                     'X-ELS-APIKey' : apiKey})
    DOIs = (resp.content).decode("utf-8")
    
    #result DOIs is a string
    return DOIs 
    
def extractUniqueDOIs(DOIs): 
    #extract the DOIs from a string of DOIs
    uniqueDOIs = []
    
    #regular expressions are easiest (and fastest)
    DOIsPattern = re.compile('10[.][0-9]{4,}(?:[.][0-9]+)*\/(?:(?!["&\'<>])\S)+') 
    duplicatedDOIs = DOIsPattern.findall(DOIs)
    
    #Get unique DOIs from the duplicated DOIs list 
    for i in duplicatedDOIs:
        if i not in uniqueDOIs:
            uniqueDOIs.append(i)
 
    return uniqueDOIs

def generateUrls(preUrl,uniqueDOIs):
    #use DOIs to generate Urls for crawling the documents
    urls = []
    
    for i in uniqueDOIs:
        i = preUrl + i
        urls.append(i)
        
    return urls

def getXMLTrees(urls, apiKey):
    #use the urls to get documents from Elsevier's API
    trees = []
    
    for i in range(len(urls)):
        resp = requests.get(urls[i], headers={'Accept' : 'text/xml',
                                        'X-ELS-APIKey' : apiKey})

        #form xml files   resp.text/resp.content (both are okay)
        root = ET.fromstring(resp.text) 
        tree = ET.ElementTree(root)
        trees.append(tree)

    # a list of trees that could be used to parse into text we want
    return trees

def getXMLTree(url, apiKey):
    #use the urls to get documents from Elsevier's API  

    resp = requests.get(url, headers={'Accept' : 'text/xml',
                                        'X-ELS-APIKey' : apiKey})

    #form xml files   resp.text/resp.content (both are okay)
    root = ET.fromstring(resp.text) 
    tree = ET.ElementTree(root)

    return tree





"""
Testing code
"""
#pipeline for crawling xmls through API and cache to local disk
preUrl = "https://api.elsevier.com/content/article/doi/"
MYAPIKEY = "APIKey" #please type in your own API keys getting from Elsevier
searchQuery = "biorefinery"
count = "100" #need to check permitted value from ScienceDirect's API  Elsevier Dev-> Look at use cases -> text mining -> APIs (SD)
years = ['1995', '1996', '1997', '1998', '1999', '2000', '2001', '2002', '2003', 
         '2004', '2005', '2006', '2007', '2008', '2009', '2010', '2011', '2012',
         '2013', '2014', '2015', '2016', '2017', '2018', '2019']
startNum = {'0':'Fir', '101':'Sec', '201':'Thr', '301':'For', '401':'Fif', '501':'Six', 
            '601':'Sev', '701':'Eig', '801':'Nin', '901':'Ten', '1001':'Ele', '1101':'twe',
            '1201':'Thrt', '1301':'Fort', '1401':'Fift', '1501':'Sixt', '1601':'Sevt', '1701':'Eigt',
            '1801':'Nint', '1901':'Twnt'}  

fileDir = "F://ThesisProject//data//biorefinery//"
start = time.time()

#create multiple folders named by a list 
for i in years:
    os.mkdir(os.path.join(fileDir,str(i)))


#save xml files into local cache
for key in startNum.keys():
    for year in years:
        query = generateQuery(searchQuery, year, count, key)
        DOIs = getDOIs(MYAPIKEY, query)
        print(DOIs)
        uniqueDOIs = extractUniqueDOIs(DOIs)
        urls = generateUrls(preUrl,uniqueDOIs)
        for i in range(len(urls)):
            f = open(fileDir + year + "//" + startNum[key] +"200_" + str(i) + ".xml", mode = "w", encoding = "utf-8")
            tree = getXMLTree(urls[i], MYAPIKEY)
             # save the contents into XML files
            tree.write(fileDir + year + "//" + startNum[key] + "200_" + str(i) + ".xml")
        
end = time.time()
timeElapsed = end - start       
        

