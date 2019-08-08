"""
***preprocess xml files***
***1.extract abstract/title/keywords from xml file
***2.clean the text
***3.sentence splitting, tokenization, lowercasing
"""

import os
import os.path
import re
import nltk
import csv
from xml.etree import ElementTree as ET
from functions import readFile, extractDOIs
from nltk.tokenize import sent_tokenize, word_tokenize


def removeNoAbstractFiles(fileList): #the fileList contains all the files needs to be examined
    newFileList = []
    noAbstractFilesList = []
    
    for i in range(len(fileList)):
        content = readFile(fileList[i])
        #print(fileList[i])
        root = ET.fromstring(content)
        abstract = root.find('.//{http://purl.org/dc/elements/1.1/}description')
        
        #don't remove those files don't have descriptions
        if abstract is None: 
            noAbstractFilesList.append(fileList[i]) #a list of xml files that don't have abstracts
            #os.remove(fileList[i])
        elif abstract.text is None:
            noAbstractFilesList.append(fileList[i])
        elif abstract.text == "Unknown":
            noAbstractFilesList.append(fileList[i])
        else:
            newFileList.append(fileList[i])
            
        #remove those files don't have descriptions  
#        if abstract is None:
#            noAbstractFilesList.append(fileList[i])
#            os.remove(fileList[i])
#        newFileList.append(fileList[i])
        
    return newFileList

def xmlTitle(xmlFileContent):
    #get title of xml files from xml contents
    
    #get the title of a xml article(a string)
    root = ET.fromstring(xmlFileContent)
    title = root.find('.//{http://purl.org/dc/elements/1.1/}title').text
    
    return title
    
def xmlAbstract(xmlFileContent):
    #get abstract of xml files from xml contents
    
    #get the abstract of a xml article(a string)
    root = ET.fromstring(xmlFileContent)
    #print("root fount")
    abstract = root.find('.//{http://purl.org/dc/elements/1.1/}description').text
    abstract = abstract.replace('ABSTRACT','')
    abstract = abstract.replace('Summary','')
    abstract = abstract.replace('Publisher','')
    abstract = abstract.strip('\n')
    abstract = abstract.replace('Abstract','') 
    
    return abstract

def xmlKeyword(xmlFileContent):
    #get keywords of xml files from xml contents
    keywordPerArt = []
    
    root = ET.fromstring(xmlFileContent) 
    
    for keywords in root.findall('.//{http://www.elsevier.com/xml/common/dtd}text'):
        if keywords.text != "Corresponding authors." and keywords.text != "Corresponding author.":
            keyword = keywords.text #keyword here is <string>
            keywordPerArt.append(keyword)
            
    return keywordPerArt
 
def xmlTitleWords(title) :
    #preprocess the title: tokenization, lowercasing
    titleWords = []
    
    titleWord = nltk.word_tokenize(title)
    titleWord = [word.lower() for word in titleWord]
    titleWords.append(titleWord)
    titleWords = [val for sublist in titleWords for val in sublist]

    return titleWords

def xmlAbstractWords(abstract) :
    #preprocess the abstract: sentence splitting, tokenization, lowercasing
    abstractWords = []
    
    abstractSenDict = nltk.sent_tokenize(abstract)
    
    for sentence in abstractSenDict:
        abstractWord = nltk.word_tokenize(sentence)
        abstractWord = [word.lower() for word in abstractWord]
        abstractWords.append(abstractWord)
    abstractWords = [val for sublist in abstractWords for val in sublist]

    return abstractWords
 
def xmlKeyWords(keywords) :
    #preprocess the keywords: tokenization, lowercasing
    keyWords = []
    
    for kword in keywords:
        keyWord = nltk.word_tokenize(kword)
        keyWord = [word.lower() for word in keyWord]
        keyWords.append(keyWord)
    #keyWords = [val for sublist in keyWords for val in sublist]
    
    return keyWords


def flatten(list): #x = [[1,2],[3],[4]]  
  for i in list:
    for j in i:
      yield j  #list(flatten(x)) = [1,2,3,4] or tuple(flatten(x)) = (1,2,3,4)
 

#"""
#Testing code
#"""
###pipeline for preprocessing the xml files  (single file)
##fileName = 'F://ThesisProject//docs//Eig200_0.xml'
##content = readFile(fileName)
##x = xmlTitle(content)
##x1 = xmlAbstract(content)
##x2 = xmlKeyword(content)
##y = xmlTitleWords(x)
##y1 = xmlAbstractWords(x1)
##y2 = xmlKeyWords(x2)



##pipeline for preprocessing the xml files  (list of files)
#
#fileDir = 'F://ThesisProject//data//biorefinery//'   #for trying: docs(no subdirectory) or Articles(with subdirectory)
#
#subFileDir = [] #a list of subdirectory of the current directory
#for i in os.listdir(fileDir):
#    i = fileDir + i
#    subFileDir.append(i)
##cleanedFileLists = removeNoAbstractFiles(subFileDir) #when there's no subFirDir under the current directory
#
#fileLists = [] #consist of a list of file list for the subdirectory   
#for i in range(len(subFileDir)):
#    fileList = os.listdir(subFileDir[i])
#    for j in range(len(fileList)):
#        fileList[j] = subFileDir[i] + "//" + fileList[j]
#    fileLists.append(fileList)
#
#cleanedFileLists = [] #consist of a list of file list for the subdirectory with only files with abstracts   
#for i in range(len(fileLists)):
#    cleanedFileList = removeNoAbstractFiles(fileLists[i])
#    cleanedFileLists.append(cleanedFileList)
#
#
#yearsInDir = os.listdir(fileDir)
#fileAbstractDicts = {}
#fileAbstractDictList = []
#savePath = 'F://ThesisProject//data//biorefinery_txt_titleAbstract//'
#
##save abstract to text    
#for i in range(len(cleanedFileLists)):
#    fileAbstractDict = {} 
#    
#    for j in range(len(cleanedFileLists[i])):
#        content = readFile(cleanedFileLists[i][j])
#        abstract = xmlAbstract(content)
#        #print(abstract)
#        title = xmlTitle(content)
#        txtFileName = cleanedFileLists[i][j].replace("//", "_")
#        pattern = re.findall('...[0-9]_.*?_.*[0-9]', txtFileName) 
#        txtFileName = pattern[0]
#        #print(txtFileName)
#        fileAbstractDict[cleanedFileLists[i][j]] = (txtFileName, title, abstract)
#        
#        completeName = os.path.join(savePath, txtFileName + ".txt")
#        file = open(completeName, "w",encoding = "utf-8")
#        file.write(title)
#        file.write('.')
#        file.write(abstract)
#        file.close()  
#        
#        
##    fileAbstractDictList.append(fileAbstractDict)
##    fileAbstractDicts[yearsInDir[i]] = fileAbstractDictList[i]



"""
Just to know how xml files are constructed
#xmlPath = 'F://ThesisProject//data//biorefinery//2019//Thr200_27.xml'
#content = readFile(xmlPath)
#
#root = ET.fromstring(content) 
#
##for para in root.findall('.//{http://www.elsevier.com/xml/common/dtd}para').text:
##    print()
#xsection = root.findall('.//{http://www.elsevier.com/xml/common/dtd}para')   #common/dtd:section     ja/dtd:body
#xbody = root.findall('.//{http://www.elsevier.com/xml/ja/dtd}').content
#xbody = root.findtext('.//{http://www.elsevier.com/xml/ja/dtd}')
#
#xsectiontext = []
#for i in xsection:
#    f = i.text
#    xsectiontext.append(f)
"""



###############################################################################get sections for an article and then generate it for wordembedding
import re   
    
def readFileBody(filename):
    with open(filename,'r',encoding = "utf-8") as f:
        content = f.read()
        content = content.split()
        content = ' '.join(content)
         
        return content

def findSectionsForAnArticle(xml_content): #assume the xml has sections
    pattern_for_section = '<ns\d:para id=*.+?>(.+?)<\/ns\d:para>'
    xml_body_uncleaned = re.findall(pattern_for_section, xml_content)
    
    return xml_body_uncleaned

def cleanContentsInXMLSections(xml_body_uncleaned):
    pattern_in_txt_tags = '<.+?>'
    pattern_in_txt_attr = '&#\d\d\d\d;'
    pattern_in_txt_extra_whitespace = '/^\s+|\s+$|\s+(?=\s)/'
    
    xml_body_cleaned = []

    for i in xml_body_uncleaned:
        cleaned_section = re.sub(pattern_in_txt_tags, '', i)   
        cleaned_section = re.sub(pattern_in_txt_extra_whitespace, '', cleaned_section)
        cleaned_section = re.sub(pattern_in_txt_attr, '', cleaned_section)
        xml_body_cleaned.append(cleaned_section)

    return xml_body_cleaned

def xmlSectionsToSentencesList(xml_body_cleaned):
    xml_sentences_an_article = []
    
    
    for i in xml_body_cleaned:
        sentence = [] 
        sentences = nltk.sent_tokenize(i)
        for sent in sentences:
            sent_word = nltk.word_tokenize(sent)
            sent_word = ' '.join( word.lower() for word in sent_word)
            sentence.append(sent_word)
        xml_sentences_an_article.append(sentence)
        
    xml_sentences_an_article_list = list(flatten(xml_sentences_an_article))
    
    return xml_sentences_an_article_list




"""
##########for testing, get all sentences for one xml file
xml_file_path = 'F://ThesisProject//data//biorefinery//2019//Thr200_28.xml'
content = readFileBody(xml_file_path)
xml_body_uncleaned = findSectionsForAnArticle(content)
xml_body_cleaned = cleanContentsInXMLSections(xml_body_uncleaned)
xml_sentences_an_article_list = xmlSectionsToSentencesList(xml_body_cleaned)
"""


file_xml_dir = 'F://ThesisProject//data//biorefinery//'   #for trying: docs(no subdirectory) or Articles(with subdirectory)
file_path_tit_abs_body = 'F://ThesisProject//data//biorefinery_txt_titleAbstractBody//'

sub_file_dir = [] #a list of subdirectory of the current directory
for i in os.listdir(file_xml_dir):
    i = file_xml_dir + i
    sub_file_dir.append(i)

file_lists = [] #consist of a list of file list for the subdirectory   
for i in range(len(sub_file_dir)):
    file_list = os.listdir(sub_file_dir[i])
    for j in range(len(file_list)):
        file_list[j] = sub_file_dir[i] + "//" + file_list[j]
    file_lists.append(file_list)

file_xml_lists = list(flatten(file_lists))

num_art_has_content = 0  
for i in file_xml_lists:
    content = readFileBody(i)
    xml_body_uncleaned = findSectionsForAnArticle(content)
    xml_body_cleaned = cleanContentsInXMLSections(xml_body_uncleaned)
    xml_sentences_an_article_list = xmlSectionsToSentencesList(xml_body_cleaned)
    xml_an_article_content = ' '.join(i for i in xml_sentences_an_article_list)
    
    if xml_an_article_content != '':  #save sections(body) to text
        num_art_has_content += 1
        txt_file_name = i.replace('F://ThesisProject//data//biorefinery//', '')
        txt_file_name = txt_file_name.replace('.xml', 'content')
        txt_file_name = txt_file_name.replace("//", "_")
        complete_file_name = os.path.join(file_path_tit_abs_body, txt_file_name + ".txt")
        file = open(complete_file_name, "w",encoding = "utf-8")
        file.write(xml_an_article_content)
        file.close()
    




















