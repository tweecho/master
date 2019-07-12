# -*- coding: utf-8 -*-
"""
Created on Mon Jun  3 19:19:29 2019

@author: Arlene
"""
from __future__ import unicode_literals, print_function
file_dir = 'F://ThesisProject//data//ann//19_ann_NER.json'


import json
import logging
import sys
import spacy


def convertDoccanoToSpacy(file_dir):
    try:
        training_data = []
        lines = []
        with open(file_dir, 'r', encoding = 'utf-8') as f:
            lines = f.readlines()
    
        for line in lines:
            data = json.loads(line)
            text = data['text']
            entities = []
    
            for annotation in data['annotations']:
                #only a single point in text annotation.
                point_start = annotation['start_offset']
                point_end = annotation['end_offset']
                label = str(annotation['label']) #1: tech  2:prod                
                entities.append((point_start, point_end+1,label))
                
            training_data.append((text, {'entities': entities}))
        
        return training_data
    
    except Exception as e:
        logging.exception("Unable to process " + file_dir + "\n" + "error = " + str(e))
        return None

x = convertDoccanoToSpacy(file_dir)
#
#
#
#
#import spacy
#import random
#
#
#import plac
#import random
#from pathlib import Path
#from spacy.util import minibatch, compounding
#
##def trainSpacy(file_dir):
##    TRAIN_DATA = convertDoccanoToSpacy(file_dir);
##    nlp = spacy.blank('en')  # create blank Language class
##    # create the built-in pipeline components and add them to the pipeline
##    # nlp.create_pipe works for built-ins that are registered with spaCy
##    if 'ner' not in nlp.pipe_names:
##        ner = nlp.create_pipe('ner')
##        nlp.add_pipe(ner, last=True)
##
##    # add labels
##    for _, annotations in TRAIN_DATA:
##        for ent in annotations.get('entities'):
##            ner.add_label(ent[2])
##
##    # get names of other pipes to disable them during training
##    other_pipes = [pipe for pipe in nlp.pipe_names if pipe != 'ner']
##    with nlp.disable_pipes(*other_pipes):  # only train NER
##        optimizer = nlp.begin_training()
##        for itn in range(1):
##            print("Statring iteration " + str(itn))
##            random.shuffle(TRAIN_DATA)
##            losses = {}
##            for text, annotations in TRAIN_DATA:
##                nlp.update(
##                    [text],  # batch of texts
##                    [annotations],  # batch of annotations
##                    drop=0.2,  # dropout - make it harder to memorise data
##                    sgd=optimizer,  # callable to update weights
##                    losses=losses)
##            print(losses)
##    
##    #do prediction
##    doc = nlp("Samsing mobiles below $100")
##    print ("Entities= " + str(["" + str(ent.text) + "_" + str(ent.label_) for ent in doc.ents]))
##
##
##a = trainSpacy(file_dir)
#
#




TRAIN_DATA = convertDoccanoToSpacy(file_dir)


import pickle
import plac
import random
from pathlib import Path
import spacy
from spacy.util import minibatch, compounding


# New entity labels
# Specify the new entity labels which you want to add here
LABEL = ['1', '2']



def main(model=None, new_model_name='new_model', output_dir='F://ThesisProject//data//ann//', n_iter=20):
    """Setting up the pipeline and entity recognizer, and training the new entity."""
    if model is not None:
        nlp = spacy.load(model)  # load existing spacy model
        print("Loaded model '%s'" % model)
    else:
        nlp = spacy.blank('en')  # create blank Language class
        print("Created blank 'en' model")
    if 'ner' not in nlp.pipe_names:
        ner = nlp.create_pipe('ner')
        nlp.add_pipe(ner)
    else:
        ner = nlp.get_pipe('ner')

    for i in LABEL:
        ner.add_label(i)   # Add new entity labels to entity recognizer

    print(i)
    
    if model is None:
        optimizer = nlp.begin_training()
    else:
        optimizer = nlp.entity.create_optimizer()

    # Get names of other pipes to disable them during training to train only NER
    other_pipes = [pipe for pipe in nlp.pipe_names if pipe != 'ner']
    with nlp.disable_pipes(*other_pipes):  # only train NER
        for itn in range(n_iter):
            random.shuffle(TRAIN_DATA)
            losses = {}
            batches = minibatch(TRAIN_DATA, size=compounding(4., 32., 1.001))
            for batch in batches:
                texts, annotations = zip(*batch)
                nlp.update(texts, annotations, sgd=optimizer, drop=0.35,
                           losses=losses)
            print('Losses', losses)

    # Test the trained model
    test_text = "A new harvesting system and the fibre properties of reed canary grass (Phalaris arundinacea L.) makes this grass an interesting new raw material source for the pulp and paper industry in the Nordic countries. Pilot scale tests in Finland shows that high quality fine paper can successfully be produced from delayed harvested reed canary grass. Birch pulp can be replaced with reed canary grass pulp in fine paper furnish without any significant differences in the functional properties of paper."
    doc = nlp(test_text)
    print("Entities in '%s'" % test_text)
    for ent in doc.ents:
        print(ent.label_, ent.text)

    # Save model 
    if output_dir is not None:
        output_dir = Path(output_dir)
        if not output_dir.exists():
            output_dir.mkdir()
        nlp.meta['name'] = new_model_name  # rename model
        nlp.to_disk(output_dir)
        print("Saved model to", output_dir)

        # Test the saved model
        print("Loading from", output_dir)
        nlp2 = spacy.load(output_dir)
        doc2 = nlp2(test_text)
        for ent in doc2.ents:
            print(ent.label_, ent.text)


if __name__ == "__main__":
    plac.call(main)




#output_dir='F://ThesisProject//data//ann//'
#nlp2 = spacy.load(output_dir)
#test_text = "A new harvesting system and the fibre properties of reed canary grass (Phalaris arundinacea L.) makes this grass an interesting new raw material source for the pulp and paper industry in the Nordic countries. Pilot scale tests in Finland shows that high quality fine paper can successfully be produced from delayed harvested reed canary grass. Birch pulp can be replaced with reed canary grass pulp in fine paper furnish without any significant differences in the functional properties of paper."
#doc2 = nlp2(test_text)
#for ent in doc2.ents:
#    print(ent.label_, ent.text)
















