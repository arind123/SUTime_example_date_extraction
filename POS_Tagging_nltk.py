# -*- coding: utf-8 -*-
"""
Created on Mon Oct 11 13:01:32 2021

@author: arind
"""
'''
pip install nltk
*open python in prompt
    import nltk
    nltk.download('punkt')
    nltk.download('averaged_perceptron_tagger')
pip install TextBlob
*open python in prompt
    import nltk
    nltk.download('brown')
'''


import nltk



def ie_preprocess(document):
    sentences = nltk.sent_tokenize(document)
    sentences = [nltk.word_tokenize(sent) for sent in sentences]
    sentences = [nltk.pos_tag(sent) for sent in sentences]
    return sentences
    
#document = 'Refinery ABC is planning for a shutdown from 1 jan 2020 to 1 June for over a period of 5 months'
document = 'Virgin Islands announced plans Monday to shut the 200,000-barrel-a-day facility and dismiss more than 250 workers just weeks after a federal crackdown over a series of pollution incidents'
#document = 'The Empire of Japan aimed to dominate Asia and the United states of America'
x = ie_preprocess(document)
x


