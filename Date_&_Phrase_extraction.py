# -*- coding: utf-8 -*-
"""
Created on Mon Oct 11 17:02:23 2021

@author: arind
"""

##Subject Verb Object extraction
## Subect Object Dependency: https://suttipong-kull.medium.com/how-to-extract-subject-verb-and-object-by-nlp-4149323a7d7d

import en_core_web_sm
from collections.abc import Iterable
import pandas as pd
import json
from sutime import SUTime
import numpy as np
from nltk import tokenize

# use spacy small model
nlp = en_core_web_sm.load()

#We can update the following list of dependencies as per our requirements,
# for reference use above mentioned website
# dependency markers for subjects
#SUBJECTS = {"nsubj", "nsubjpass", "csubj", "csubjpass","agent","pobj","expl"}
SUBJECTS = {"nsubj", "nsubjpass", "csubj", "csubjpass", "agent", "expl"}
# dependency markers for objects
OBJECTS = {"dobj", "iobj", "prep","dative","attr","oprd","acomp","pobj"}
#OBJECTS = {"dobj", "dative", "attr", "oprd"}
# POS tags that will break adjoining items
BREAKER_POS = {"CCONJ", "VERB"}
# words that are negations
NEGATIONS = {"no", "not", "n't", "never", "none"}


# does dependency set contain any coordinating conjunctions?
def contains_conj(depSet):
    return "and" in depSet or "or" in depSet or "nor" in depSet or \
           "but" in depSet or "yet" in depSet or "so" in depSet or "for" in depSet


# get subs joined by conjunctions
def _get_subs_from_conjunctions(subs):
    more_subs = []
    for sub in subs:
        # rights is a generator
        rights = list(sub.rights)
        rightDeps = {tok.lower_ for tok in rights}
        if contains_conj(rightDeps):
            more_subs.extend([tok for tok in rights if tok.dep_ in SUBJECTS or tok.pos_ == "NOUN"])
            if len(more_subs) > 0:
                more_subs.extend(_get_subs_from_conjunctions(more_subs))
    return more_subs


# get objects joined by conjunctions
def _get_objs_from_conjunctions(objs):
    more_objs = []
    for obj in objs:
        # rights is a generator
        rights = list(obj.rights)
        rightDeps = {tok.lower_ for tok in rights}
        if contains_conj(rightDeps):
            more_objs.extend([tok for tok in rights if tok.dep_ in OBJECTS or tok.pos_ == "NOUN"])
            if len(more_objs) > 0:
                more_objs.extend(_get_objs_from_conjunctions(more_objs))
    return more_objs


# find sub dependencies
def _find_subs(tok):
    head = tok.head
    while head.pos_ != "VERB" and head.pos_ != "NOUN" and head.head != head:
        head = head.head
    if head.pos_ == "VERB":
        subs = [tok for tok in head.lefts if tok.dep_ == "SUB"]
        if len(subs) > 0:
            verb_negated = _is_negated(head)
            subs.extend(_get_subs_from_conjunctions(subs))
            return subs, verb_negated
        elif head.head != head:
            return _find_subs(head)
    elif head.pos_ == "NOUN":
        return [head], _is_negated(tok)
    return [], False


# is the tok set's left or right negated?
def _is_negated(tok):
    parts = list(tok.lefts) + list(tok.rights)
    for dep in parts:
        if dep.lower_ in NEGATIONS:
            return True
    return False


# get all the verbs on tokens with negation marker
def _find_svs(tokens):
    svs = []
    verbs = [tok for tok in tokens if tok.pos_ == "VERB"]
    for v in verbs:
        subs, verbNegated = _get_all_subs(v)
        if len(subs) > 0:
            for sub in subs:
                svs.append((sub.orth_, "!" + v.orth_ if verbNegated else v.orth_))
    return svs


# get grammatical objects for a given set of dependencies (including passive sentences)
def _get_objs_from_prepositions(deps, is_pas):
    objs = []
    for dep in deps:
        if dep.pos_ == "ADP" and (dep.dep_ == "prep" or (is_pas and dep.dep_ == "agent")):
            objs.extend([tok for tok in dep.rights if tok.dep_  in OBJECTS or
                         (tok.pos_ == "PRON" and tok.lower_ == "me") or
                         (is_pas and tok.dep_ == 'pobj')])
    return objs


# get objects from the dependencies using the attribute dependency
def _get_objs_from_attrs(deps, is_pas):
    for dep in deps:
        if dep.pos_ == "NOUN" and dep.dep_ == "attr":
            verbs = [tok for tok in dep.rights if tok.pos_ == "VERB"]
            if len(verbs) > 0:
                for v in verbs:
                    rights = list(v.rights)
                    objs = [tok for tok in rights if tok.dep_ in OBJECTS]
                    objs.extend(_get_objs_from_prepositions(rights, is_pas))
                    if len(objs) > 0:
                        return v, objs
    return None, None


# xcomp; open complement - verb has no suject
def _get_obj_from_xcomp(deps, is_pas):
    for dep in deps:
        if dep.pos_ == "VERB" and dep.dep_ == "xcomp":
            v = dep
            rights = list(v.rights)
            objs = [tok for tok in rights if tok.dep_ in OBJECTS]
            objs.extend(_get_objs_from_prepositions(rights, is_pas))
            if len(objs) > 0:
                return v, objs
    return None, None


# get all functional subjects adjacent to the verb passed in
def _get_all_subs(v):
    verb_negated = _is_negated(v)
    subs = [tok for tok in v.lefts if tok.dep_ in SUBJECTS and tok.pos_ != "DET"]
    if len(subs) > 0:
        subs.extend(_get_subs_from_conjunctions(subs))
    else:
        foundSubs, verb_negated = _find_subs(v)
        subs.extend(foundSubs)
    return subs, verb_negated

# get all functional subjects adjacent to the verb passed in
def _get_all_subs_ver_2(v):
    verb_negated = _is_negated(v)
    subs = [tok for tok in v.lefts if tok.dep_ in SUBJECTS and tok.pos_ != "DET"]
    subs.append([tok for tok in v.rights if tok.dep_ in SUBJECTS and tok.pos_ != "DET"])
    if len(subs) > 0:
        subs.extend(_get_subs_from_conjunctions(subs))
    else:
        foundSubs, verb_negated = _find_subs(v)
        subs.extend(foundSubs)
    return subs, verb_negated


# find the main verb - or any aux verb if we can't find it
def _find_verbs(tokens):
    verbs = [tok for tok in tokens if _is_non_aux_verb(tok)]
    if len(verbs) == 0:
        verbs = [tok for tok in tokens if _is_verb(tok)]
    return verbs

# find all the verbs
def _find_all_verbs(tokens):
    #verbs = [tok for tok in tokens if _is_non_aux_verb(tok)]
    #if len(verbs) == 0:
    verbs = [tok for tok in tokens if _is_verb(tok)]
    return verbs

# is the token a verb?  (excluding auxiliary verbs)
def _is_non_aux_verb(tok):
    return tok.pos_ == "VERB" and (tok.dep_ != "aux" and tok.dep_ != "auxpass")


# is the token a verb?  (excluding auxiliary verbs)
def _is_verb(tok):
    return tok.pos_ == "VERB" or tok.pos_ == "AUX"


# return the verb to the right of this verb in a CCONJ relationship if applicable
# returns a tuple, first part True|False and second part the modified verb if True
def _right_of_verb_is_conj_verb(v):
    # rights is a generator
    rights = list(v.rights)

    # VERB CCONJ VERB (e.g. he beat and hurt me)
    if len(rights) > 1 and rights[0].pos_ == 'CCONJ':
        for tok in rights[1:]:
            if _is_non_aux_verb(tok):
                return True, tok

    return False, v


# get all objects for an active/passive sentence
def _get_all_objs(v, is_pas):
    # rights is a generator
    rights = list(v.rights)

    objs = [tok for tok in rights if tok.dep_ in OBJECTS or (is_pas and tok.dep_ == 'pobj')]
    objs.extend(_get_objs_from_prepositions(rights, is_pas))

    #potentialNewVerb, potentialNewObjs = _get_objs_from_attrs(rights)
    #if potentialNewVerb is not None and potentialNewObjs is not None and len(potentialNewObjs) > 0:
    #    objs.extend(potentialNewObjs)
    #    v = potentialNewVerb

    potential_new_verb, potential_new_objs = _get_obj_from_xcomp(rights, is_pas)
    if potential_new_verb is not None and potential_new_objs is not None and len(potential_new_objs) > 0:
        objs.extend(potential_new_objs)
        v = potential_new_verb
    if len(objs) > 0:
        objs.extend(_get_objs_from_conjunctions(objs))
    return v, objs


# return true if the sentence is passive - at he moment a sentence is assumed passive if it has an auxpass verb
def _is_passive(tokens):
    for tok in tokens:
        if tok.dep_ == "auxpass":
            return True
    return False


# resolve a 'that' where/if appropriate
def _get_that_resolution(toks):
    for tok in toks:
        if 'that' in [t.orth_ for t in tok.lefts]:
            return tok.head
    return None


# simple stemmer using lemmas
def _get_lemma(word: str):
    tokens = nlp(word)
    if len(tokens) == 1:
        return tokens[0].lemma_
    return word


# print information for displaying all kinds of things of the parse tree
def printDeps(toks):
    for tok in toks:
        print(tok.orth_, tok.dep_, tok.pos_, tok.head.orth_, [t.orth_ for t in tok.lefts], [t.orth_ for t in tok.rights])


# expand an obj / subj np using its chunk
def expand(item, tokens, visited):
    if item.lower_ == 'that':
        temp_item = _get_that_resolution(tokens)
        if temp_item is not None:
            item = temp_item

    parts = []

    if hasattr(item, 'lefts'):
        for part in item.lefts:
            if part.pos_ in BREAKER_POS:
                break
            if not part.lower_ in NEGATIONS:
                parts.append(part)

    parts.append(item)

    if hasattr(item, 'rights'):
        for part in item.rights:
            if part.pos_ in BREAKER_POS:
                break
            if not part.lower_ in NEGATIONS:
                parts.append(part)

    if hasattr(parts[-1], 'rights'):
        for item2 in parts[-1].rights:
            if item2.pos_ == "DET" or item2.pos_ == "NOUN":
                if item2.i not in visited:
                    visited.add(item2.i)
                    parts.extend(expand(item2, tokens, visited))
            break

    return parts


# convert a list of tokens to a string
def to_str(tokens):
    if isinstance(tokens, Iterable):
        return ' '.join([item.text for item in tokens])
    else:
        return ''


# find verbs and their subjects / objects to create SVOs, detect passive/active sentences
def findSVOs(tokens):
    svos = []
    is_pas = _is_passive(tokens)
    #verbs = _find_verbs(tokens)
    verbs = _find_all_verbs(tokens)
    visited = set()  # recursion detection
    for v in verbs:
        subs, verbNegated = _get_all_subs(v)
        # hopefully there are subs, if not, don't examine this verb any longer
        if len(subs) > 0:
            isConjVerb, conjV = _right_of_verb_is_conj_verb(v)
            if isConjVerb:
                v2, objs = _get_all_objs(conjV, is_pas)
                for sub in subs:
                    for obj in objs:
                        objNegated = _is_negated(obj)
                        if is_pas:  # reverse object / subject for passive
                            svos.append((to_str(expand(obj, tokens, visited)),
                                         "!" + v.lemma_ if verbNegated or objNegated else v.lemma_, to_str(expand(sub, tokens, visited))))
                            svos.append((to_str(expand(obj, tokens, visited)),
                                         "!" + v2.lemma_ if verbNegated or objNegated else v2.lemma_, to_str(expand(sub, tokens, visited))))
                        else:
                            svos.append((to_str(expand(sub, tokens, visited)),
                                         "!" + v.lower_ if verbNegated or objNegated else v.lower_, to_str(expand(obj, tokens, visited))))
                            svos.append((to_str(expand(sub, tokens, visited)),
                                         "!" + v2.lower_ if verbNegated or objNegated else v2.lower_, to_str(expand(obj, tokens, visited))))
            else:
                v, objs = _get_all_objs(v, is_pas)
                for sub in subs:
                    if len(objs) > 0:
                        for obj in objs:
                            objNegated = _is_negated(obj)
                            if is_pas:  # reverse object / subject for passive
                                svos.append((to_str(expand(obj, tokens, visited)),
                                             "!" + v.lemma_ if verbNegated or objNegated else v.lemma_, to_str(expand(sub, tokens, visited))))
                            else:
                                svos.append((to_str(expand(sub, tokens, visited)),
                                             "!" + v.lower_ if verbNegated or objNegated else v.lower_, to_str(expand(obj, tokens, visited))))
                    else:
                        # no obj - just return the SV parts
                        svos.append((to_str(expand(sub, tokens, visited)),
                                     "!" + v.lower_ if verbNegated else v.lower_,))

    return svos


## Date extract and use above mentioned functions for phrase extraction
test_case = '''Expert says that only those with built-in flexibility will survive.
'''

#test_case = 'I have a dog. I got in last tuesday. It was 09/10/2021. My birthday 14 th march came early. 20 th sep i got there.I had march in  my calender.'
#test_case = 'I was born in march 1992'
sutime = SUTime(mark_time_ranges=True, include_range=True)

'''
### Result using a loop
result = json.dumps(sutime.parse(test_case), sort_keys=True, indent=4)

#Take the entire sentence where the date occcurs.

jdata = json.loads(result)

for d in jdata:
    print(d)
    for key, value in d.items():
        if key == 'start':
            start = value
        if key == 'end':
            end = value
        if key == 'text':
            date_string = value
            
    full_stop='.'
    list_index_full_stop = []
    for i in range(len(test_case)):
        if (test_case[i] == full_stop):
            list_index_full_stop.append(i)
    #print(start,end)            
    arr_index_full_stop = np.array(list_index_full_stop)
    #print(arr_index_full_stop)
    sent_start = arr_index_full_stop[arr_index_full_stop <= start].max()
    #print(sent_start)
    if max(list_index_full_stop)==end:
        sent_end = end
    else:
        sent_end = arr_index_full_stop[arr_index_full_stop >= end].min()
    #print(sent_end)
    sentence = test_case[sent_start+1:sent_end]
    #extract subject_verb_objects
    tokens = nlp(sentence)
    svos = findSVOs(tokens)
    #print(svos)
    
    print('Date String: {} \nSentence: {}\nSubject_Verb_Object: {}\n\n'.format(date_string,sentence,svos))
'''
#create DataFrame
list_sentences = list(tokenize.sent_tokenize(test_case))


sent_dict_list = [] 
#### Create extracted dataframe dictionary    
for sent in list_sentences:
    #print(sent)
    result = json.dumps(sutime.parse(sent), sort_keys=True, indent=4)
    jdata = json.loads(result)
    #print(jdata)
    if len(jdata)==0:
        tokens = nlp(sent)
        svos = findSVOs(tokens)
        result_dict = {'Sentence': sent,
                       'date_text':'NA',
                       'timex_value':'NA',
                       'date_type':'NA',
                       'date_value':'NA',
                       'svos':svos}
        print('\nDate String: {}   Timex_value: {}   Type: {}  Value: {}  \nSentence: {}\nSubject_Verb_Object: {}\n\n'.format('NA','NA','NA','NA',sent,svos))
        sent_dict_list.append(result_dict)
    else:
        for dt in jdata:
            val = str(dt.get('value'))
            if not val.startswith('P'):#Check for valid Date e.g PWD,P1D,P5M etc
                key_list = dt.keys()
                
                if 'text' in key_list:
                    date_text = dt.get("text")
                else:
                    date_text = 'NA'
                if 'timex-value' in key_list:
                    timex_value = dt.get("timex-value")
                else:
                    timex_value = 'NA'
                if 'type' in key_list:
                    date_type = dt.get('type')
                else:
                    date_type = 'NA'
                if 'value' in key_list:
                    date_value = dt.get('value')
                else:
                    date_value = 'NA'
                        
                tokens = nlp(sent)
                svos = findSVOs(tokens)
                result_dict = {'Sentence': sent,
                           'date_text':date_text,
                           'timex_value':timex_value,
                           'date_type':date_type,
                           'date_value':date_value,
                           'svos':svos}
                print('\nDate String: {}   Timex_value: {}   Type: {}  Value: {}  \nSentence: {}\nSubject_Verb_Object: {}\n\n'.format(date_text,timex_value,date_type,date_value,sent,svos))
                sent_dict_list.append(result_dict)
                        
        
sentence_df = pd.DataFrame(sent_dict_list)
      
