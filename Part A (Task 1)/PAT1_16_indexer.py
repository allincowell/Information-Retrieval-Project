import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

import os, pickle, re, sys
from collections import Counter

STOPWORDS = stopwords.words('english')
LEMMATIZER = WordNetLemmatizer()

def remove_punc(text):
    return re.sub(r'[^\w\s]', '', text)

def remove_stopwords(tokens):
    return [token for token in tokens if token not in STOPWORDS]

def lemmatizer(tokens):
    return Counter([LEMMATIZER.lemmatize(token) for token in tokens])

def parse(text):
    processed = {}
    start, end = text.index('<DOCNO>'), text.index('</DOCNO>')
    processed['doc_id'] = text[start + 7 : end].strip()
    
    start, end = text.index('<TEXT>'), text.index('</TEXT>')
    pp_text = text[start + 6 : end].strip().lower()
    pp_text = remove_punc(pp_text)
    pp_text_tokens = word_tokenize(pp_text)
    pp_text_tokens = remove_stopwords(pp_text_tokens)
    pp_text_tokens = lemmatizer(pp_text_tokens)
    processed['text'] = pp_text_tokens.copy()
    
    return processed

invertedIndex = {}
count = 0

corpus_path = sys.argv[1]

directories = []

for dirs in next(os.walk(corpus_path))[1]:
    if next(os.walk(corpus_path + '/'+  dirs))[2]:
        count += 1
        directories.append(corpus_path + '/' + dirs)
        
count, num_dir = 0, len(directories)

print("Creating Inverted Index....")
for directory in directories:
    count += 1
    
    for file_name in next(os.walk(directory))[2]:
        with open(directory + '/' + file_name) as f:
            processed = parse(f.read())

        for term in processed['text']:
            if term not in invertedIndex:
                invertedIndex[term] = []
            invertedIndex[term].append((processed['doc_id'], processed['text'][term]))
            
for term, postings_list in invertedIndex.items():
    postings_list.sort()

print("Saving Inverted Index....")
with open('model_queries_16.pth', 'wb') as f:
    pickle.dump(invertedIndex, f, protocol=pickle.HIGHEST_PROTOCOL)