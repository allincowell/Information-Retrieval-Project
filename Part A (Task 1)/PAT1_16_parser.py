import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

import re, json, sys

STOPWORDS = stopwords.words('english')
LEMMATIZER = WordNetLemmatizer()

def remove_punc(text):
    return re.sub(r'[^\w\s]', '', text)

def remove_stopwords(tokens):
    return [token for token in tokens if token not in STOPWORDS]

def lemmatizer(tokens):
    return [LEMMATIZER.lemmatize(token) for token in tokens]

def parse(text):
    pp_text = remove_punc(text)
    pp_text_tokens = word_tokenize(pp_text)
    pp_text_tokens = remove_stopwords(pp_text_tokens)
    pp_text_tokens = lemmatizer(pp_text_tokens)
    
    return pp_text_tokens

query_text_path = sys.argv[1]

with open(query_text_path, 'r') as f:
    text = f.read()

ptr = 0
query_list = []

while True:
    try:
        start, end = text.index('<num>', ptr), text.index('</num>', ptr)
        query_num = int(text[start + 5 : end])
        ptr = end + 7
        
        start, end = text.index('<title>', ptr), text.index('</title>', ptr)
        query_str = parse(text[start + 7 : end].lower())
        ptr = end + 8
        
        query_list.append([query_num, query_str])
        
    except ValueError:
        break

with open('queries_16.txt', 'w') as f:
    json.dump(query_list, f, indent = 4)