import pickle, math, os, json, re, sys

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

corpus_path, model_path, queries_path = sys.argv[1], sys.argv[2], sys.argv[3]

with open(model_path, "rb") as f:
    invertedIndex = pickle.load(f)

V = list(invertedIndex.keys())
df = {}

docs = set()

for term, postings_list in invertedIndex.items():
    df[term] = len(postings_list)
    for doc_id, _ in postings_list:
        docs.add(doc_id)
        
N = len(docs)

def calculate_tf_idf_vec(doc_vecs, N, variant="lnc"):
    
    for doc_id, vec in doc_vecs.items():
        if variant[0] == 'l':
            for t, tf in vec.items():
                vec[t] = 1 + math.log10(tf) if tf > 0 else 0
        elif variant[0] == 'a':
            max_tf = max(list(vec.values()))

            for t, tf in vec.items():
                vec[t] = 0.5 + (0.5 * tf / max_tf) #DOUBT

        if variant[1] == 'n':
            pass
        elif variant[1] == 't':
            for t in vec:
                if t in V:
                    vec[t] *= math.log10(N / df[t])
        elif variant[1] == 'p':
            for t in vec:
                if t in V:
                    vec[t] *= max(0, math.log10((N - df[t]) / df[t]))

        if variant[2] == 'c':
            norm = 0.0
            for t in vec:
                norm += vec[t] ** 2
            norm = math.sqrt(norm)

            for t in vec:
                vec[t] /= norm
                
        doc_vecs[doc_id] = vec

doc_vec_A, doc_vec_B, doc_vec_C = {}, {}, {}

for doc in docs:
    doc_vec_A[doc] = {}
    doc_vec_B[doc] = {}
    doc_vec_C[doc] = {}

for term, postings_list in invertedIndex.items():
    for doc_id, freq in postings_list:
        doc_vec_A[doc_id][term] = freq
        doc_vec_B[doc_id][term] = freq
        doc_vec_C[doc_id][term] = freq

calculate_tf_idf_vec(doc_vec_A, N, 'lnc')
calculate_tf_idf_vec(doc_vec_B, N, 'lnc')
calculate_tf_idf_vec(doc_vec_C, N, 'anc')

with open(queries_path) as f:
    query_list = json.load(f)

query_vec_A, query_vec_B, query_vec_C = {}, {}, {}

for query_num, query_tokens in query_list:
    vec = {}
    
    for term in query_tokens:
        if term not in vec:
            vec[term] = 0
        vec[term] += 1
    
    query_vec_A[query_num] = vec    
    query_vec_B[query_num] = vec    
    query_vec_C[query_num] = vec
    
calculate_tf_idf_vec(query_vec_A, N, 'ltc')
calculate_tf_idf_vec(query_vec_B, N, 'lpc')
calculate_tf_idf_vec(query_vec_C, N, 'apc')

def compare(query_vec, doc_vecs, anc_apc=False):
    ranked_list = {}
    
    for doc_id, doc_vec in doc_vecs.items():
        score = 0.0
        
        for t in query_vec:
            if t in doc_vec:
                score += query_vec[t] * doc_vec[t]
                
        ranked_list[doc_id] = score
        
        
    sorted_ranked_list = dict(sorted(ranked_list.items(), key = lambda x: x[1], reverse = True))
    
    sorted_ranked_list_top_50 = []
    
    count = 0
    for doc in sorted_ranked_list:
        sorted_ranked_list_top_50.append(doc)
        count += 1
        if count == 50:
            break
            
    return sorted_ranked_list_top_50

a = open('PAT2_16_ranked_list_A.csv', 'w')
b = open('PAT2_16_ranked_list_B.csv', 'w')
c = open('PAT2_16_ranked_list_C.csv', 'w')

for query_num, _ in query_list:
    ranked_list_A = compare(query_vec_A[query_num], doc_vec_A)
    ranked_list_B = compare(query_vec_B[query_num], doc_vec_B)
    ranked_list_C = compare(query_vec_C[query_num], doc_vec_C)
    
    for doc in ranked_list_A:
        a.write(str(query_num) + ',' + doc + '\n')
    for doc in ranked_list_B:
        b.write(str(query_num) + ',' + doc + '\n')
    for doc in ranked_list_C:
        c.write(str(query_num) + ',' + doc + '\n')
    
a.close()
b.close()
c.close()