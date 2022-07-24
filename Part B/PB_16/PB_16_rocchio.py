import pickle, math, os, json, re, sys, csv

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

corpus_path, model_path, golden_results_file, results_file, queries_path = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5]
#corpus_path, model_path, queries_path ='./Data/en_BDNews24', './model_queries_16.pth', './queries_16.txt'
#golden_results_file, results_file = './Data/rankedRelevantDocList.csv', './PAT2_16_ranked_list_A.csv',

# load inverted index
with open(model_path, "rb") as f:
    invertedIndex = pickle.load(f)

# dimension of each vector
V = list(invertedIndex.keys())

# document frequency
df = {}


# calculate doc frequency and make a set of docs
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
                vec[t] = 0.5 + (0.5 * tf / max_tf) 

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

doc_vec_A = {}

for doc in docs:
    doc_vec_A[doc] = {}

for term, postings_list in invertedIndex.items():
    for doc_id, freq in postings_list:
        doc_vec_A[doc_id][term] = freq

calculate_tf_idf_vec(doc_vec_A, N, 'lnc')

with open(queries_path) as f:
    query_list = json.load(f)

query_vec_A = {}

for query_num, query_tokens in query_list:
    vec = {}
    
    for term in query_tokens:
        if term not in vec:
            vec[term] = 0
        vec[term] += 1
    
    query_vec_A[query_num] = vec    
    
calculate_tf_idf_vec(query_vec_A, N, 'ltc')    

def compare(query_vec, doc_vecs, anc_apc=False):
    ranked_list = {}
    
    for doc_id, doc_vec in doc_vecs.items():
        score = 0.0
        
        for t in doc_vec:
            if t in query_vec:
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

###################################################################################################

results = {}

with open(results_file) as f:
    reader = csv.reader(f)
    for query_num, doc in reader:
        query_num = int(query_num)
        if query_num not in results:
            results[query_num] = []
        results[query_num].append(doc)

golden_results = {}

with open(golden_results_file) as f:
    reader = csv.reader(f)
    header = next(reader)
    for query_num, doc, relevance in reader:
        query_num = int(query_num)
        if query_num not in golden_results:
            golden_results[query_num] = {}
        golden_results[query_num][doc] = int(relevance)

def rocchio_formula(query_vec, relevent_docs, non_relevant_docs, alpha, beta, gamma):
    qm = {}
    for term, value in query_vec.items():
        qm[term] = alpha*value

    for doc_id in relevent_docs:
        for term, value in doc_vec_A[doc_id].items():
            if term not in qm: qm[term] = 0
            qm[term] += beta*value/len(relevent_docs)
    
    for doc_id in non_relevant_docs:
        for term, value in doc_vec_A[doc_id].items():
            if term not in qm: qm[term] = 0
            qm[term] -= gamma*value/len(non_relevant_docs)
    
    return qm
    

def modify_query_vecs(alpha, beta, gamma, mode='RF'):
    modified_queries = {}
    for query_num, golden_docs in golden_results.items():
        relevant_docs = set()
        non_relevant_docs = set()
        # RF
        if mode == 'RF':
            count = 20
            for doc_id in results[query_num]:
                if (count<=0): break
                if (doc_id in golden_docs) and (golden_docs[doc_id]==2):
                    relevant_docs.add(doc_id)
                else: non_relevant_docs.add(doc_id)
                count -= 1

        # PSRF
        else:
            count = 10
            for doc_id in results[query_num]:
                if (count<=0): break
                relevant_docs.add(doc_id)
                count -= 1
        modified_queries[query_num] = rocchio_formula(query_vec_A[query_num], relevant_docs, non_relevant_docs, alpha, beta, gamma)
    
    return modified_queries


##################### functions for evaluation ###################################

def precision(results, K=10):
    precision_vals = []
    
    for query_num, doc_list in results.items():
        if query_num not in golden_results:
            precision_vals.append((query_num, 0))
            continue
            
        docs_retrieved, relevant_docs_retrieved = 0, 0
        average_precision = 0.0
        
        k=K
        for doc in doc_list:
            if k == 0:
                break
            k -= 1
            docs_retrieved += 1

            if doc in golden_results[query_num]:
                relevant_docs_retrieved += 1
                average_precision += relevant_docs_retrieved / docs_retrieved

        if relevant_docs_retrieved:
            average_precision /= relevant_docs_retrieved

        precision_vals.append((query_num, average_precision))
        
    return precision_vals

def ndcg(results, K=10):
    ndcg_vals = []
    
    for query_num, doc_list in results.items():
        if query_num not in golden_results:
            ndcg_vals.append((query_num, 0))
            continue
        
        golden_doc_list_relevance = list(golden_results[query_num].values())
        golden_doc_list_relevance.sort(reverse=True)
        golden_doc_list_relevance = golden_doc_list_relevance[:K]
        
        prev = golden_doc_list_relevance[0]
        
        for i in range(1, len(golden_doc_list_relevance)):
            golden_doc_list_relevance[i] /= math.log2(i + 1)
            golden_doc_list_relevance[i] += prev
            prev = golden_doc_list_relevance[i]
            
        doc_list = doc_list[:K]
        ndcg = 0.0
        
        if doc_list[0] in golden_results[query_num]:
            ndcg = golden_results[query_num][doc_list[0]]
        
        for i in range(1, len(doc_list)):
            if doc_list[i] in golden_results[query_num]:
                rel_score = golden_results[query_num][doc_list[i]]
                rel_score /= math.log2(i + 1)
                ndcg += rel_score
                
        ndcg /= golden_doc_list_relevance[-1]
        ndcg_vals.append((query_num, ndcg))
        
    return ndcg_vals

################## run system on modified queries #############################

def evaluate(alpha, beta, gamma, mode):
    modified_queries = modify_query_vecs(alpha,beta,gamma,mode)
    new_results = {}

    print('calculating ', mode,' alpha=',alpha,' beta=',beta,' gamma=',gamma,sep='')
    print('comparing queries with document vectors...')
    print('progress:', end=' ', flush=True)
    for query_num, query_vec in modified_queries.items():
        print('+',end='', flush=True)
        new_results[query_num] = compare(query_vec, doc_vec_A)
    print('')

    precision_vals = precision(new_results,20)
    map_20 = 0.0
    for _, prec in precision_vals:
        map_20 += prec
    map_20 /= len(precision_vals)

    ndcg_vals = ndcg(new_results, K=20)
    mndcg_20 = 0.0
    for _, ndcg_val in ndcg_vals:
        mndcg_20 += ndcg_val
    mndcg_20 /= len(ndcg_vals)

    return [map_20, mndcg_20]

f = open('PB_16_rocchio_RF_metrics.csv', 'w')
f.write('alpha,beta,gamma,mAP@20,NDCG@20\n')
map_20, mndcg_20 = evaluate(1,1,0.5,'RF')
f.write('1,1,0.5,'+str(map_20)+','+str(mndcg_20)+'\n')
map_20, mndcg_20 = evaluate(0.5,0.5,0.5,'RF')
f.write('0.5,0.5,0.5,'+str(map_20)+','+str(mndcg_20)+'\n')
map_20, mndcg_20 = evaluate(1,0.5,0,'RF')
f.write('1,0.5,0,'+str(map_20)+','+str(mndcg_20))
f.close()

f = open('PB_16_rocchio_PsRF_metrics.csv', 'w')
f.write('alpha,beta,gamma,mAP@20,NDCG@20\n')
map_20, mndcg_20 = evaluate(1,1,0.5,'PsRF')
f.write('1,1,0.5,'+str(map_20)+','+str(mndcg_20)+'\n')
map_20, mndcg_20 = evaluate(0.5,0.5,0.5,'PsRF')
f.write('0.5,0.5,0.5,'+str(map_20)+','+str(mndcg_20)+'\n')
map_20, mndcg_20 = evaluate(1,0.5,0,'PsRF')
f.write('1,0.5,0,'+str(map_20)+','+str(mndcg_20))
f.close()