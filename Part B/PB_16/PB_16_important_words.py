import pickle, math,sys, csv
from collections import Counter

corpus_path, model_path, results_file = sys.argv[1], sys.argv[2], sys.argv[3]
#corpus_path, model_path, results_file ='./Data/en_BDNews24', './model_queries_16.pth', './PAT2_16_ranked_list_A.csv',

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

#Function to calculate tf-idx vector of documents
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

results = {}

with open(results_file) as f:
    reader = csv.reader(f)
    for query_num, doc in reader:
        query_num = int(query_num)
        if query_num not in results:
            results[query_num] = []
        results[query_num].append(doc)

################## finding words #############################

def find_top_words():
    ranked_words = {}
    for query_num,docs in results.items():
        count = 10
        relevant_docs = set()
        for doc_id in results[query_num]:
            if (count<=0): break
            relevant_docs.add(doc_id)
            count -= 1

        sum_vector = Counter()
        for doc_id in relevant_docs:
            sum_vector.update(doc_vec_A[doc_id])
        
        sum_vector_dict = dict(sum_vector)
        
        for value in sum_vector_dict:
            sum_vector_dict[value] /= len(relevant_docs)
        
        sorted_vector = sorted(sum_vector_dict.items(),key = lambda item:item[1],reverse=True)
        words = ""
        for i in range(5):
            words += sorted_vector[i][0]
            words += ", "
        ranked_words[query_num] = words
    return ranked_words


f = open('PB_16_important_words.csv','w')
writer = csv.writer(f)
top_words_result = find_top_words()
for query_id,top_words in top_words_result.items():
    writer.writerow([str(query_id),top_words])
f.close()