import pickle, json, sys

model_path, queries_path = sys.argv[1], sys.argv[2]

with open(model_path, "rb") as f:
    invertedIndex = pickle.load(f)
with open(queries_path) as f:
    query_list = json.load(f)

def intersect(list1, list2):
    m, n = len(list1), len(list2)
    i, j = 0, 0
    intersection = []
    
    while i < m and j < n:
        if list1[i][0] == list2[j][0]:
            intersection.append(list1[i])
            i += 1
            j += 1
        elif list1[i][0] < list2[j][0]:
            i += 1
        else:
            j += 1
            
    return intersection.copy()

query_docs_list = []

for query in query_list:
    query_num, query_tokens = query
    
    try:
        docs = invertedIndex[query_tokens[0]]
    
        for term in query_tokens:
            docs = intersect(docs, invertedIndex[term])
    except KeyError:
        docs = []
        
    query_docs_list.append([query_num, docs])

f = open('PAT1_16_results.txt', 'w')

for doc in query_docs_list:
    f.write(str(doc[0]) + ':')
    for doc, _ in doc[1]:
        f.write(doc + ' ')
    f.write('\n')
f.close()