import csv, math, sys

golden_results_file, results_file, query_text_path = sys.argv[1], sys.argv[2], sys.argv[3]

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

metrics = {}

precision_vals = precision(results, 10)
map_10 = 0.0
for query_num, prec in precision_vals:
    metrics[query_num] = []
    metrics[query_num].append(prec)
    map_10 += prec
map_10 = map_10 / len(precision_vals)

precision_vals = precision(results, 20)
map_20 = 0.0
for query_num, prec in precision_vals:
    metrics[query_num].append(prec)
    map_20 += prec
map_20 = map_20 / len(precision_vals)

ndcg_vals = ndcg(results, K= 10)
mndcg_10 = 0.0
for query_num, ndcg_val in ndcg_vals:
    metrics[query_num].append(ndcg_val)
    mndcg_10 += ndcg_val
mndcg_10 = mndcg_10 / len(ndcg_vals)

ndcg_vals = ndcg(results, K=20)
mndcg_20 = 0.0
for query_num, ndcg_val in ndcg_vals:
    metrics[query_num].append(ndcg_val)
    mndcg_20 += ndcg_val

mndcg_20 = mndcg_20 / len(ndcg_vals)

K = results_file[-5]
metrics_file = "PAT2_16_metrics_" + K + ".csv"

with open(query_text_path, 'r') as f:
    text = f.read()

ptr = 0
query_list = {}

while True:
    try:
        start, end = text.index('<num>', ptr), text.index('</num>', ptr)
        query_num = int(text[start + 5 : end])
        ptr = end + 7
        
        start, end = text.index('<title>', ptr), text.index('</title>', ptr)
        query_str = text[start + 7 : end]
        ptr = end + 8
        
        query_list[query_num] = query_str
        
    except ValueError:
        break

f = open(metrics_file, 'w')
f.write('Query ID, AP@10, AP@20, NDCG@10, NDCG@20\n')
for query_num, metric_vec in metrics.items():
    f.write(str(query_num) + ',' + query_list[query_num] + ',')
    for metric in metric_vec:
        f.write(str(metric) + ',')
    f.write('\n')

f.write('Average,,' + str(map_10) + ',' + str(map_20) + ',' + str(mndcg_10) + ',' + str(mndcg_20) + ',')
f.close()