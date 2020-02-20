import pickle

with open("data/qrels.dev.small.pkl", 'rb') as f:
    qrels = pickle.load(f)
    
with open("top1000.dev.sorted.ranks.pkl", 'rb') as f:
    ranks = pickle.load(f)

Q = len(qrels)
print("Total number of queries: {}".format(Q))
print("Number of queries considered: {}".format(len(ranks)))

# Calculate Mean Average Precision #

def PrecisionAtK(qid, k):
    relevants = qrels[qid]
    pids = ranks[qid][:k]
    number_of_relevants_retrieved = 0
    for pid in pids:
        if pid in relevants:
            number_of_relevants_retrieved += 1
    return number_of_relevants_retrieved/k

def AveragePrecision(qid):
    relevants = qrels[qid]
    soma = 0
    for pid in relevants:
        try:
            rank = ranks[qid].index(pid)+1
            soma += PrecisionAtK(qid, rank)
        except:
            continue
    return soma/len(relevants)

soma = 0
for qid in qrels:
    soma += AveragePrecision(qid)
print("MAP: {}".format(soma/Q))

# Calculate Mean Reciprocal Rank and MRR@10 #

def ReciprocalRank(qid, k):
    relevants = qrels[qid]
    try:
        pids = ranks[qid][:k]
        for i, pid in enumerate(pids):
            if pid in relevants:
                return 1/(i+1)
    except:
        return 0
    return 0

soma_mrr = 0
soma_mrr10 = 0
for qid in qrels:
    soma_mrr += ReciprocalRank(qid, 1000)
    soma_mrr10 += ReciprocalRank(qid, 10)
print("MRR: {}".format(soma_mrr/Q))
print("MRR@10: {}".format(soma_mrr10/Q))
