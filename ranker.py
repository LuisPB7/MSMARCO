import pickle
from RobertaModelTest import RobertaTest
from BertModelTest import BertTest
import torch
import csv
import operator
from torch.utils.data import Dataset, DataLoader
from transformers import RobertaTokenizer, BertTokenizer

torch.manual_seed(0)
torch.backends.cudnn.deterministic=True
torch.backends.cudnn.benchmark=False

#############################################

K = 1000 # Re-rank top K

with open("data/passages.pkl", 'rb') as handle:
    passages = pickle.load(handle)

with open("data/queries.pkl", 'rb') as handle:
    queries = pickle.load(handle)

class PadSequence:
    def __call__(self, batch):

        qids = [x[0] for x in batch]
        pids = [x[1] for x in batch]
        queries_and_pos_ids = torch.nn.utils.rnn.pad_sequence([x[2].squeeze(dim=0) for x in batch], batch_first=True)
        queries_and_pos_attn_masks = torch.nn.utils.rnn.pad_sequence([x[3].squeeze(dim=0) for x in batch], batch_first=True)
        queries_and_pos_token_types = torch.nn.utils.rnn.pad_sequence([x[4].squeeze(dim=0) for x in batch], batch_first=True)

        return qids, pids, queries_and_pos_ids, queries_and_pos_attn_masks, queries_and_pos_token_types

class TestDataset(Dataset):

    def __init__(self, lucene_candidates):
        super(LuceneDataset, self).__init__()
        self.data = []
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-cased")
        for qid in lucene_candidates:
            for pid in lucene_candidates[qid]:
                self.data.append((qid, pid))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        query = queries[self.data[index][0]]
        passage = passages[self.data[index][1]]
        passage_dict = self.tokenizer.encode_plus(query, text_pair=passage, max_length=256, return_tensors='pt')
        return self.data[index][0], self.data[index][1], \
               passage_dict['input_ids'], passage_dict['attention_mask'], passage_dict['token_type_ids']

model = BertTest().cuda()
model.load_state_dict(torch.load('bert-initial-weights.pt'), strict=True)
model = model.eval()

##############################################################################
with open("data/top1000.dev.dai.ranks.pkl", 'rb') as handle:
    top1000_dev = pickle.load(handle)

topK_dev_sorted_ranks = {} # Will contain qid:[pid1, pid2, etc] sorted in descending order of score
topK_dev_sorted_scores = {}

topK_dev = {}

for qid in top1000_dev:
    topK_dev[qid] = top1000_dev[qid][:K]

with torch.no_grad():
    with open('top{}.dev.sorted.tsv'.format(K), 'wt', encoding='utf-8') as out_file:
        tsv_writer = csv.writer(out_file, delimiter=' ')
        query_dict = {qid:{} for qid in topK_dev}
        data = DataLoader(TestDataset(topK_dev), shuffle=True, num_workers=8, batch_size=256, collate_fn=PadSequence())
        for batch in data:
            qids = batch[0]
            pids = batch[1]
            outputs = model(batch[2].cuda(), batch[3].cuda(), batch[4].cuda())
            for i, pred in enumerate(outputs):
                query_dict[qids[i]][pids[i]] = pred
        for qid in query_dict:
            sorted_pids = sorted(query_dict[qid].items(), key=operator.itemgetter(1), reverse=True)
            for i, tup in enumerate(sorted_pids):
                try:
                    topK_dev_sorted_ranks[qid].append(tup[0])
                except:
                    topK_dev_sorted_ranks[qid] = [tup[0]]
                topK_dev_sorted_scores["{}-{}".format(qid, tup[0])] = tup[1].item()
                tsv_writer.writerow([qid, 'Q0', tup[0], i+1, tup[1].item(), 'STANDARD'])

for qid in top1000_dev:
    top1000_dev[qid][:K] = topK_dev_sorted_ranks[qid]

with open("top{}.dev.sorted.ranks.pkl".format(K), 'wb') as f:
    pickle.dump(top1000_dev, f, protocol=pickle.HIGHEST_PROTOCOL)

#with open("top{}.dev.sorted.scores.pkl".format(K), 'wb') as f:
#    pickle.dump(topK_dev_sorted_scores, f, protocol=pickle.HIGHEST_PROTOCOL)

##############################################################################

#run_model(model, 32, dataset_train, dataset_test, two_fold)
