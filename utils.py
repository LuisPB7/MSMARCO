import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import unicodedata
import csv
import numpy
import copy
import pickle
import random
import re
from collections import OrderedDict
from transformers import RobertaTokenizer, BertTokenizer

torch.manual_seed(0)
torch.backends.cudnn.deterministic=True
torch.backends.cudnn.benchmark=False

# Class to pad variable length sequence according to biggest sequence in the batch
class PadSequence:
    def __call__(self, batch):

        queries_and_pos_ids = torch.nn.utils.rnn.pad_sequence([x[0].squeeze(dim=0) for x in batch], batch_first=True)
        queries_and_pos_attn_masks = torch.nn.utils.rnn.pad_sequence([x[1].squeeze(dim=0) for x in batch], batch_first=True)
        queries_and_pos_type_ids = torch.nn.utils.rnn.pad_sequence([x[2].squeeze(dim=0) for x in batch], batch_first=True)

        queries_and_neg25_ids = torch.nn.utils.rnn.pad_sequence([x[3].squeeze(dim=0) for x in batch], batch_first=True)
        queries_and_neg25_attn_masks = torch.nn.utils.rnn.pad_sequence([x[4].squeeze(dim=0) for x in batch], batch_first=True)
        queries_and_neg25_type_ids = torch.nn.utils.rnn.pad_sequence([x[5].squeeze(dim=0) for x in batch], batch_first=True)

        queries_and_neg75_ids = torch.nn.utils.rnn.pad_sequence([x[6].squeeze(dim=0) for x in batch], batch_first=True)
        queries_and_neg75_attn_masks = torch.nn.utils.rnn.pad_sequence([x[7].squeeze(dim=0) for x in batch], batch_first=True)
        queries_and_neg75_type_ids = torch.nn.utils.rnn.pad_sequence([x[8].squeeze(dim=0) for x in batch], batch_first=True)

        return queries_and_pos_ids, queries_and_pos_attn_masks, queries_and_pos_type_ids, \
               queries_and_neg25_ids, queries_and_neg25_attn_masks, queries_and_neg25_type_ids, \
               queries_and_neg75_ids, queries_and_neg75_attn_masks, queries_and_neg75_type_ids

class TRECDataset(Dataset):
  
    def __init__(self, data, name=None):
        super(TRECDataset, self).__init__()
        if name:
            self.data = list(csv.DictReader(open('data/TSVs/{}.tsv'.format(name), encoding='utf-8'), \
                                        delimiter='\t', fieldnames=['query', 'pos_passage', 'neg25_passage', 'neg75_passage']))
        else:
            self.data = data
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-cased")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        query = self.data[index]['query']
        pos_passage = self.data[index]['pos_passage']
        neg25_passage = self.data[index]['neg25_passage']
        neg75_passage = self.data[index]['neg75_passage']

        query_and_pos_dict = self.tokenizer.encode_plus(query, text_pair=pos_passage, max_length=512, return_tensors='pt', add_special_tokens=True)
        query_and_neg25_dict = self.tokenizer.encode_plus(query, text_pair=neg25_passage, max_length=512, return_tensors='pt', add_special_tokens=True)
        query_and_neg75_dict = self.tokenizer.encode_plus(query, text_pair=neg75_passage, max_length=512, return_tensors='pt', add_special_tokens=True)

        return query_and_pos_dict['input_ids'], query_and_pos_dict['attention_mask'], query_and_pos_dict['token_type_ids'], \
               query_and_neg25_dict['input_ids'], query_and_neg25_dict['attention_mask'], query_and_neg25_dict['token_type_ids'], \
               query_and_neg75_dict['input_ids'], query_and_neg75_dict['attention_mask'], query_and_neg75_dict['token_type_ids']

    def setData(self,data):
        self.data = data

def getDataset():
    print("Generating new train dataset...")
    with open("data/top1000.train.dai.ranks.pkl", 'rb') as handle:
        top1000_train = pickle.load(handle)

    with open("data/queries.pkl", 'rb') as handle:
        queries = pickle.load(handle)

    with open("data/passages.pkl", 'rb') as handle:
        passages = pickle.load(handle)

    with open("data/qrels.train.pkl", 'rb') as handle:
        qrels = pickle.load(handle)

    data = []
    for qid in qrels:
        pos_pids = qrels[qid]
        try:
            lucene_passages = top1000_train[qid]
        except:
            continue
        negative25_passages = lucene_passages[0:25]
        negative75_passages = lucene_passages[25:]
        for pos_pid in pos_pids:
            try:
                neg25_pid = random.choice(negative25_passages)
                neg75_pid = random.choice(negative75_passages)
            except:
               continue
            while (neg25_pid in pos_pids) or (neg75_pid in pos_pids):
                neg25_pid = random.choice(negative25_passages)
                neg75_pid = random.choice(negative75_passages)
            data.append( OrderedDict([ ('query',queries[qid]),('pos_passage',passages[pos_pid]),\
                                       ('neg25_passage',passages[neg25_pid]), ('neg75_passage',passages[neg75_pid]) ]) )
    return data
