import os
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch import nn
from torch.autograd import Variable
from utils import PadSequence, TRECDataset, getDataset
import pickle
from transformers import RobertaModel, BertModel
from variables import *
from adamod import AdaMod
from torch.optim.lr_scheduler import ReduceLROnPlateau

import pytorch_lightning as pl

torch.manual_seed(0)
torch.backends.cudnn.deterministic=True
torch.backends.cudnn.benchmark=False

class BERT(pl.LightningModule):

    def __init__(self):
        super(BERT, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-cased')
        self.lin = nn.Linear(768*2, 1)

    def forward(self, pos_passages, pos_passages_attn_masks, pos_passages_token_types, \
                neg25_passages, neg25_passages_attn_masks, neg25_passages_token_types, \
                neg75_passages, neg75_passages_attn_masks, neg75_passages_token_types):
        
        pos_passage_hidden_states, pos_cls = self.bert(pos_passages, attention_mask=pos_passages_attn_masks, token_type_ids=pos_passages_token_types)
        neg25_passage_hidden_states, neg25_cls = self.bert(neg25_passages, attention_mask=neg25_passages_attn_masks, token_type_ids=neg25_passages_token_types)
        neg75_passage_hidden_states, neg75_cls = self.bert(neg75_passages, attention_mask=neg75_passages_attn_masks, token_type_ids=neg75_passages_token_types)

        # Compute score between query and positive passage by max-pooling over final hidden states, linear layer, and sigmoid
        seq_len1 = pos_passage_hidden_states.size()[1]
        pos_passage_representation = nn.MaxPool2d((seq_len1,1))(pos_passage_hidden_states).squeeze(dim=1)
        pos_passage_representation = torch.cat([pos_passage_representation, pos_cls], dim=1)
        pos_score = self.lin(pos_passage_representation)

        # Compute score between query and negative 25 passage with the same process
        seq_len2 = neg25_passage_hidden_states.size()[1]
        neg25_passage_representation = nn.MaxPool2d((seq_len2,1))(neg25_passage_hidden_states).squeeze(dim=1)
        neg25_passage_representation = torch.cat([neg25_passage_representation, neg25_cls], dim=1)
        neg25_score = self.lin(neg25_passage_representation)

        # Compute score between query and negative 75 passage with the same process
        seq_len3 = neg75_passage_hidden_states.size()[1]
        neg75_passage_representation = nn.MaxPool2d((seq_len3,1))(neg75_passage_hidden_states).squeeze(dim=1)
        neg75_passage_representation = torch.cat([neg75_passage_representation, neg75_cls], dim=1)
        neg75_score = self.lin(neg75_passage_representation)
        
        return pos_score, neg25_score, neg75_score

    def hinge_loss(self, pos_scores, neg_scores):
        return torch.mean( torch.max(  torch.zeros_like(pos_scores).cuda(), torch.ones_like(pos_scores).cuda() - pos_scores + neg_scores  ) )

    def training_step(self, batch, batch_nb):
        # REQUIRED
        #b_size = batch[0].size()[0]
        outputs = self.forward(batch[0].cuda(), batch[1].cuda(), batch[2].cuda(), batch[3].cuda(), \
                         batch[4].cuda(), batch[5].cuda(), batch[6].cuda(), batch[7].cuda(), batch[8].cuda())
        pos_scores = outputs[0]
        neg25_scores = outputs[1]
        neg75_scores = outputs[2]
        loss = self.hinge_loss(pos_scores, neg25_scores) + self.hinge_loss(pos_scores, neg75_scores) + 0.25*self.hinge_loss(neg25_scores, neg75_scores)
        return {'loss': loss}

    def configure_optimizers(self):
        # REQUIRED
        optimizer = AdaMod(self.parameters(), lr=LR)
        #scheduler = ReduceLROnPlateau(optimizer, patience=LR_PATIENCE, min_lr=LR*0.1)
        return optimizer

    @pl.data_loader
    def train_dataloader(self):
        data = getDataset()
        return DataLoader(TRECDataset(data), shuffle=True, num_workers=8, batch_size=BATCH_SIZE, collate_fn=PadSequence())

    def on_epoch_start(self):
        new_data = getDataset()
        self.train_dataloader().dataset.setData(new_data)
