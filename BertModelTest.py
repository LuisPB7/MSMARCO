import torch
from torch import nn
from torchvision import transforms, datasets
from torch.autograd import Variable
import pickle
from transformers import RobertaModel, BertModel

torch.manual_seed(0)
torch.backends.cudnn.deterministic=True
torch.backends.cudnn.benchmark=False

class BertTest(nn.Module):
    def __init__(self):
        super(BertTest, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-cased', output_hidden_states=False)
        self.lin = nn.Linear(768*2, 1)
    
    def forward(self, passages, passages_attn_masks, token_types):        
        passage_hidden_states, cls = self.bert(passages, attention_mask=passages_attn_masks, token_type_ids=token_types)
        seq_len = passage_hidden_states.size()[1]
        passage_representation = nn.MaxPool2d((seq_len,1))(passage_hidden_states).squeeze(dim=1)
        passage_representation = torch.cat([passage_representation, cls], dim=1)
        score = self.lin(passage_representation)     
        return score
